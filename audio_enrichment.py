"""
배치 오디오 특성 예측 + DB 저장 모듈
M2 TF-IDF + GBR 모델을 사용하여 누락된 오디오 특성을 채워넣음
"""
from fastapi import APIRouter
from sqlalchemy import text
from database import SessionLocal
from pydantic import BaseModel
from typing import List, Optional, Dict
from pathlib import Path
import json
import logging
import threading
from datetime import datetime

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["Audio Enrichment"])

AUDIO_FEATURES = [
    'danceability', 'energy', 'speechiness', 'acousticness',
    'instrumentalness', 'liveness', 'valence', 'tempo', 'loudness'
]

# 백그라운드 작업 상태 추적
_enrichment_status = {
    "running": False,
    "total": 0,
    "processed": 0,
    "failed": 0,
    "last_run": None
}


def _get_audio_service():
    """M2의 실제 ML 기반 AudioPredictionService 로드"""
    from M2.service import AudioPredictionService
    model_path = Path(__file__).parent / "M2" / "tfidf_gbr_models.pkl"
    service = AudioPredictionService(model_path=str(model_path))
    service.load_model()
    return service


def save_audio_features_to_db(db, track_id: int, features: Dict[str, float], existing_metadata=None):
    """
    오디오 특성을 DB에 저장
    - 개별 컬럼 (danceability, energy 등) 업데이트
    - external_metadata JSON에 audio_features 키로 병합
    """
    metadata = {}
    if existing_metadata:
        try:
            metadata = json.loads(existing_metadata) if isinstance(existing_metadata, str) else existing_metadata
            if not isinstance(metadata, dict):
                metadata = {}
        except (json.JSONDecodeError, TypeError):
            metadata = {}

    metadata['audio_features'] = features
    metadata['audio_features_source'] = 'tfidf_gbr_predicted'
    metadata['audio_features_updated_at'] = datetime.now().isoformat()

    update_query = text("""
        UPDATE tracks SET
            danceability = :danceability,
            energy = :energy,
            speechiness = :speechiness,
            acousticness = :acousticness,
            instrumentalness = :instrumentalness,
            liveness = :liveness,
            valence = :valence,
            tempo = :tempo,
            loudness = :loudness,
            external_metadata = :metadata
        WHERE track_id = :track_id
    """)

    db.execute(update_query, {
        "track_id": track_id,
        "danceability": round(features.get('danceability', 0), 4),
        "energy": round(features.get('energy', 0), 4),
        "speechiness": round(features.get('speechiness', 0), 4),
        "acousticness": round(features.get('acousticness', 0), 4),
        "instrumentalness": round(features.get('instrumentalness', 0), 4),
        "liveness": round(features.get('liveness', 0), 4),
        "valence": round(features.get('valence', 0), 4),
        "tempo": round(min(features.get('tempo', 120.0), 999.999), 3),
        "loudness": round(max(features.get('loudness', -6.0), -99.99), 2),
        "metadata": json.dumps(metadata, ensure_ascii=False)
    })


def enrich_tracks_batch(track_rows: list, db, batch_size: int = 500) -> dict:
    """
    트랙 리스트에 대해 배치 오디오 특성 예측 후 DB 저장

    Args:
        track_rows: [{'track_id', 'title', 'artist', 'album', 'duration', 'external_metadata'}, ...]
        db: SQLAlchemy session
        batch_size: ML 배치 크기
    """
    audio_service = _get_audio_service()
    success = 0
    failed = 0

    for i in range(0, len(track_rows), batch_size):
        batch = track_rows[i:i + batch_size]

        predictions = []
        batch_failed = False
        for t in batch:
            try:
                pred = audio_service.predict_single(
                    artist=t.get('artist') or '',
                    track_name=t.get('title') or '',
                    duration_ms=(t.get('duration') or 200) * 1000
                )
                predictions.append(pred)
            except Exception as e:
                logger.error(f"Batch prediction failed: {e}")
                failed += len(batch)
                batch_failed = True
                break

        if batch_failed:
            continue

        for track, features in zip(batch, predictions):
            try:
                save_audio_features_to_db(db, track['track_id'], features, track.get('external_metadata'))
                success += 1
            except Exception as e:
                logger.error(f"Failed to save track {track['track_id']}: {e}")
                failed += 1

        db.commit()
        _enrichment_status["processed"] = success
        logger.info(f"[Enrich] Batch {i // batch_size + 1}: {len(batch)} tracks processed")

    return {"success": success, "failed": failed}


def enrich_user_tracks(user_id: int, db) -> dict:
    """
    특정 유저의 PMS 트랙 중 오디오 특성이 없는 것들을 채워넣음
    회원가입(init-models) 시 M1 학습 전에 호출
    """
    query = text("""
        SELECT t.track_id, t.title, t.artist, t.album, t.duration, t.external_metadata
        FROM tracks t
        JOIN playlist_tracks pt ON t.track_id = pt.track_id
        JOIN playlists p ON pt.playlist_id = p.playlist_id
        WHERE p.user_id = :user_id
        AND p.space_type = 'PMS'
        AND (t.danceability IS NULL OR t.energy IS NULL)
    """)
    rows = db.execute(query, {"user_id": user_id}).fetchall()

    if not rows:
        logger.info(f"[Enrich] User {user_id}: PMS 트랙 모두 오디오 특성 보유")
        return {"success": True, "enriched": 0, "failed": 0}

    track_rows = [
        {
            'track_id': r[0], 'title': r[1], 'artist': r[2],
            'album': r[3], 'duration': r[4], 'external_metadata': r[5]
        }
        for r in rows
    ]

    result = enrich_tracks_batch(track_rows, db)
    logger.info(f"[Enrich] User {user_id}: {result['success']}/{len(track_rows)} PMS 트랙 enriched")
    return {"success": True, "enriched": result["success"], "failed": result["failed"]}


# ==================== API Endpoints ====================

class EnrichAllResponse(BaseModel):
    success: bool
    message: str
    total_missing: int
    job_started: bool


class EnrichTracksRequest(BaseModel):
    track_ids: List[int]


class EnrichTracksResponse(BaseModel):
    success: bool
    enriched_count: int
    failed_count: int


@router.post("/enrich-all", response_model=EnrichAllResponse)
async def enrich_all_tracks():
    """
    오디오 특성이 없는 모든 트랙을 일괄 처리 (백그라운드)
    GET /api/enrich-status 로 진행 상태 확인
    """
    global _enrichment_status

    if _enrichment_status["running"]:
        return EnrichAllResponse(
            success=False,
            message="이미 실행 중",
            total_missing=_enrichment_status["total"],
            job_started=False
        )

    db = SessionLocal()
    try:
        total_missing = db.execute(
            text("SELECT COUNT(*) FROM tracks WHERE danceability IS NULL OR energy IS NULL")
        ).scalar()
    finally:
        db.close()

    if total_missing == 0:
        return EnrichAllResponse(
            success=True,
            message="모든 트랙에 오디오 특성이 이미 있음",
            total_missing=0,
            job_started=False
        )

    _enrichment_status = {
        "running": True,
        "total": total_missing,
        "processed": 0,
        "failed": 0,
        "last_run": datetime.now().isoformat()
    }

    thread = threading.Thread(target=_run_batch_enrichment, daemon=True)
    thread.start()

    return EnrichAllResponse(
        success=True,
        message=f"{total_missing}개 트랙 백그라운드 처리 시작",
        total_missing=total_missing,
        job_started=True
    )


def _run_batch_enrichment():
    """백그라운드 일괄 처리"""
    global _enrichment_status
    db = SessionLocal()
    try:
        query = text("""
            SELECT track_id, title, artist, album, duration, external_metadata
            FROM tracks
            WHERE danceability IS NULL OR energy IS NULL
        """)
        rows = db.execute(query).fetchall()

        track_rows = [
            {
                'track_id': r[0], 'title': r[1], 'artist': r[2],
                'album': r[3], 'duration': r[4], 'external_metadata': r[5]
            }
            for r in rows
        ]

        result = enrich_tracks_batch(track_rows, db)
        _enrichment_status["processed"] = result["success"]
        _enrichment_status["failed"] = result["failed"]
        logger.info(f"[Enrich] 일괄 처리 완료: {result}")
    except Exception as e:
        logger.error(f"[Enrich] 일괄 처리 실패: {e}")
        _enrichment_status["failed"] = _enrichment_status["total"]
    finally:
        _enrichment_status["running"] = False
        db.close()


@router.post("/enrich-tracks", response_model=EnrichTracksResponse)
async def enrich_specific_tracks(request: EnrichTracksRequest):
    """특정 track_id 리스트의 오디오 특성을 채워넣음 (동기 처리)"""
    db = SessionLocal()
    try:
        placeholders = ",".join([f":id_{i}" for i in range(len(request.track_ids))])
        params = {f"id_{i}": tid for i, tid in enumerate(request.track_ids)}

        query = text(f"""
            SELECT track_id, title, artist, album, duration, external_metadata
            FROM tracks
            WHERE track_id IN ({placeholders})
            AND (danceability IS NULL OR energy IS NULL)
        """)
        rows = db.execute(query, params).fetchall()

        if not rows:
            return EnrichTracksResponse(success=True, enriched_count=0, failed_count=0)

        track_rows = [
            {
                'track_id': r[0], 'title': r[1], 'artist': r[2],
                'album': r[3], 'duration': r[4], 'external_metadata': r[5]
            }
            for r in rows
        ]

        result = enrich_tracks_batch(track_rows, db)
        return EnrichTracksResponse(
            success=True,
            enriched_count=result["success"],
            failed_count=result["failed"]
        )
    except Exception as e:
        logger.error(f"[Enrich] 트랙 처리 실패: {e}")
        return EnrichTracksResponse(success=False, enriched_count=0, failed_count=len(request.track_ids))
    finally:
        db.close()


@router.get("/enrich-status")
async def get_enrichment_status():
    """오디오 특성 커버리지 및 백그라운드 작업 상태 확인"""
    db = SessionLocal()
    try:
        total = db.execute(text("SELECT COUNT(*) FROM tracks")).scalar()
        missing = db.execute(
            text("SELECT COUNT(*) FROM tracks WHERE danceability IS NULL OR energy IS NULL")
        ).scalar()

        return {
            "total_tracks": total,
            "tracks_with_features": total - missing,
            "tracks_missing_features": missing,
            "coverage_percent": round((total - missing) / total * 100, 1) if total > 0 else 0,
            "background_job": _enrichment_status
        }
    finally:
        db.close()
