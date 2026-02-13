"""
QLTY Router — FastAPI 엔드포인트

외부(Spring Boot, FE)에서 호출할 수 있는 API.

엔드포인트:
- POST /api/qlty/enrich        : 단일/배치 트랙 오디오 피처 보강
- POST /api/qlty/batch-update  : DB tracks 테이블 일괄 업데이트
- POST /api/qlty/train         : DL 모델 학습 (spotify_reference 기반)
- GET  /api/qlty/health        : 모듈 상태 확인
"""
import json
import logging
import asyncio
from typing import List, Optional, AsyncGenerator
from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import text

from .pipeline import TrackInput, enrich_track, enrich_batch, reload_dl_model
from .train import train_models

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/qlty", tags=["QLTY"])


# ==================== Request / Response Models ====================


class TrackRequest(BaseModel):
    """단일 트랙 입력"""
    title: str = Field(..., description="트랙 제목")
    artist: str = Field(..., description="아티스트명")
    album: str = Field("", description="앨범명")
    genre: str = Field("", description="장르")
    duration_ms: float = Field(0, description="트랙 길이(ms)")
    popularity: float = Field(0, description="인기도(0-100)")
    isrc: str = Field("", description="ISRC 코드")


class EnrichRequest(BaseModel):
    """피처 보강 요청"""
    tracks: List[TrackRequest] = Field(..., description="보강할 트랙 목록")
    skip_api: bool = Field(False, description="ReccoBeats API 스킵")
    skip_llm: bool = Field(False, description="LLM 추정 스킵")


class TrackResult(BaseModel):
    """단일 트랙 결과"""
    title: str
    artist: str
    features: dict = Field(default_factory=dict)
    sources: dict = Field(default_factory=dict)
    attempted: List[str] = Field(default_factory=list)
    feature_count: int = 0
    coverage: float = 0.0


class EnrichResponse(BaseModel):
    """피처 보강 응답"""
    success: bool
    total: int
    complete: int
    results: List[TrackResult]


# ==================== DB Dependency ====================

def get_db():
    """DB 세션 (database.py에서 가져옴)"""
    try:
        from database import SessionLocal
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()
    except ImportError:
        logger.warning("[QLTY Router] database 모듈 없음 — DB 없이 진행")
        yield None


# ==================== Endpoints ====================


@router.post("/enrich", response_model=EnrichResponse)
async def enrich_tracks(request: EnrichRequest, db: Session = Depends(get_db)):
    """
    트랙 오디오 피처 보강 API.

    4단계 우선순위로 시도:
    1. ReccoBeats API (ISRC 필요)
    2. spotify_reference DB 매칭
    3. DL 모델 예측
    4. LLM 추정

    요청 예시:
    ```json
    {
        "tracks": [
            {"title": "Blinding Lights", "artist": "The Weeknd", "isrc": "USUG11904154"},
            {"title": "Shape of You", "artist": "Ed Sheeran"}
        ]
    }
    ```
    """
    # TrackRequest → TrackInput 변환
    track_inputs = [
        TrackInput(
            title=t.title,
            artist=t.artist,
            album=t.album,
            genre=t.genre,
            duration_ms=t.duration_ms,
            popularity=t.popularity,
            isrc=t.isrc,
        )
        for t in request.tracks
    ]

    # 파이프라인 실행
    results = await enrich_batch(
        track_inputs,
        db=db,
        skip_api=request.skip_api,
        skip_llm=request.skip_llm,
    )

    # 응답 조립
    track_results = []
    for track, result in zip(request.tracks, results):
        track_results.append(
            TrackResult(
                title=track.title,
                artist=track.artist,
                features=result.features,
                sources=result.sources,
                attempted=result.attempted,
                feature_count=len(result.features),
                coverage=round(result.coverage, 3),
            )
        )

    complete_count = sum(1 for r in results if r.is_complete)

    return EnrichResponse(
        success=True,
        total=len(results),
        complete=complete_count,
        results=track_results,
    )


@router.post("/batch-update")
async def batch_update_tracks(
    limit: int = Query(default=0, description="최대 처리 곡 수 (0=전체)"),
    db: Session = Depends(get_db),
):
    """
    DB tracks 테이블의 피처 누락 트랙을 일괄 보강 + UPDATE.
    SSE 스트리밍으로 실시간 진행상황을 출력한다.
    """
    if db is None:
        return {"success": False, "error": "DB 연결 없음"}

    # 누락 트랙 조회
    query = "SELECT track_id, title, artist, album, genre, duration, popularity, isrc FROM tracks WHERE danceability IS NULL ORDER BY track_id"
    if limit > 0:
        query += f" LIMIT {limit}"

    rows = db.execute(text(query)).fetchall()
    total = len(rows)

    if total == 0:
        return {"success": True, "message": "업데이트 대상 없음", "total": 0}

    async def stream_progress() -> AsyncGenerator[str, None]:
        updated = 0
        failed = 0
        source_stats = {}

        yield f"data: {json.dumps({'event': 'start', 'total': total})}\n\n"

        for i, row in enumerate(rows):
            track_id, title, artist = row[0], row[1], row[2]
            track_input = TrackInput(
                title=title,
                artist=artist,
                album=row[3] or "",
                genre=row[4] or "",
                duration_ms=(row[5] or 0) * 1000,
                popularity=row[6] or 0,
                isrc=row[7] or "",
            )

            try:
                result = await enrich_track(track_input, db=db)
                features = result.features

                if features:
                    set_clauses = []
                    params = {"tid": track_id}
                    for col in ["danceability", "energy", "valence", "tempo", "acousticness", "instrumentalness", "liveness", "speechiness", "loudness", "music_key", "mode", "time_signature"]:
                        if col in features:
                            set_clauses.append(f"{col} = :{col}")
                            params[col] = features[col]

                    if set_clauses:
                        db.execute(text(f"UPDATE tracks SET {', '.join(set_clauses)} WHERE track_id = :tid"), params)
                        updated += 1

                    for src in result.sources.values():
                        source_stats[src] = source_stats.get(src, 0) + 1

                    yield f"data: {json.dumps({'event': 'progress', 'current': i+1, 'total': total, 'updated': updated, 'track': f'{artist} - {title}', 'sources': list(set(result.sources.values()))})}\n\n"
                else:
                    failed += 1
                    yield f"data: {json.dumps({'event': 'skip', 'current': i+1, 'total': total, 'track': f'{artist} - {title}', 'reason': 'no features'})}\n\n"

            except Exception as e:
                failed += 1
                yield f"data: {json.dumps({'event': 'error', 'current': i+1, 'total': total, 'track': f'{artist} - {title}', 'error': str(e)})}\n\n"

            # 20곡마다 커밋
            if (i + 1) % 20 == 0:
                db.commit()

            await asyncio.sleep(0)  # 이벤트 루프 양보

        db.commit()
        yield f"data: {json.dumps({'event': 'done', 'total': total, 'updated': updated, 'failed': failed, 'source_stats': source_stats})}\n\n"

    return StreamingResponse(
        stream_progress(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/train")
async def train_qlty_model(db: Session = Depends(get_db)):
    """
    DL 모델 학습 API.

    spotify_reference 테이블(86K+)로 LightGBM 모델을 학습한다.
    학습 완료 후 자동으로 모델을 리로드한다.

    주의: 학습에 수 분이 걸릴 수 있음 (임베딩 생성 포함).
    """
    if db is None:
        return {"success": False, "error": "DB 연결 없음"}

    try:
        result = train_models(db)

        # 새 모델 리로드
        reload_dl_model()

        return {
            "success": True,
            "message": "QLTY 모델 학습 완료",
            **result,
        }
    except Exception as e:
        logger.error(f"[QLTY Train] 학습 실패: {e}")
        return {"success": False, "error": str(e)}


@router.get("/health")
async def qlty_health():
    """
    QLTY 모듈 상태 확인.

    각 컴포넌트의 사용 가능 여부를 반환:
    - dl_model: 학습된 모델 파일 존재 여부
    - gemini: API 키 설정 여부
    - db: 데이터베이스 연결 가능 여부
    - reccobeats: API 접근 가능 (항상 True, 외부 서비스)
    """
    import os
    from pathlib import Path

    # DL 모델 파일 확인
    model_path = Path(__file__).resolve().parent / "qlty_models.pkl"
    dl_ready = model_path.exists()

    # Gemini API 키 확인
    gemini_ready = bool(os.getenv("GOOGLE_API_KEY", ""))

    # DB 연결 확인
    db_ready = False
    try:
        from database import test_connection
        db_ready = test_connection()
    except Exception:
        pass

    return {
        "status": "ready" if (dl_ready and db_ready) else "partial",
        "components": {
            "dl_model": dl_ready,
            "gemini": gemini_ready,
            "db": db_ready,
            "reccobeats": True,  # 외부 API, 항상 시도 가능
        },
        "model_path": str(model_path) if dl_ready else None,
    }
