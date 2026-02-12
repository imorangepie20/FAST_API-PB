"""
사용자 모델 초기화 API
회원가입 시 M1, M2, M3 모델을 사용자 플레이리스트 기반으로 학습
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text
from database import get_db, SessionLocal
import os
import shutil
from pathlib import Path
import logging
import pickle
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)
router = APIRouter(tags=["User Model Initialization"])

# 기본 경로 설정
BASE_DIR = Path(__file__).parent

# M1 기본 모델 경로
M1_BASE_MODEL = BASE_DIR / "M1" / "audio_predictor.pkl"
M1_USER_MODELS_DIR = BASE_DIR / "M1" / "user_models"

# M2 기본 모델 경로
M2_BASE_MODEL = BASE_DIR / "M2" / "tfidf_gbr_models.pkl"
M2_USER_MODELS_DIR = BASE_DIR / "M2" / "user_svm_models"

# M3 기본 모델 경로
M3_MODEL_DIR = BASE_DIR / "M3"
M3_BASE_MODEL = BASE_DIR / "M3" / "recommender_U2_MyInitialT_20260201.cbm"
M3_USER_MODELS_DIR = BASE_DIR / "M3" / "user_models"


def get_email_prefix(email: str) -> str:
    """이메일에서 @ 앞자리 추출"""
    if not email or '@' not in email:
        return "unknown"
    return email.split('@')[0]


def get_user_playlists(db: Session, user_id: int) -> list:
    """사용자의 PMS(개인 플레이리스트)에서 트랙 정보 조회 - 학습 데이터"""
    try:
        print(f"[GetPlaylists] DB 조회 시작: userId={user_id}")
        
        # PMS만 조회 (사용자 개인 취향 학습용)
        query = text("""
            SELECT t.track_id, t.title, t.artist, t.album, t.duration, t.external_metadata
            FROM tracks t
            JOIN playlist_tracks pt ON t.track_id = pt.track_id
            JOIN playlists p ON pt.playlist_id = p.playlist_id
            WHERE p.user_id = :user_id AND p.space_type = 'PMS'
        """)
        result = db.execute(query, {"user_id": user_id}).fetchall()

        tracks = []
        for r in result:
            tracks.append({
                'track_id': r[0],
                'title': r[1],
                'artist': r[2],
                'album': r[3] or '',
                'duration': r[4] or 200,
                'external_metadata': r[5]
            })

        print(f"[GetPlaylists] 조회 완료: userId={user_id}, track_count={len(tracks)}")
        if len(tracks) > 0:
            print(f"[GetPlaylists] 첫 번째 트랙: {tracks[0].get('title')} - {tracks[0].get('artist')}")
        
        logger.info(f"[GetPlaylists] userId={user_id}, track_count={len(tracks)}")
        return tracks
    except Exception as e:
        print(f"[GetPlaylists] ERROR: {e}")
        logger.error(f"[GetPlaylists] Failed: {e}")
        import traceback
        traceback.print_exc()
        return []


def train_m1_model(email: str, user_id: int, tracks: list, db: Session) -> dict:
    """M1 모델 학습: 사용자 플레이리스트 기반 Ridge 모델"""
    try:
        email_prefix = get_email_prefix(email)
        user_folder = M1_USER_MODELS_DIR / email_prefix
        user_folder.mkdir(parents=True, exist_ok=True)
        user_model_path = user_folder / f"{email_prefix}_.pkl"

        print(f"[M1] train_m1_model 시작: email={email}, tracks={len(tracks)}")
        print(f"[M1] 사용자 폴더: {user_folder}")
        print(f"[M1] 모델 파일 경로: {user_model_path}")
        print(f"[M1] 기본 모델 존재: {M1_BASE_MODEL.exists()} ({M1_BASE_MODEL})")

        # 트랙이 없으면 기본 모델 복사
        if not tracks:
            print(f"[M1] 트랙 없음 - 기본 모델 복사")
            if M1_BASE_MODEL.exists():
                shutil.copy2(M1_BASE_MODEL, user_model_path)
                print(f"[M1] 기본 모델 복사 완료: {user_model_path}")
                logger.info(f"[M1] 기본 모델 복사 (트랙 없음): {user_model_path}")
                return {"success": True, "path": str(user_model_path), "status": "base_model_copied"}
            else:
                print(f"[M1] ERROR: 기본 모델도 없음!")
                return {"success": False, "error": "No tracks and no base model"}

        # M1 서비스를 사용하여 실제 학습
        try:
            print(f"[M1] M1RecommendationService 로드 중...")
            from M1.service import M1RecommendationService
            model_path = str(M1_BASE_MODEL) if M1_BASE_MODEL.exists() else ""
            service = M1RecommendationService(model_path=model_path)

            # 1. 기본 모델을 사용자 폴더로 복사
            print(f"[M1] 기본 모델 → 사용자 폴더 복사...")
            service.copy_base_model_to_user(email)

            # 2. PMS 트랙으로 모델 추가학습 및 저장
            print(f"[M1] PMS 트랙으로 추가학습 시작...")
            train_result = service.train_user_model(db, user_id, email)
            print(f"[M1] 학습 결과: {train_result}")

            if train_result.get("success"):
                print(f"[M1] 학습 성공! model_path={train_result.get('model_path')}")
                logger.info(f"[M1] 모델 학습 완료: {train_result.get('model_path')}, tracks={len(tracks)}")
                return {"success": True, "path": train_result.get("model_path"), "status": "trained", "track_count": len(tracks)}
            else:
                print(f"[M1] 학습 실패: {train_result.get('message')}")
                logger.warning(f"[M1] 학습 실패: {train_result.get('message')}")
                # 학습 실패시 기본 모델 복사
                shutil.copy2(M1_BASE_MODEL, user_model_path)
                return {"success": True, "path": str(user_model_path), "status": "base_model_copied"}

        except Exception as e:
            print(f"[M1] 학습 중 예외 발생: {e}")
            import traceback
            traceback.print_exc()
            logger.warning(f"[M1] 학습 실패, 기본 모델 복사: {e}")
            if M1_BASE_MODEL.exists():
                shutil.copy2(M1_BASE_MODEL, user_model_path)
                return {"success": True, "path": str(user_model_path), "status": "base_model_copied"}
            return {"success": False, "error": str(e)}

    except Exception as e:
        print(f"[M1] 모델 생성 실패: {e}")
        import traceback
        traceback.print_exc()
        logger.error(f"[M1] 모델 생성 실패: {e}")
        return {"success": False, "error": str(e)}


def train_m2_model(email: str, user_id: int, tracks: list, db: Session) -> dict:
    """M2 모델 학습: SVM + Text Embedding 기반"""
    try:
        email_prefix = get_email_prefix(email)
        M2_USER_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        user_model_path = M2_USER_MODELS_DIR / f"{email_prefix}_.pkl"

        # 트랙이 5개 미만이면 기본 모델 복사
        if len(tracks) < 5:
            if M2_BASE_MODEL.exists():
                shutil.copy2(M2_BASE_MODEL, user_model_path)
                logger.info(f"[M2] 기본 모델 복사 (트랙 부족): {user_model_path}")
                return {"success": True, "path": str(user_model_path), "status": "base_model_copied"}
            else:
                return {"success": False, "error": "No tracks and no base model"}

        try:
            from M2.service import get_m2_service
            m2_service = get_m2_service()

            # Positive 트랙 (사용자 플레이리스트)
            positive_tracks = [
                {
                    'track_name': t['title'],
                    'artist': t['artist'],
                    'album_name': t['album'],
                    'tags': '',
                    'duration_ms': t['duration'] * 1000
                }
                for t in tracks
            ]

            # Negative 트랙 (EMS 공용 풀에서 랜덤 샘플링)
            neg_query = text("""
                SELECT DISTINCT t.title, t.artist, t.album, t.duration
                FROM tracks t
                JOIN playlist_tracks pt ON t.track_id = pt.track_id
                JOIN playlists p ON pt.playlist_id = p.playlist_id
                WHERE p.space_type = 'EMS'
                ORDER BY RAND()
                LIMIT :limit
            """)
            neg_result = db.execute(neg_query, {"limit": len(tracks) * 2}).fetchall()

            negative_tracks = [
                {
                    'track_name': r[0],
                    'artist': r[1],
                    'album_name': r[2] or '',
                    'tags': '',
                    'duration_ms': (r[3] or 200) * 1000
                }
                for r in neg_result
            ]

            # M2 모델 학습
            result = m2_service.train_user_model(
                user_id=user_id,
                positive_tracks=positive_tracks,
                negative_tracks=negative_tracks
            )

            logger.info(f"[M2] 모델 학습 완료: {user_model_path}, positive={len(positive_tracks)}, negative={len(negative_tracks)}")
            return {"success": True, "path": str(user_model_path), "status": "trained", "track_count": len(tracks)}

        except Exception as e:
            logger.warning(f"[M2] 학습 실패, 기본 모델 복사: {e}")
            # 학습 실패시 기본 모델 복사
            if M2_BASE_MODEL.exists():
                shutil.copy2(M2_BASE_MODEL, user_model_path)
                return {"success": True, "path": str(user_model_path), "status": "base_model_copied", "error": str(e)}
            else:
                user_model_path.touch()
                return {"success": True, "path": str(user_model_path), "status": "pending", "error": str(e)}

    except Exception as e:
        logger.error(f"[M2] 모델 생성 실패: {e}")
        return {"success": False, "error": str(e)}


def generate_gms_recommendations(user_id: int, db: Session, model_name: str = "M1") -> dict:
    """모델 학습 후 GMS 추천 플레이리스트 생성 (지정된 모델 사용)"""
    try:
        from M1.service import M1RecommendationService
        from pathlib import Path

        # 지정된 모델 사용
        # 회원가입 때는 기본값 M1 사용, 추후 Settings에서 선택한 모델 사용 가능
        service = None
        recommendations = None

        if model_name == "M1":
            model_path = str(M1_BASE_MODEL) if M1_BASE_MODEL.exists() else ""
            service = M1RecommendationService(model_path=model_path)
            recommendations = service.get_recommendations(db, user_id)
        elif model_name == "M2":
            # M는 직접 구현 후 DB 저장 필요
            from M2.service import get_m2_service
            m2_service = get_m2_service()
            from sqlalchemy import text
            pms_query = text("""
                SELECT t.title, t.artist, t.album, t.duration
                FROM tracks t
                JOIN playlist_tracks pt ON t.track_id = pt.track_id
                JOIN playlists p ON pt.playlist_id = p.playlist_id
                WHERE p.user_id = :user_id AND p.space_type = 'PMS'
            """)
            pms_result = db.execute(pms_query, {"user_id": user_id}).fetchall()
            positive_tracks = [
                {'track_name': r[0], 'artist': r[1], 'album_name': r[2] or '', 'tags': '', 'duration_ms': (r[3] or 200) * 1000}
                for r in pms_result
            ]
            ems_query = text("""
                SELECT DISTINCT t.track_id, t.title, t.artist, t.album, t.duration, t.external_metadata
                FROM tracks t
                JOIN playlist_tracks pt ON t.track_id = pt.track_id
                JOIN playlists p ON pt.playlist_id = p.playlist_id
                WHERE p.space_type = 'EMS'
                ORDER BY RAND()
                LIMIT 200
            """)
            ems_result = db.execute(ems_query).fetchall()
            candidate_tracks = [
                {'track_id': r[0], 'track_name': r[1], 'artist': r[2], 'album_name': r[3] or '', 'tags': '', 'duration_ms': (r[4] or 200) * 1000}
                for r in ems_result
            ]
            recommendations_data = m2_service.get_recommendations(
                user_id=user_id,
                candidate_tracks=candidate_tracks,
                top_k=20,
                threshold=0.5
            )
            recommendations = pd.DataFrame(recommendations_data) if recommendations_data else pd.DataFrame()
        elif model_name == "M3":
            # M3도 직접 구현 후 DB 저장 필요
            from M3.service import get_m3_service
            m3_service = get_m3_service()
            result = m3_service.get_recommendations(db, user_id, top_k=20)
            recommendations = pd.DataFrame(result.get("recommendations", [])) if result.get("recommendations") else pd.DataFrame()
        else:
            return {"success": False, "error": f"Unknown model: {model_name}"}

        if recommendations is not None and not recommendations.empty:
            # NaN 처리
            recommendations = recommendations.fillna(0)
            recommendations = recommendations.replace([np.inf, -np.inf], 0)

            # M1 서비스로 GMS 저장
            if service:
                playlist_id = service.save_gms_playlist(db, user_id, recommendations)
            else:
                # M2/M3 또는 service가 None이면 M1RecommendationService 새로 생성
                m1_service = M1RecommendationService()
                playlist_id = m1_service.save_gms_playlist(db, user_id, recommendations)

            logger.info(f"[GMS] 추천 생성 완료: userId={user_id}, model={model_name}, tracks={len(recommendations)}, playlistId={playlist_id}")
            return {
                "success": True,
                "status": "created",
                "model": model_name,
                "track_count": len(recommendations),
                "playlist_id": playlist_id
            }
        else:
            logger.warning(f"[GMS] 추천 결과 없음: userId={user_id}, model={model_name}")
            return {"success": False, "status": "no_recommendations", "reason": "EMS 데이터 부족 또는 추천 조건 미충족"}

    except Exception as e:
        logger.error(f"[GMS] 추천 생성 실패 (model={model_name}): {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


def train_m3_model(email: str, user_id: int, tracks: list, db: Session) -> dict:
    """M3 모델 학습: CatBoost 기반 사용자 맞춤 모델"""
    try:
        email_prefix = get_email_prefix(email)
        M3_USER_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        user_model_path = M3_USER_MODELS_DIR / f"{email_prefix}_.cbm"

        # 트랙이 10개 미만이면 기본 모델 복사
        if len(tracks) < 10:
            if M3_BASE_MODEL.exists():
                shutil.copy2(M3_BASE_MODEL, user_model_path)
                logger.info(f"[M3] 기본 모델 복사 (트랙 부족): {user_model_path}")
                return {"success": True, "path": str(user_model_path), "status": "base_model_copied"}
            else:
                base_models = list(M3_MODEL_DIR.glob("*.cbm"))
                if base_models:
                    shutil.copy2(base_models[0], user_model_path)
                    return {"success": True, "path": str(user_model_path), "status": "base_model_copied"}
                return {"success": False, "error": "No base model found"}

        # M3 서비스를 사용하여 실제 학습
        try:
            from M3.service import get_m3_service
            m3_service = get_m3_service()

            # PMS 트랙으로 모델 학습
            train_result = m3_service.train_user_model(db, user_id)

            if train_result.get("success"):
                logger.info(f"[M3] 모델 학습 완료: userId={user_id}, tracks={len(tracks)}")
                return {"success": True, "path": str(user_model_path), "status": "trained", "track_count": len(tracks)}
            else:
                logger.warning(f"[M3] 학습 실패: {train_result.get('message')}")
                # 학습 실패시 기본 모델 복사
                if M3_BASE_MODEL.exists():
                    shutil.copy2(M3_BASE_MODEL, user_model_path)
                    return {"success": True, "path": str(user_model_path), "status": "base_model_copied"}
                return {"success": False, "error": train_result.get("message")}

        except Exception as e:
            logger.warning(f"[M3] 학습 실패, 기본 모델 복사: {e}")
            if M3_BASE_MODEL.exists():
                shutil.copy2(M3_BASE_MODEL, user_model_path)
                return {"success": True, "path": str(user_model_path), "status": "base_model_copied", "error": str(e)}
            return {"success": False, "error": str(e)}

    except Exception as e:
        logger.error(f"[M3] 모델 생성 실패: {e}")
        return {"success": False, "error": str(e)}


@router.post("/init-models")
async def init_user_models(request: dict):
    """
    사용자 모델 초기화 API - 플레이리스트 기반 학습 (비동기)

    Request: {"email": "apple100@gmail.com", "userId": 1}
    Response: {
        "success": true,
        "message": "모델 초기화 완료",
        "track_count": 50,
        "models": {
            "M1": {"success": true, "status": "trained", "track_count": 50},
            "M2": {"success": true, "status": "pending", "reason": "training_async"},
            "M3": {"success": true, "status": "pending", "reason": "training_async"}
        },
        "async_training": true
    }
    """
    print(f"[InitModels] ========== API 호출됨 ==========")
    print(f"[InitModels] Request: {request}")
    
    db = SessionLocal()
    try:
        email = request.get("email")
        user_id = request.get("userId")
        model = request.get("model", "M1")  # 기본값 M1

        print(f"[InitModels] Parsed: email={email}, userId={user_id}, model={model}")

        if not email:
            print(f"[InitModels] ERROR: email is required")
            return {"success": False, "error": "email is required"}
        if not user_id:
            print(f"[InitModels] ERROR: userId is required")
            return {"success": False, "error": "userId is required"}

        logger.info(f"[InitModels] 사용자 모델 초기화 시작: email={email}, userId={user_id}, model={model}")

        # 모델 설정 저장 (user_preferences 테이블)
        try:
            db.execute(text("""
                INSERT INTO user_preferences (user_id, ai_model)
                VALUES (:uid, :model)
                ON DUPLICATE KEY UPDATE ai_model = :model
            """), {"uid": user_id, "model": model})
            db.commit()
            logger.info(f"[InitModels] 모델 설정 저장: userId={user_id}, model={model}")
        except Exception as e:
            logger.warning(f"[InitModels] 모델 설정 저장 실패 (무시): {e}")

        # 사용자 플레이리스트 조회 (재시도 로직 포함)
        import time
        max_retries = 5
        retry_delay = 3  # 초
        tracks = []
        
        for attempt in range(max_retries):
            tracks = get_user_playlists(db, user_id)
            print(f"[InitModels] PMS 트랙 조회 시도 {attempt + 1}/{max_retries}: {len(tracks)}곡")
            
            if len(tracks) > 0:
                print(f"[InitModels] PMS 트랙 발견! {len(tracks)}곡")
                break
            
            if attempt < max_retries - 1:
                print(f"[InitModels] PMS 트랙 없음, {retry_delay}초 후 재시도...")
                time.sleep(retry_delay)
                # DB 세션 새로고침 (트랜잭션 커밋 반영)
                db.close()
                db = SessionLocal()
        
        logger.info(f"[InitModels] 플레이리스트 트랙 수: {len(tracks)}")
        
        if len(tracks) == 0:
            print(f"[InitModels] WARNING: {max_retries}회 시도 후에도 PMS에 트랙이 없습니다! userId={user_id}")
            print(f"[InitModels] 기본 모델만 복사됩니다.")

        # PMS 트랙 오디오 특성 채우기 (M1 학습 전 필수)
        print(f"[InitModels] PMS 트랙 오디오 특성 enrichment 시작...")
        try:
            from audio_enrichment import enrich_user_tracks
            enrich_result = enrich_user_tracks(user_id, db)
            print(f"[InitModels] Enrichment 결과: {enrich_result}")
        except Exception as e:
            print(f"[InitModels] Enrichment 실패 (계속 진행): {e}")

        # M1 모델 학습 (동기, 즉시 완료)
        print(f"[InitModels] M1 모델 학습 시작...")
        m1_result = train_m1_model(email, user_id, tracks, db)
        print(f"[InitModels] M1 모델 학습 결과: {m1_result}")

        results = {
            "M1": m1_result,
            # M2, M3는 비동기로 백그라운드 학습
            "M2": {"success": True, "status": "pending", "reason": "training_async"},
            "M3": {"success": True, "status": "pending", "reason": "training_async"}
        }

        # GMS 추천 생성 (모델 학습 후 자동 추천)
        print(f"[InitModels] GMS 추천 생성 시작...")
        gms_result = generate_gms_recommendations(user_id, db, model_name=model)
        results["GMS"] = gms_result
        print(f"[InitModels] GMS 추천 결과: {gms_result}")

        # M2, M3 비동기 학습 시작 (백그라운드)
        async_train_m2_m3(email, user_id, tracks)

        # 전체 성공 여부 확인 (partial success도 OK)
        any_success = any(r.get("success", False) for r in results.values())

        logger.info(f"[InitModels] M1 완료: email={email}, track_count={len(tracks)}, m1_result={m1_result}")
        logger.info(f"[InitModels] M2/M3 비동기 학습 시작: email={email}, userId={user_id}")

        print(f"[InitModels] ========== 완료 ==========")
        print(f"[InitModels] success={any_success}, track_count={len(tracks)}")
        print(f"[InitModels] models={results}")

        return {
            "success": any_success,
            "message": "모델 초기화 완료 (M2/M3 비동기 학습 중)",
            "track_count": len(tracks),
            "async_training": True,
            "models": results
        }

    except Exception as e:
        logger.error(f"[InitModels] 사용자 모델 초기화 실패: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}
    finally:
        db.close()


def async_train_m2_m3(email: str, user_id: int, tracks: list):
    """M2, M3 모델 비동기 학습 (백그라운드)"""
    import threading
    
    print(f"[AsyncTrain] FUNCTION CALLED: user_id={user_id}, tracks={len(tracks)}")
    
    def train_background():
        try:
            print(f"[AsyncTrain] 백그라운드 쓰레드 시작: user_id={user_id}")
            db = SessionLocal()
            logger.info(f"[AsyncTrain] M2/M3 백그라운드 학습 시작: user_id={user_id}")
            
            # M2 학습
            if len(tracks) >= 5:
                logger.info(f"[AsyncTrain] M2 학습 시작...")
                logger.info(f"[AsyncTrain] M2 학습 시작")
                m2_result = train_m2_model(email, user_id, tracks, db)
                logger.info(f"[AsyncTrain] M2 완료: {m2_result.get('status')}")
            
            # M3 학습
            if len(tracks) >= 5:
                logger.info(f"[AsyncTrain] M3 학습 시작...")
                logger.info(f"[AsyncTrain] M3 학습 시작")
                m3_result = train_m3_model(email, user_id, tracks, db)
                logger.info(f"[AsyncTrain] M3 완료: {m3_result.get('status')}")
            
            db.close()
            logger.info(f"[AsyncTrain] M2/M3 백그라운드 학습 완료")
            print(f"[AsyncTrain] 백그라운드 쓰레드 완료: user_id={user_id}")
        except Exception as e:
            logger.error(f"[AsyncTrain] M2/M3 학습 실패: {e}")
            print(f"[AsyncTrain] 백그라운드 쓰레드 실패: {e}")
    
    thread = threading.Thread(target=train_background)
    thread.daemon = True
    thread.start()
    print(f"[AsyncTrain] 쓰레드 완료")
