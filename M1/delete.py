from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db
from app.services.정혜원.spotify_service import SpotifyRecommendationService
import os

router = APIRouter(tags=["정혜원 - Delete"])

MODEL_PATH = os.path.join("app", "models", "정혜원", "audio_predictor.pkl")
spotify_service = SpotifyRecommendationService(model_path=MODEL_PATH)

@router.delete("/api/ai/playlist/{playlist_id}/tracks")
def delete_tracks(playlist_id: int, track_ids: list[int], db: Session = Depends(get_db)):
    """
    Delete tracks from GMS playlist (Step 4)
    """
    try:
        result = spotify_service.delete_tracks_from_playlist(db, playlist_id, track_ids)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/deleted-track")
def deleted_track(request: dict, db: Session = Depends(get_db)):
    """
    Spring Boot TestServiceImpl에서 호출하는 삭제 + 재학습 엔드포인트
    POST /deleted-track
    요청: {"users_id": 1, "playlists_id": 10, "tracks_id": 5}
    - 트랙 삭제 후 해당 트랙 정보로 모델 재학습
    """
    try:
        user_id = request.get("users_id")
        playlist_id = request.get("playlists_id")
        track_id = request.get("tracks_id")

        if not all([user_id, playlist_id, track_id]):
            return {"message": "users_id, playlists_id, tracks_id가 모두 필요합니다."}

        user_id = int(user_id)
        playlist_id = int(playlist_id)
        track_id = int(track_id)

        # 1. 트랙 삭제
        delete_result = spotify_service.delete_tracks_from_playlist(db, playlist_id, [track_id])

        # 2. 삭제된 트랙으로 모델 재학습
        retrain_result = spotify_service.retrain_with_feedback(db, user_id, [track_id])

        return {
            "message": f"트랙 {track_id} 삭제 및 모델 재학습 완료.",
            "retrain_metrics": retrain_result.get("metrics", {})
        }
    except Exception as e:
        return {"message": f"오류 발생: {str(e)}"}

@router.post("/api/ai/retrain/{user_id}")
def retrain_model(user_id: int, deleted_track_ids: list[int], db: Session = Depends(get_db)):
    """
    Retrain model based on user feedback (Step 5)
    """
    try:
        result = spotify_service.retrain_with_feedback(db, user_id, deleted_track_ids)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
