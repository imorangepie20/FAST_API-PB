from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db
from app.services.정혜원.spotify_service import SpotifyRecommendationService
import os

router = APIRouter(tags=["정혜원 - Analysis"])

MODEL_PATH = os.path.join("app", "models", "정혜원", "audio_predictor.pkl")
spotify_service = SpotifyRecommendationService(model_path=MODEL_PATH)

@router.post("/analyze")
def analyze_user(request: dict, db: Session = Depends(get_db)):
    """
    Spring Boot TestController에서 호출하는 분석 엔드포인트
    요청: {"userid": "1"}
    응답: {"message": "..."}

    userid를 받아서 추천 생성 + GMS 플레이리스트 저장까지 실행
    """
    try:
        user_id = request.get("userid")
        if not user_id:
            return {"message": "userid가 필요합니다"}

        user_id = int(user_id)

        # 추천 로직 실행
        results = spotify_service.get_recommendations(db, user_id)
        if results.empty:
            return {"message": f"사용자 {user_id}의 PMS 데이터가 없습니다."}

        # GMS 플레이리스트 저장
        playlist_id = spotify_service.save_gms_playlist(db, user_id, results)

        return {
            "message": f"사용자 {user_id}의 추천이 완료되었습니다. GMS 플레이리스트(ID: {playlist_id})에 {len(results)}곡 저장."
        }
    except Exception as e:
        return {"message": f"오류 발생: {str(e)}"}

@router.post("/api/ai/recommend/{user_id}")
def get_recommendations(user_id: int, db: Session = Depends(get_db)):
    """
    Generate recommendations for a user and save as GMS playlist (Step 3)
    """
    try:
        results = spotify_service.get_recommendations(db, user_id)
        if results.empty:
            return {"message": "No preferences found for this user", "recommendations": []}

        # Save as GMS playlist in DB
        playlist_id = spotify_service.save_gms_playlist(db, user_id, results)

        # Convert top 10 to list of dicts
        top_recommendations = results.head(10).copy()

        # 컬럼명 정리: final_score 또는 recommendation_score를 score로 통일
        if 'final_score' in top_recommendations.columns:
            top_recommendations['score'] = top_recommendations['final_score']
        elif 'recommendation_score' in top_recommendations.columns:
            top_recommendations['score'] = top_recommendations['recommendation_score']

        recommendations = top_recommendations.to_dict(orient='records')

        return {
            "user_id": user_id,
            "playlist_id": playlist_id,
            "message": "GMS playlist created successfully",
            "recommendations": recommendations
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
