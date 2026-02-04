"""
M1 API Router
트랙 분석, 추천, 피드백 재학습 API
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import sys

# 현재 디렉토리를 path에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import get_db
from M1.service import M1RecommendationService

router = APIRouter(prefix="/api/m1", tags=["M1 - Audio Feature Prediction"])

# 서비스 초기화
MODEL_PATH = os.path.join(os.path.dirname(__file__), "audio_predictor.pkl")
m1_service = M1RecommendationService(model_path=MODEL_PATH)


# ==================== Request/Response Models ====================

class AnalyzeRequest(BaseModel):
    userid: int

class RecommendResponse(BaseModel):
    user_id: int
    playlist_id: int
    message: str
    recommendations: List[Dict[str, Any]]

class DeleteTrackRequest(BaseModel):
    users_id: int
    playlists_id: int
    tracks_id: int

class RetrainRequest(BaseModel):
    user_id: int
    deleted_track_ids: List[int]


# ==================== API Endpoints ====================

@router.get("/health")
async def health_check():
    """M1 모듈 상태 확인"""
    return {
        "status": "healthy",
        "module": "M1 - Audio Feature Prediction",
        "model_loaded": m1_service.is_model_loaded(),
        "model_path": MODEL_PATH
    }


@router.post("/analyze")
async def analyze_user(request: AnalyzeRequest, db: Session = Depends(get_db)):
    """
    사용자 분석 및 추천 생성 (Spring Boot TestController 연동)
    
    - PMS 데이터에서 사용자 프로필 생성
    - EMS 데이터에서 후보 트랙 추출
    - AI 점수 계산 후 GMS 플레이리스트 저장
    """
    try:
        user_id = request.userid
        
        # 추천 생성
        results = m1_service.get_recommendations(db, user_id)
        
        if results.empty:
            return {"message": f"사용자 {user_id}의 PMS 데이터가 없습니다."}
        
        # GMS 플레이리스트 저장
        playlist_id = m1_service.save_gms_playlist(db, user_id, results)
        
        return {
            "message": f"사용자 {user_id}의 추천이 완료되었습니다. GMS 플레이리스트(ID: {playlist_id})에 {len(results)}곡 저장.",
            "playlist_id": playlist_id,
            "track_count": len(results)
        }
    except Exception as e:
        return {"message": f"오류 발생: {str(e)}"}


@router.post("/recommend/{user_id}", response_model=RecommendResponse)
async def get_recommendations(user_id: int, db: Session = Depends(get_db)):
    """
    사용자 추천 생성 및 GMS 플레이리스트 저장
    
    1. PMS에서 사용자 선호 트랙 분석
    2. EMS에서 후보 트랙 평가
    3. final_score >= 0.7 트랙을 GMS에 저장
    4. Top 10 추천 반환
    """
    try:
        results = m1_service.get_recommendations(db, user_id)
        
        if results.empty:
            return RecommendResponse(
                user_id=user_id,
                playlist_id=0,
                message="No preferences found for this user",
                recommendations=[]
            )
        
        # GMS 플레이리스트 저장
        playlist_id = m1_service.save_gms_playlist(db, user_id, results)
        
        # Top 10 반환
        top_recommendations = results.head(10).copy()
        
        if 'final_score' in top_recommendations.columns:
            top_recommendations['score'] = top_recommendations['final_score']
        elif 'recommendation_score' in top_recommendations.columns:
            top_recommendations['score'] = top_recommendations['recommendation_score']
        
        recommendations = top_recommendations.to_dict(orient='records')
        
        return RecommendResponse(
            user_id=user_id,
            playlist_id=playlist_id,
            message="GMS playlist created successfully",
            recommendations=recommendations
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/deleted-track")
async def deleted_track(request: DeleteTrackRequest, db: Session = Depends(get_db)):
    """
    트랙 삭제 + 모델 재학습 (Spring Boot TestServiceImpl 연동)
    
    - 트랙을 플레이리스트에서 삭제
    - 삭제된 트랙을 '싫어요'로 학습
    """
    try:
        user_id = request.users_id
        playlist_id = request.playlists_id
        track_id = request.tracks_id
        
        # 1. 트랙 삭제
        delete_result = m1_service.delete_tracks_from_playlist(db, playlist_id, [track_id])
        
        # 2. 모델 재학습
        retrain_result = m1_service.retrain_with_feedback(db, user_id, [track_id])
        
        return {
            "message": f"트랙 {track_id} 삭제 및 모델 재학습 완료.",
            "retrain_metrics": retrain_result.get("metrics", {})
        }
    except Exception as e:
        return {"message": f"오류 발생: {str(e)}"}


@router.delete("/playlist/{playlist_id}/tracks")
async def delete_tracks(
    playlist_id: int, 
    track_ids: List[int] = Query(..., description="삭제할 트랙 ID 목록"),
    db: Session = Depends(get_db)
):
    """GMS 플레이리스트에서 트랙 삭제"""
    try:
        result = m1_service.delete_tracks_from_playlist(db, playlist_id, track_ids)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/retrain/{user_id}")
async def retrain_model(
    user_id: int,
    request: RetrainRequest,
    db: Session = Depends(get_db)
):
    """
    피드백 기반 모델 재학습
    
    - 삭제된 트랙들을 '싫어요' 데이터로 활용
    - PreferenceClassifier 재학습
    """
    try:
        result = m1_service.retrain_with_feedback(db, user_id, request.deleted_track_ids)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/user/{user_id}/profile")
async def get_user_profile(user_id: int, db: Session = Depends(get_db)):
    """
    사용자 음악 취향 프로필 조회
    
    - PMS 데이터 기반 분석 결과 반환
    """
    try:
        profile = m1_service.get_user_profile(db, user_id)
        return {
            "user_id": user_id,
            "profile": profile
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
