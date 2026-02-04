"""
AI Music Analysis API
EMS (Explore Music Space) 데이터 기반 AI 분석 서버

Models:
- M1: Audio Feature Prediction (오디오 특성 예측 + 하이브리드 추천)
- M2: Content-based Recommendation (TF-IDF + GBR)
- M3: Collaborative Filtering (CatBoost)
"""
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager
import uvicorn
import os
import sys

# 현재 디렉토리를 path에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ==================== Lifespan (시작/종료 이벤트) ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 시작/종료 시 실행"""
    # 시작 시
    print("=" * 60)
    print("🚀 AI Music Analysis API 시작")
    print("=" * 60)
    
    # DB 연결 테스트
    try:
        from database import test_connection
        if test_connection():
            print("✅ Database 연결 성공")
        else:
            print("⚠️ Database 연결 실패 - API는 계속 실행됨")
    except Exception as e:
        print(f"⚠️ Database 모듈 로드 실패: {e}")
    
    # M1 모델 상태
    try:
        model_path = os.path.join(os.path.dirname(__file__), "M1", "audio_predictor.pkl")
        if os.path.exists(model_path):
            print(f"✅ M1 모델 파일 존재: {model_path}")
        else:
            print(f"⚠️ M1 모델 파일 없음: {model_path}")
    except Exception as e:
        print(f"⚠️ M1 모델 확인 실패: {e}")
    
    print("=" * 60)
    
    yield  # 앱 실행
    
    # 종료 시
    print("🛑 AI Music Analysis API 종료")


# ==================== FastAPI 앱 초기화 ====================

app = FastAPI(
    title="AI Music Analysis API",
    description="""
## EMS 데이터 기반 음악 분석 및 추천 API

### 모델
- **M1**: 오디오 특성 예측 + 하이브리드 추천
- **M2**: TF-IDF 콘텐츠 기반 추천 (예정)
- **M3**: CatBoost 협업 필터링 (예정)

### 연동
- Node.js Backend EMS API와 통신
- Spring Boot Backend와 연동
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:5173",
        "http://localhost:8080",  # Spring Boot
        "https://homological-ashlyn-supercrowned.ngrok-free.dev",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== M1 Router 등록 ====================

try:
    from M1.router import router as m1_router
    app.include_router(m1_router)
    print("✅ M1 Router 등록 완료")
except Exception as e:
    print(f"⚠️ M1 Router 등록 실패: {e}")


# ==================== Pydantic Models (공통) ====================

class TrackFeatures(BaseModel):
    """트랙 오디오 특성"""
    tempo: Optional[float] = None
    energy: Optional[float] = None
    danceability: Optional[float] = None
    valence: Optional[float] = None
    acousticness: Optional[float] = None
    instrumentalness: Optional[float] = None

class TrackInput(BaseModel):
    """분석할 트랙 정보"""
    trackId: int
    title: str
    artist: str
    album: Optional[str] = None
    duration: Optional[int] = None
    genre: Optional[str] = None
    audioFeatures: Optional[TrackFeatures] = None


# ==================== Root Endpoints ====================

@app.get("/")
async def root():
    """API 상태 및 엔드포인트 목록"""
    return {
        "status": "running",
        "service": "AI Music Analysis API",
        "version": "1.0.0",
        "environment": os.getenv("ENVIRONMENT", "development"),
        "models": {
            "M1": "Audio Feature Prediction + Hybrid Recommendation",
            "M2": "TF-IDF Content-based (예정)",
            "M3": "CatBoost Collaborative Filtering (예정)"
        },
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "m1": {
                "health": "/api/m1/health",
                "analyze": "/api/m1/analyze",
                "recommend": "/api/m1/recommend/{user_id}",
                "profile": "/api/m1/user/{user_id}/profile",
                "deleted_track": "/api/m1/deleted-track",
                "retrain": "/api/m1/retrain/{user_id}"
            }
        }
    }


@app.get("/health")
async def health_check():
    """전체 시스템 헬스 체크"""
    health_status = {
        "status": "healthy",
        "api": True,
        "database": False,
        "models": {
            "M1": False,
            "M2": False,
            "M3": False
        }
    }
    
    # DB 연결 확인
    try:
        from database import test_connection
        health_status["database"] = test_connection()
    except:
        pass
    
    # M1 모델 확인
    try:
        model_path = os.path.join(os.path.dirname(__file__), "M1", "audio_predictor.pkl")
        health_status["models"]["M1"] = os.path.exists(model_path)
    except:
        pass
    
    # M2 모델 확인
    try:
        model_path = os.path.join(os.path.dirname(__file__), "M2", "tfidf_gbr_models.pkl")
        health_status["models"]["M2"] = os.path.exists(model_path)
    except:
        pass
    
    # M3 모델 확인
    try:
        m3_dir = os.path.join(os.path.dirname(__file__), "M3")
        cbm_files = [f for f in os.listdir(m3_dir) if f.endswith('.cbm')]
        health_status["models"]["M3"] = len(cbm_files) > 0
    except:
        pass
    
    return health_status


# ==================== Legacy Endpoints (하위 호환) ====================

@app.post("/analyze")
async def legacy_analyze(request: dict):
    """
    레거시 분석 엔드포인트 (Spring Boot 호환)
    → /api/m1/analyze로 리다이렉트
    """
    try:
        from M1.router import analyze_user, AnalyzeRequest
        from database import get_db, SessionLocal
        
        db = SessionLocal()
        try:
            req = AnalyzeRequest(userid=int(request.get("userid", 0)))
            return await analyze_user(req, db)
        finally:
            db.close()
    except Exception as e:
        return {"message": f"오류: {str(e)}"}


@app.post("/deleted-track")
async def legacy_deleted_track(request: dict):
    """
    레거시 삭제 엔드포인트 (Spring Boot 호환)
    → /api/m1/deleted-track로 리다이렉트
    """
    try:
        from M1.router import deleted_track, DeleteTrackRequest
        from database import SessionLocal
        
        db = SessionLocal()
        try:
            req = DeleteTrackRequest(
                users_id=int(request.get("users_id", 0)),
                playlists_id=int(request.get("playlists_id", 0)),
                tracks_id=int(request.get("tracks_id", 0))
            )
            return await deleted_track(req, db)
        finally:
            db.close()
    except Exception as e:
        return {"message": f"오류: {str(e)}"}


# ==================== EMS 데이터 분석 (공통) ====================

@app.get("/api/ems/analysis")
async def analyze_ems_data(user_id: int = Query(..., description="사용자 ID")):
    """EMS 데이터 종합 분석 - M1 프로필 조회"""
    try:
        from M1.service import M1RecommendationService
        from database import SessionLocal
        
        model_path = os.path.join(os.path.dirname(__file__), "M1", "audio_predictor.pkl")
        service = M1RecommendationService(model_path=model_path)
        
        db = SessionLocal()
        try:
            profile = service.get_user_profile(db, user_id)
            return {
                "userId": user_id,
                "profile": profile,
                "analysisDate": "2026-02-04"
            }
        finally:
            db.close()
    except Exception as e:
        return {"error": str(e)}


# ==================== 서버 실행 ====================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("ENVIRONMENT", "development") == "development"
    )
