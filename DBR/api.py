"""
DBR API — Divergence-Based Routing FastAPI 엔드포인트

- POST /api/dbr/route  : DBR 단독 테스트 (QLTY/M1 피처 입력 → 라우팅 결과)
- GET  /api/dbr/health : M1 모델 상태 + DBR 설정 확인
"""
import logging
from typing import Dict, Optional
from fastapi import APIRouter
from pydantic import BaseModel, Field

from .router_logic import apply_divergence_routing
from . import m1_predictor
from .constants import (
    QLTY_GENRES, QLTY_FEATURES, ROUTING_FEATURES,
    DIV_CRITICAL, DIV_MID, DIV_LOW, POP_HIGH,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/dbr", tags=["DBR"])


class RouteRequest(BaseModel):
    """DBR 단독 테스트 요청"""
    qlty_features: Dict[str, float] = Field(..., description="QLTY(Gemini+Search) 예측 결과")
    m1_features: Dict[str, float] = Field(..., description="M1(Ridge) 예측 결과")
    genre: str = Field("", description="트랙 장르 (소문자)")
    popularity: float = Field(0, description="인기도 (0-100)")


class RouteResponse(BaseModel):
    """DBR 라우팅 결과"""
    routed_features: Dict[str, float]
    decisions: Dict[str, str]


class PredictRequest(BaseModel):
    """M1 예측 + DBR 라우팅 요청 (QLTY 피처만 제공)"""
    title: str
    artist: str
    album: str = ""
    genre: str = ""
    duration_ms: float = 0
    popularity: float = 0
    qlty_features: Dict[str, float] = Field(..., description="QLTY(Gemini+Search) 예측 결과")


@router.post("/route", response_model=RouteResponse)
async def route_features(request: RouteRequest):
    """
    DBR 단독 테스트.
    QLTY와 M1 피처를 직접 입력하여 라우팅 결과를 확인한다.
    """
    routed = apply_divergence_routing(
        request.qlty_features,
        request.m1_features,
        genre=request.genre.lower(),
        popularity=request.popularity,
    )

    # 각 피처별 어떤 결정이 내려졌는지 추적
    decisions = {}
    for feat in ROUTING_FEATURES:
        qlty_val = request.qlty_features.get(feat)
        m1_val = request.m1_features.get(feat)
        routed_val = routed.get(feat)

        if qlty_val is None or m1_val is None:
            decisions[feat] = "skip (missing)"
        elif routed_val == qlty_val:
            decisions[feat] = "qlty"
        elif routed_val == m1_val:
            decisions[feat] = "m1"
        else:
            decisions[feat] = "blended"

    return RouteResponse(
        routed_features=routed,
        decisions=decisions,
    )


@router.post("/predict-and-route")
async def predict_and_route(request: PredictRequest):
    """
    M1 예측을 자동 수행한 후 DBR 라우팅을 적용한다.
    QLTY 피처만 제공하면 M1 예측 → DBR 비교 → 최적 결과 반환.
    """
    m1_features = m1_predictor.predict(
        title=request.title,
        artist=request.artist,
        album=request.album,
        genre=request.genre,
        duration_ms=request.duration_ms,
        popularity=request.popularity,
    )

    if not m1_features:
        return {
            "success": False,
            "error": "M1 모델 로드 실패 또는 예측 불가",
            "fallback": request.qlty_features,
        }

    routed = apply_divergence_routing(
        request.qlty_features,
        m1_features,
        genre=request.genre.lower(),
        popularity=request.popularity,
    )

    return {
        "success": True,
        "qlty_features": request.qlty_features,
        "m1_features": m1_features,
        "routed_features": routed,
    }


@router.get("/health")
async def dbr_health():
    """DBR 모듈 상태 확인."""
    m1_ready = m1_predictor.is_loaded()

    return {
        "status": "ready" if m1_ready else "partial",
        "m1_model_loaded": m1_ready,
        "config": {
            "qlty_genres": sorted(QLTY_GENRES),
            "qlty_features": sorted(QLTY_FEATURES),
            "routing_features": sorted(ROUTING_FEATURES),
            "thresholds": {
                "div_critical": DIV_CRITICAL,
                "div_mid": DIV_MID,
                "div_low": DIV_LOW,
                "pop_high": POP_HIGH,
            },
        },
    }
