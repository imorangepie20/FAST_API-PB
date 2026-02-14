"""
DBR (Divergence-Based Routing) 모듈

QLTY(Gemini+Search)와 M1(Ridge Regression) 예측의 차이(divergence)를
기준으로 피처별 최적 값을 선택하는 지능형 라우팅 시스템.

사용법:
    from DBR import apply_divergence_routing, m1_predictor
    routed = apply_divergence_routing(qlty_features, m1_features, genre, popularity)
"""
from .router_logic import apply_divergence_routing, norm_diff
from . import m1_predictor
from .constants import (
    QLTY_GENRES, QLTY_FEATURES, ROUTING_FEATURES,
    DIV_CRITICAL, DIV_MID, DIV_LOW, POP_HIGH,
)
