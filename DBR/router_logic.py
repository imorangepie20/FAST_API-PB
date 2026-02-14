"""
Divergence-Based Routing — 핵심 라우팅 로직

QLTY(Gemini+Search)와 M1(Ridge Regression) 예측을 비교하여
피처별로 최적 값을 선택하는 7단계 의사결정 트리.

결정 우선순위:
1. div > 0.16        → M1 (QLTY 신뢰 불가)
2. QLTY 강점 장르    → QLTY
3. QLTY 강점 피처    → QLTY
4. div < 0.05        → 평균 (두 모델 합의)
5. div < 0.10        → 가중 평균 (M1:60 QLTY:40)
6. popularity >= 80  → 가중 평균
7. default           → M1 (안전)
"""
import logging
from typing import Dict

from .constants import (
    QLTY_GENRES, QLTY_FEATURES, ROUTING_FEATURES,
    DIV_CRITICAL, DIV_MID, DIV_LOW, POP_HIGH,
)

logger = logging.getLogger(__name__)


def norm_diff(feature: str, diff: float) -> float:
    """피처별 normalized difference (0~1 스케일로 통일)."""
    if feature == "tempo":
        return diff / 250.0
    if feature == "loudness":
        return diff / 60.0
    return diff


def apply_divergence_routing(
    qlty_features: Dict,
    m1_features: Dict,
    genre: str,
    popularity: float,
) -> Dict:
    """
    QLTY와 M1 예측을 비교하여 피처별로 최적 값을 선택한다.

    Args:
        qlty_features: Gemini+Search 예측 결과
        m1_features: M1 Ridge Regression 예측 결과
        genre: 트랙 장르 (소문자)
        popularity: 인기도 (0-100)

    Returns:
        라우팅된 최종 피처 dict (routing 대상 외 피처는 qlty_features 유지)
    """
    routed = dict(qlty_features)  # QLTY 값을 기본으로 복사

    for feat in ROUTING_FEATURES:
        qlty_val = qlty_features.get(feat)
        m1_val = m1_features.get(feat)

        if qlty_val is None or m1_val is None:
            continue

        div = norm_diff(feat, abs(m1_val - qlty_val))

        # 1. High divergence → M1 (QLTY 17% 승률)
        if div > DIV_CRITICAL:
            routed[feat] = m1_val
            continue

        # 2. QLTY 강점 장르 → QLTY 유지
        if genre in QLTY_GENRES:
            continue

        # 3. QLTY 강점 피처 → QLTY 유지
        if feat in QLTY_FEATURES:
            continue

        # 4. Low divergence → 평균 (두 모델 합의)
        if div < DIV_LOW:
            routed[feat] = round((m1_val + qlty_val) / 2, 3)
            continue

        # 5. Mid divergence → 가중 평균 (M1:60 QLTY:40)
        if div < DIV_MID:
            routed[feat] = round(m1_val * 0.6 + qlty_val * 0.4, 3)
            continue

        # 6. High popularity → 가중 평균
        if popularity >= POP_HIGH:
            routed[feat] = round(m1_val * 0.6 + qlty_val * 0.4, 3)
            continue

        # 7. Default → M1
        routed[feat] = m1_val

    return routed
