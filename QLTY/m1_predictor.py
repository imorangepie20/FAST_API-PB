"""
M1 Predictor Wrapper — Divergence-Based Routing용

M1의 AudioFeaturePredictor(Ridge Regression)를 로드하여
QLTY 파이프라인에서 divergence 비교에 사용한다.

M1 모델 파일: /app/M1/audio_predictor.pkl
"""
import logging
import os
from pathlib import Path
from typing import Optional, Dict

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

_predictor = None
_load_attempted = False

# M1이 예측하는 피처 (QLTY와 비교할 8개)
M1_FEATURES = [
    "danceability", "energy", "valence", "tempo",
    "acousticness", "instrumentalness", "speechiness", "loudness",
]


def _get_predictor():
    """M1 AudioFeaturePredictor를 lazy-load한다."""
    global _predictor, _load_attempted
    if _predictor is not None:
        return _predictor
    if _load_attempted:
        return None

    _load_attempted = True

    model_path = Path(__file__).resolve().parent.parent / "M1" / "audio_predictor.pkl"
    if not model_path.exists():
        logger.warning(f"[QLTY M1] M1 모델 파일 없음: {model_path}")
        return None

    try:
        import sys
        sys.path.insert(0, str(model_path.parent.parent))
        from M1.spotify_recommender import AudioFeaturePredictor

        predictor = AudioFeaturePredictor()
        predictor.load(str(model_path))
        _predictor = predictor
        logger.info("[QLTY M1] M1 AudioFeaturePredictor 로드 완료")
        return _predictor
    except Exception as e:
        logger.error(f"[QLTY M1] M1 모델 로드 실패: {e}")
        return None


def predict(
    title: str,
    artist: str,
    album: str = "",
    genre: str = "",
    duration_ms: float = 0,
    popularity: float = 0,
) -> Optional[Dict]:
    """
    M1 Ridge Regression으로 오디오 피처를 예측한다.

    Returns:
        {"danceability": 0.65, "energy": 0.72, ...} 또는 None
    """
    predictor = _get_predictor()
    if predictor is None:
        return None

    try:
        df = pd.DataFrame([{
            "title": title,
            "artists": artist,
            "album_name": album,
            "track_genre": genre,
            "duration_ms": duration_ms,
            "popularity": popularity,
        }])

        predictions = predictor.predict(df)

        result = {}
        for feat in M1_FEATURES:
            col = f"predicted_{feat}"
            if col in predictions.columns:
                val = float(predictions.iloc[0][col])
                # 0~1 범위 피처 클리핑
                if feat not in ("tempo", "loudness"):
                    val = max(0.0, min(1.0, val))
                result[feat] = round(val, 3)

        return result if result else None

    except Exception as e:
        logger.error(f"[QLTY M1] 예측 실패: {e}")
        return None
