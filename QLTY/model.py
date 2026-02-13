"""
Audio Feature Prediction Model — 3순위

메타데이터(title, artist, album, genre, duration, popularity)로
오디오 피처를 예측하는 딥러닝 + LightGBM 하이브리드 모델.

딥러닝 부분: Sentence-Transformers MiniLM-L6-v2 → 384D 텍스트 임베딩
예측 부분: LightGBM Regressor (피처별 개별 모델)

왜 LightGBM?
- 86K 학습 데이터에서 커스텀 NN보다 정확도 높음 (tabular data 특성)
- 학습 시간 수 초 (PyTorch NN은 수 분)
- sentence-transformers가 이미 딥러닝 파트 담당
"""
import os
import logging
import numpy as np
import joblib
from pathlib import Path
from typing import Optional, Dict, List, Tuple

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent

# 예측 대상 오디오 피처
# Head A: 86K 학습 가능 (spotify_85k에 있음)
HEAD_A_FEATURES = [
    "danceability", "energy", "tempo", "loudness",
    "instrumentalness", "music_key", "mode",
]

# Head B: 1.7K 학습 가능 (high_popularity에만 있음)
HEAD_B_FEATURES = [
    "valence", "acousticness", "speechiness", "liveness",
    "time_signature",
]

ALL_FEATURES = HEAD_A_FEATURES + HEAD_B_FEATURES

# 피처별 값 범위 (예측값 클리핑용)
FEATURE_RANGES = {
    "danceability": (0.0, 1.0),
    "energy": (0.0, 1.0),
    "valence": (0.0, 1.0),
    "acousticness": (0.0, 1.0),
    "instrumentalness": (0.0, 1.0),
    "liveness": (0.0, 1.0),
    "speechiness": (0.0, 1.0),
    "tempo": (40.0, 250.0),
    "loudness": (-60.0, 5.0),
    "music_key": (0, 11),
    "mode": (0, 1),
    "time_signature": (1, 7),
}


class AudioFeatureModel:
    """
    텍스트 임베딩(384D) + 수치 피처(2D) → LightGBM per feature

    입력:
        - text: "artist | track_name | album | genre" 연결 문자열
        - duration_ms: 트랙 길이 (밀리초)
        - popularity: 인기도 (0-100)

    출력:
        - dict: {"danceability": 0.65, "energy": 0.8, ...}
    """

    def __init__(self, model_dir: Optional[str] = None):
        self.model_dir = Path(model_dir) if model_dir else BASE_DIR
        self.embedder = None       # SentenceTransformer 인스턴스
        self.models = {}           # {feature_name: trained_lgbm_model}
        self._loaded = False

    def _load_embedder(self):
        """MiniLM-L6-v2 임베딩 모델 로드 (M2와 동일)"""
        if self.embedder is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("[QLTY Model] SentenceTransformer 로드 완료")
        except Exception as e:
            logger.error(f"[QLTY Model] SentenceTransformer 로드 실패: {e}")
            raise

    def load(self) -> bool:
        """저장된 모델 파일 로드"""
        model_path = self.model_dir / "qlty_models.pkl"
        if not model_path.exists():
            logger.warning(f"[QLTY Model] 모델 파일 없음: {model_path}")
            return False

        try:
            data = joblib.load(model_path)
            self.models = data.get("models", {})
            self._loaded = bool(self.models)
            logger.info(
                f"[QLTY Model] 로드 완료: {list(self.models.keys())}"
            )
            return self._loaded
        except Exception as e:
            logger.error(f"[QLTY Model] 로드 실패: {e}")
            return False

    def save(self) -> str:
        """학습된 모델을 파일로 저장"""
        model_path = self.model_dir / "qlty_models.pkl"
        joblib.dump({"models": self.models}, model_path)
        logger.info(f"[QLTY Model] 저장 완료: {model_path}")
        return str(model_path)

    def _make_input(
        self,
        title: str,
        artist: str,
        album: str = "",
        genre: str = "",
        duration_ms: float = 0,
        popularity: float = 0,
    ) -> np.ndarray:
        """
        텍스트 + 수치 데이터를 하나의 피처 벡터로 결합.

        1. 텍스트 4개를 ' | '로 연결 → MiniLM으로 384D 임베딩
        2. duration_ms를 분 단위로 변환 (스케일 맞춤)
        3. popularity를 0-1로 정규화
        4. 최종: 384D + 2D = 386D 벡터
        """
        self._load_embedder()

        # 텍스트 임베딩 (384D)
        text = f"{artist} | {title} | {album} | {genre}"
        embedding = self.embedder.encode([text])[0]  # shape: (384,)

        # 수치 피처 (2D)
        duration_min = duration_ms / 60000.0 if duration_ms else 0.0
        pop_norm = popularity / 100.0 if popularity else 0.0
        numeric = np.array([duration_min, pop_norm])

        # 결합 (386D)
        return np.concatenate([embedding, numeric]).reshape(1, -1)

    def predict(
        self,
        title: str,
        artist: str,
        album: str = "",
        genre: str = "",
        duration_ms: float = 0,
        popularity: float = 0,
    ) -> Optional[Dict]:
        """
        메타데이터로 오디오 피처를 예측한다.

        Returns:
            {"danceability": 0.65, ...} 또는 None (모델 미로드 시)
        """
        if not self._loaded or not self.models:
            logger.warning("[QLTY Model] 모델이 로드되지 않음")
            return None

        try:
            X = self._make_input(
                title, artist, album, genre, duration_ms, popularity
            )

            predictions = {}
            for feature_name, model in self.models.items():
                raw_pred = model.predict(X)[0]

                # 값 범위 클리핑
                lo, hi = FEATURE_RANGES.get(feature_name, (None, None))
                if lo is not None and hi is not None:
                    raw_pred = np.clip(raw_pred, lo, hi)

                # key, mode, time_signature는 정수로 반올림
                if feature_name in ("music_key", "mode", "time_signature"):
                    raw_pred = int(round(raw_pred))
                else:
                    raw_pred = round(float(raw_pred), 3)

                predictions[feature_name] = raw_pred

            return predictions

        except Exception as e:
            logger.error(f"[QLTY Model] 예측 실패: {e}")
            return None
