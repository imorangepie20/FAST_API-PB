"""
QLTY Model Training Pipeline

spotify_reference 테이블의 86K 데이터를 사용하여
오디오 피처 예측 모델을 학습한다.

학습 흐름:
1. DB에서 학습 데이터 로드 (track_name, artist, album, genre, duration, popularity + audio features)
2. SentenceTransformer로 텍스트 → 384D 임베딩
3. 임베딩 + 수치피처 = 386D 입력 벡터
4. 피처별 LightGBM 모델 학습 (Head A: 86K, Head B: 1.7K)
5. 모델 저장
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sqlalchemy import text
from sqlalchemy.orm import Session

from .model import AudioFeatureModel, HEAD_A_FEATURES, HEAD_B_FEATURES, ALL_FEATURES

logger = logging.getLogger(__name__)


def load_training_data(db: Session) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    spotify_reference에서 학습 데이터를 두 그룹으로 로드한다.

    Returns:
        (df_a, df_b):
            df_a — Head A 학습용 (86K, 7개 피처 보유)
            df_b — Head B 학습용 (1.7K, 12개 피처 전부 보유)
    """
    # Head A: danceability가 있는 모든 레코드
    query_a = text("""
        SELECT track_name, artist_name, album_name, genre,
               duration_ms, popularity,
               danceability, energy, tempo, loudness,
               instrumentalness, music_key, mode
        FROM spotify_reference
        WHERE danceability IS NOT NULL
          AND track_name IS NOT NULL
          AND artist_name IS NOT NULL
    """)

    # Head B: valence까지 있는 레코드 (high_popularity 소스)
    query_b = text("""
        SELECT track_name, artist_name, album_name, genre,
               duration_ms, popularity,
               valence, acousticness, speechiness, liveness,
               time_signature
        FROM spotify_reference
        WHERE valence IS NOT NULL
          AND track_name IS NOT NULL
          AND artist_name IS NOT NULL
    """)

    df_a = pd.read_sql(query_a, db.bind)
    df_b = pd.read_sql(query_b, db.bind)

    logger.info(f"[QLTY Train] Head A 데이터: {len(df_a)}행")
    logger.info(f"[QLTY Train] Head B 데이터: {len(df_b)}행")

    return df_a, df_b


def generate_embeddings(
    model: AudioFeatureModel,
    df: pd.DataFrame,
    batch_size: int = 256,
) -> np.ndarray:
    """
    DataFrame의 텍스트 컬럼을 384D 임베딩으로 변환하고,
    수치 피처 2개를 결합하여 386D 입력 행렬을 만든다.

    Args:
        model: AudioFeatureModel (embedder 사용)
        df: 학습 데이터
        batch_size: 임베딩 배치 크기 (메모리 관리)

    Returns:
        np.ndarray shape (N, 386)
    """
    model._load_embedder()

    # 텍스트 조합: "artist | title | album | genre"
    texts = (
        df["artist_name"].fillna("")
        + " | "
        + df["track_name"].fillna("")
        + " | "
        + df["album_name"].fillna("")
        + " | "
        + df["genre"].fillna("")
    ).tolist()

    # 배치 임베딩 (메모리 효율)
    logger.info(f"[QLTY Train] {len(texts)}건 텍스트 임베딩 생성 중...")
    embeddings = model.embedder.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
    )  # shape: (N, 384)

    # 수치 피처
    duration_min = (df["duration_ms"].fillna(0) / 60000.0).values.reshape(-1, 1)
    pop_norm = (df["popularity"].fillna(0) / 100.0).values.reshape(-1, 1)

    # 결합: (N, 384) + (N, 1) + (N, 1) = (N, 386)
    X = np.hstack([embeddings, duration_min, pop_norm])

    logger.info(f"[QLTY Train] 입력 행렬: {X.shape}")
    return X


def train_models(db: Session) -> Dict:
    """
    전체 학습 파이프라인을 실행한다.

    Returns:
        {"model_path": "...", "metrics": {...}} 학습 결과 리포트
    """
    # LightGBM import (requirements.txt에 이미 있음)
    import lightgbm as lgb

    model = AudioFeatureModel()
    df_a, df_b = load_training_data(db)

    if len(df_a) < 100:
        raise ValueError(f"학습 데이터 부족: Head A {len(df_a)}행 (최소 100)")

    metrics = {}

    # ========== Head A 학습 (86K, 7개 피처) ==========
    logger.info("[QLTY Train] === Head A 학습 시작 ===")
    X_a = generate_embeddings(model, df_a)

    for feature in HEAD_A_FEATURES:
        y = df_a[feature].values.astype(float)

        # NaN 행 제거
        valid_mask = ~np.isnan(y)
        X_valid = X_a[valid_mask]
        y_valid = y[valid_mask]

        if len(y_valid) < 50:
            logger.warning(f"[QLTY Train] {feature}: 데이터 부족 ({len(y_valid)}), 스킵")
            continue

        # 80/20 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X_valid, y_valid, test_size=0.2, random_state=42
        )

        # LightGBM 학습
        lgb_model = lgb.LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=8,
            num_leaves=63,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,  # 로그 억제
        )
        lgb_model.fit(X_train, y_train)

        # 평가
        y_pred = lgb_model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        model.models[feature] = lgb_model
        metrics[feature] = {"mae": round(mae, 4), "r2": round(r2, 4), "n_train": len(y_train)}
        logger.info(f"  {feature}: MAE={mae:.4f}, R²={r2:.4f} (N={len(y_train)})")

    # ========== Head B 학습 (1.7K, 5개 피처) ==========
    if len(df_b) >= 50:
        logger.info("[QLTY Train] === Head B 학습 시작 ===")
        X_b = generate_embeddings(model, df_b)

        for feature in HEAD_B_FEATURES:
            y = df_b[feature].values.astype(float)

            valid_mask = ~np.isnan(y)
            X_valid = X_b[valid_mask]
            y_valid = y[valid_mask]

            if len(y_valid) < 30:
                logger.warning(f"[QLTY Train] {feature}: 데이터 부족 ({len(y_valid)}), 스킵")
                continue

            X_train, X_test, y_train, y_test = train_test_split(
                X_valid, y_valid, test_size=0.2, random_state=42
            )

            # Head B는 데이터 적으므로 오버피팅 방지 설정
            lgb_model = lgb.LGBMRegressor(
                n_estimators=150,        # 적은 반복
                learning_rate=0.03,      # 느린 학습
                max_depth=5,             # 얕은 트리
                num_leaves=31,
                min_child_samples=10,
                subsample=0.7,
                colsample_bytree=0.7,
                reg_alpha=0.1,           # L1 정규화
                reg_lambda=0.1,          # L2 정규화
                random_state=42,
                verbose=-1,
            )
            lgb_model.fit(X_train, y_train)

            y_pred = lgb_model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            model.models[feature] = lgb_model
            metrics[feature] = {"mae": round(mae, 4), "r2": round(r2, 4), "n_train": len(y_train)}
            logger.info(f"  {feature}: MAE={mae:.4f}, R²={r2:.4f} (N={len(y_train)})")
    else:
        logger.warning(f"[QLTY Train] Head B 스킵: 데이터 {len(df_b)}행 (최소 50)")

    # 모델 저장
    model._loaded = True
    model_path = model.save()

    result = {
        "model_path": model_path,
        "metrics": metrics,
        "total_features": len(model.models),
        "head_a_data": len(df_a),
        "head_b_data": len(df_b),
    }

    logger.info(f"[QLTY Train] 학습 완료: {len(model.models)}개 피처 모델")
    return result
