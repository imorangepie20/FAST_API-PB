"""
M3 Service - CatBoost 기반 협업 필터링 추천 서비스
m3.py의 핵심 로직을 서비스 클래스로 정리

주요 기능:
- PMS 플레이리스트 분석
- CatBoost 모델로 사용자 취향 벡터 생성
- 유클리드 거리 기반 EMS 트랙 추천
"""
import os
import glob
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR
DATASET_PATH = BASE_DIR.parent / 'dataset' / 'dataset.csv'
USER_MODELS_DIR = BASE_DIR / 'user_models'
USER_MODELS_DIR.mkdir(parents=True, exist_ok=True)

# 피처 컬럼
FEATURES = ['artists', 'album_name', 'track_genre']
TARGET_COLUMNS = [
    'danceability', 'energy', 'key', 'loudness', 'mode',
    'speechiness', 'acousticness', 'instrumentalness', 'liveness'
]


class M3RecommendationService:
    """M3 CatBoost 기반 추천 서비스"""
    
    def __init__(self):
        self.model = None
        self.df = None  # EMS 데이터셋
        self._model_loaded = False
    
    def _load_dataset(self) -> bool:
        """EMS 데이터셋 로드"""
        if self.df is not None:
            return True
        
        if not DATASET_PATH.exists():
            logger.warning(f"데이터셋 파일 없음: {DATASET_PATH}")
            return False
        
        try:
            self.df = pd.read_csv(DATASET_PATH)
            for col in FEATURES:
                self.df[col] = self.df[col].fillna('unknown').astype(str)
            for col in TARGET_COLUMNS:
                if col in self.df.columns:
                    self.df[col] = self.df[col].fillna(0)
            
            logger.info(f"데이터셋 로드 완료: {len(self.df)} 곡")
            return True
        except Exception as e:
            logger.error(f"데이터셋 로드 실패: {e}")
            return False
    
    def _get_latest_model_path(self, user_id: int) -> Optional[str]:
        """사용자의 최신 모델 파일 경로 조회 (이메일 기반)"""
        try:
            from database import SessionLocal
            from sqlalchemy import text

            db = SessionLocal()
            try:
                query = text("SELECT email FROM users WHERE user_id = :user_id")
                result = db.execute(query, {"user_id": user_id}).fetchone()
                if result:
                    email_prefix = result[0].split('@')[0]
                    model_path = USER_MODELS_DIR / f"{email_prefix}_.cbm"
                    if model_path.exists():
                        return str(model_path)
                return None
            except Exception as e:
                logger.error(f"사용자 모델 경로 조회 실패: {e}")
                return None
            finally:
                db.close()
        except Exception as e:
            logger.error(f"DB 연결 실패: {e}")
            return None
    
    def _get_any_model_path(self) -> Optional[str]:
        """아무 모델 파일이나 반환 (기본 모델용)"""
        search_pattern = str(MODEL_DIR / "*.cbm")
        model_files = glob.glob(search_pattern)
        
        if not model_files:
            return None
        
        return model_files[0]
    
    def _load_model(self, model_path: str) -> bool:
        """CatBoost 모델 로드"""
        try:
            from catboost import CatBoostRegressor
            
            self.model = CatBoostRegressor()
            self.model.load_model(model_path)
            self._model_loaded = True
            logger.info(f"CatBoost 모델 로드 완료: {model_path}")
            return True
        except Exception as e:
            logger.error(f"CatBoost 모델 로드 실패: {e}")
            return False
    
    def train_user_model(
        self,
        db,
        user_id: int,
        playlist_title: str = "MyPlaylist"
    ) -> Dict:
        """
        사용자 CatBoost 모델 학습 (PMS 데이터 기반)
        - M1과 동일한 구조: PMS 트랙으로 개인화 모델 학습
        - 저장 위치: user_models/{email_prefix}_.cbm
        """
        from sqlalchemy import text

        # 1. 사용자 이메일 조회
        try:
            query = text("SELECT email FROM users WHERE user_id = :user_id")
            result = db.execute(query, {"user_id": user_id}).fetchone()
            if not result:
                return {"success": False, "message": "사용자를 찾을 수 없습니다"}
            email_prefix = result[0].split('@')[0]
            model_path = USER_MODELS_DIR / f"{email_prefix}_.cbm"
        except Exception as e:
            logger.error(f"사용자 이메일 조회 실패: {e}")
            return {"success": False, "message": f"사용자 정보 조회 실패: {e}"}

        # 2. PMS 트랙 조회 (사용자 선호 데이터)
        pms_query = text("""
            SELECT t.track_id, t.title as track_name, t.artist as artists,
                   t.album as album_name, COALESCE(t.genre, 'unknown') as track_genre,
                   COALESCE(t.duration, 200) as duration
            FROM tracks t
            JOIN playlist_tracks pt ON t.track_id = pt.track_id
            JOIN playlists p ON pt.playlist_id = p.playlist_id
            WHERE p.user_id = :user_id AND p.space_type = 'PMS'
        """)
        pms_result = db.execute(pms_query, {"user_id": user_id}).fetchall()

        if not pms_result or len(pms_result) < 5:
            return {
                "success": False,
                "message": f"PMS 데이터 부족 ({len(pms_result) if pms_result else 0}개, 최소 5개 필요)"
            }

        print(f"[M3] 사용자 {user_id}의 PMS 트랙 {len(pms_result)}곡 로드")

        # 3. 데이터프레임 생성 (PMS만 사용 - M1과 동일)
        columns = ['track_id', 'track_name', 'artists', 'album_name', 'track_genre', 'duration']
        self.df = pd.DataFrame(pms_result, columns=columns)
        print(f"[M3] 학습 데이터: PMS {len(pms_result)}곡 (M1과 동일하게 PMS만 사용)")
        self.df['duration_ms'] = self.df['duration'] * 1000
        self.df['popularity'] = 50

        for col in FEATURES:
            self.df[col] = self.df[col].fillna('unknown').astype(str)

        # 5. M1 AudioFeaturePredictor로 audio features 예측
        try:
            from M1.spotify_recommender import AudioFeaturePredictor

            m1_model_path = BASE_DIR.parent / "M1" / "audio_predictor.pkl"
            predictor = AudioFeaturePredictor(model_type='Ridge')

            if m1_model_path.exists():
                predictor.load(str(m1_model_path))
                print(f"[M3] M1 모델 로드 완료: {m1_model_path}")

            predicted_df = predictor.predict(self.df)

            feature_mapping = {
                'danceability': 'predicted_danceability',
                'energy': 'predicted_energy',
                'loudness': 'predicted_loudness',
                'speechiness': 'predicted_speechiness',
                'acousticness': 'predicted_acousticness',
                'instrumentalness': 'predicted_instrumentalness',
                'liveness': 'predicted_liveness',
            }

            for target_col, pred_col in feature_mapping.items():
                if pred_col in predicted_df.columns:
                    values = predicted_df[pred_col].values
                    if np.std(values) < 0.01:
                        noise = np.random.normal(0, 0.1, len(values))
                        values = np.clip(values + noise, 0, 1)
                    self.df[target_col] = values
                else:
                    self.df[target_col] = 0.5

            # key, mode 다양성 부여
            self.df['key'] = self.df.apply(
                lambda x: (hash(str(x['artists']) + str(x['track_genre'])) % 12), axis=1
            )
            self.df['mode'] = self.df.apply(
                lambda x: (hash(str(x['artists'])) % 2), axis=1
            )

            # 다양성 추가
            for col in TARGET_COLUMNS:
                if col in self.df.columns:
                    values = self.df[col].values.astype(float)
                    noise = np.random.normal(0, 0.15, len(values))
                    if col == 'key':
                        values = (values + np.random.randint(0, 3, len(values))) % 12
                    elif col == 'loudness':
                        values = values + np.random.normal(0, 2, len(values))
                    elif col != 'mode':
                        values = np.clip(values + noise, 0, 1)
                    self.df[col] = values

            print(f"[M3] M1 audio features 예측 완료: {len(self.df)}곡")

        except Exception as e:
            logger.error(f"M1 예측 실패, 기본값 사용: {e}")
            for col in TARGET_COLUMNS:
                if col not in self.df.columns:
                    self.df[col] = 0.5 if col not in ['key', 'loudness', 'mode'] else 0

        for col in TARGET_COLUMNS:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna(0)

        # 6. CatBoost 모델 학습
        try:
            from catboost import CatBoostRegressor, Pool
            from sklearn.model_selection import train_test_split

            # 학습 데이터 준비
            train_df, eval_df = train_test_split(self.df, test_size=0.3, random_state=42)

            train_pool = Pool(
                data=train_df[FEATURES],
                label=train_df[TARGET_COLUMNS],
                cat_features=FEATURES
            )
            eval_pool = Pool(
                data=eval_df[FEATURES],
                label=eval_df[TARGET_COLUMNS],
                cat_features=FEATURES
            )

            # 모델 학습
            new_model = CatBoostRegressor(
                iterations=500,
                learning_rate=0.05,
                depth=6,
                loss_function='MultiRMSE',
                random_seed=42,
                verbose=False
            )
            new_model.fit(train_pool, eval_set=eval_pool, early_stopping_rounds=50)

            # 모델 저장 (이메일 기반 경로)
            new_model.save_model(str(model_path))

            # 현재 모델 업데이트
            self.model = new_model
            self._model_loaded = True

            logger.info(f"사용자 {user_id} 모델 학습 완료: {model_path}")

            return {
                "success": True,
                "message": f"사용자 {user_id} 모델 학습 완료",
                "user_id": user_id,
                "model_path": str(model_path),
                "metrics": {
                    "iterations": 500,
                    "learning_rate": 0.05,
                    "depth": 6,
                    "loss_function": "MultiRMSE"
                }
            }

        except Exception as e:
            logger.error(f"모델 학습 오류: {e}")
            return {
                "success": False,
                "message": f"학습 오류: {str(e)}"
            }
    
    def get_recommendations(
        self,
        db,
        user_id: int,
        top_k: int = 50,
        track_ids: list = None
    ) -> Dict:
        """추천 트랙 생성

        Args:
            db: 데이터베이스 세션
            user_id: 사용자 ID
            top_k: 추천할 트랙 수
            track_ids: 특정 트랙 ID 목록 (데모 페이지에서 동일 트랙 비교용)
        """
        from sqlalchemy import text

        # 데이터셋 로드 (없으면 DB에서 EMS 트랙 사용)
        if not self._load_dataset():
            logger.warning("외부 데이터셋 없음, DB EMS 트랙 + M1 Audio Predictor 사용")

            # track_ids가 제공되면 해당 트랙만 조회 (데모 페이지용)
            if track_ids and len(track_ids) > 0:
                track_ids_str = ','.join(map(str, track_ids))
                ems_query = text(f"""
                    SELECT t.track_id, t.title as track_name, t.artist as artists,
                           t.album as album_name, COALESCE(t.genre, 'unknown') as track_genre,
                           COALESCE(t.duration, 200) as duration
                    FROM tracks t
                    WHERE t.track_id IN ({track_ids_str})
                """)
                ems_result = db.execute(ems_query).fetchall()
                logger.info(f"특정 트랙 ID로 조회: {len(ems_result)}곡 (요청: {len(track_ids)}개)")
            else:
                # 기존 EMS 랜덤 쿼리
                ems_query = text("""
                    SELECT t.track_id, t.title as track_name, t.artist as artists,
                           t.album as album_name, COALESCE(t.genre, 'unknown') as track_genre,
                           COALESCE(t.duration, 200) as duration
                    FROM tracks t
                    JOIN playlist_tracks pt ON t.track_id = pt.track_id
                    JOIN playlists p ON pt.playlist_id = p.playlist_id
                    WHERE p.space_type = 'EMS'
                    ORDER BY RAND()
                    LIMIT 500
                """)
                ems_result = db.execute(ems_query).fetchall()

            if not ems_result:
                return {
                    "success": False,
                    "message": "EMS 데이터가 없습니다",
                    "recommendations": []
                }

            # 데이터프레임 생성
            columns = ['track_id', 'track_name', 'artists', 'album_name', 'track_genre', 'duration']
            self.df = pd.DataFrame(ems_result, columns=columns)
            self.df['duration_ms'] = self.df['duration'] * 1000
            self.df['popularity'] = 50  # M1이 필요로 하는 popularity 컬럼

            for col in FEATURES:
                self.df[col] = self.df[col].fillna('unknown').astype(str)

            logger.info(f"DB에서 EMS 트랙 {len(self.df)}곡 로드, M1으로 audio features 예측 시작")

            # M1 AudioFeaturePredictor로 audio features 예측
            try:
                from M1.spotify_recommender import AudioFeaturePredictor

                m1_model_path = BASE_DIR.parent / "M1" / "audio_predictor.pkl"
                predictor = AudioFeaturePredictor(model_type='Ridge')

                if m1_model_path.exists():
                    predictor.load(str(m1_model_path))
                    logger.info(f"M1 모델 로드 완료: {m1_model_path}")
                else:
                    logger.warning(f"M1 모델 없음: {m1_model_path}, 기본 예측 사용")

                # M1으로 audio features 예측
                predicted_df = predictor.predict(self.df)

                # 예측된 audio features를 TARGET_COLUMNS에 매핑
                feature_mapping = {
                    'danceability': 'predicted_danceability',
                    'energy': 'predicted_energy',
                    'loudness': 'predicted_loudness',
                    'speechiness': 'predicted_speechiness',
                    'acousticness': 'predicted_acousticness',
                    'instrumentalness': 'predicted_instrumentalness',
                    'liveness': 'predicted_liveness',
                }

                for target_col, pred_col in feature_mapping.items():
                    if pred_col in predicted_df.columns:
                        values = predicted_df[pred_col].values
                        # 예측값 분산이 너무 작으면 노이즈 추가
                        if np.std(values) < 0.01:
                            logger.info(f"{target_col} 분산 부족, 다양성 추가")
                            noise = np.random.normal(0, 0.1, len(values))
                            values = np.clip(values + noise, 0, 1)
                        self.df[target_col] = values
                    else:
                        self.df[target_col] = 0.5  # 기본값

                # key, mode 다양성 부여
                self.df['key'] = self.df.apply(
                    lambda x: (hash(str(x['artists']) + str(x['track_genre'])) % 12), axis=1
                )
                self.df['mode'] = self.df.apply(
                    lambda x: (hash(str(x['artists'])) % 2), axis=1
                )

                # 모든 target에 강제 다양성 추가
                logger.info("M3 추천용: 모든 target에 다양성 추가 중...")
                for col in TARGET_COLUMNS:
                    if col in self.df.columns:
                        values = self.df[col].values.astype(float)
                        noise = np.random.normal(0, 0.15, len(values))
                        if col in ['key']:
                            values = (values + np.random.randint(0, 3, len(values))) % 12
                        elif col in ['mode']:
                            pass
                        elif col in ['loudness']:
                            values = values + np.random.normal(0, 2, len(values))
                        else:
                            values = np.clip(values + noise, 0, 1)
                        self.df[col] = values

                logger.info(f"M1 audio features 예측 완료: {len(self.df)}곡")

            except Exception as e:
                logger.error(f"M1 예측 실패, 기본값 사용: {e}")
                # M1 실패 시 기본값 사용
                for col in TARGET_COLUMNS:
                    if col not in self.df.columns:
                        self.df[col] = 0.5 if col not in ['key', 'loudness', 'mode'] else 0

            for col in TARGET_COLUMNS:
                if col in self.df.columns:
                    self.df[col] = self.df[col].fillna(0)
        
        # 모델 로드
        model_path = self._get_latest_model_path(user_id)
        if not model_path:
            model_path = self._get_any_model_path()
        
        if not model_path:
            return {
                "success": False,
                "message": "사용 가능한 모델이 없습니다",
                "recommendations": []
            }
        
        if not self._load_model(model_path):
            return {
                "success": False,
                "message": "모델 로드 실패",
                "recommendations": []
            }
        
        try:
            # PMS 트랙 조회
            query = text("""
                SELECT p.title, t.title as track_title, t.artist, t.album, t.external_metadata
                FROM tracks t
                JOIN playlist_tracks pt ON t.track_id = pt.track_id
                JOIN playlists p ON pt.playlist_id = p.playlist_id
                WHERE p.user_id = :uid AND p.space_type = 'PMS'
            """)
            result = db.execute(query, {"uid": user_id}).fetchall()
            
            if not result:
                return {
                    "success": False,
                    "message": "분석할 PMS 데이터가 없습니다",
                    "recommendations": []
                }
            
            pms_tracks = pd.DataFrame(result, columns=['title', 'track_title', 'artist', 'album', 'external_metadata'])
            
            # 장르 추출
            def safe_get_genre(meta):
                try:
                    if meta:
                        return json.loads(meta).get('genre', 'unknown')
                except:
                    pass
                return 'unknown'
            
            pms_tracks['track_genre'] = pms_tracks['external_metadata'].apply(safe_get_genre)
            
            # EMS 트랙의 실제 오디오 피처 확인
            ems_has_audio = (self.df[TARGET_COLUMNS].notna().sum(axis=1) > 3).sum()
            logger.info(f"EMS 트랙 중 오디오 피처 보유: {ems_has_audio}/{len(self.df)}")
            
            # 오디오 피처 기반 추천 (PMS 오디오 프로파일 생성)
            # PMS 오디오 피처 평균 계산 (DB에서 조회)
            pms_audio_query = text("""
                SELECT AVG(t.danceability) as danceability,
                       AVG(t.energy) as energy,
                       AVG(t.music_key) as `key`,
                       AVG(t.loudness) as loudness,
                       AVG(t.mode) as mode,
                       AVG(t.speechiness) as speechiness,
                       AVG(t.acousticness) as acousticness,
                       AVG(t.instrumentalness) as instrumentalness,
                       AVG(t.liveness) as liveness,
                       COUNT(t.danceability) as has_features_count
                FROM tracks t
                JOIN playlist_tracks pt ON t.track_id = pt.track_id
                JOIN playlists p ON pt.playlist_id = p.playlist_id
                WHERE p.user_id = :uid AND p.space_type = 'PMS'
            """)
            pms_audio_result = db.execute(pms_audio_query, {"uid": user_id}).fetchone()

            # PMS 오디오 피처가 있는지 확인
            has_features = pms_audio_result and pms_audio_result[-1] and pms_audio_result[-1] > 0

            if has_features:
                # DB에 실제 오디오 피처가 있으면 사용
                user_taste_vector = np.array([
                    float(pms_audio_result[0] or 0.5),
                    float(pms_audio_result[1] or 0.5),
                    float(pms_audio_result[2] or 0),
                    float(pms_audio_result[3] or -6.0),
                    float(pms_audio_result[4] or 0),
                    float(pms_audio_result[5] or 0.1),
                    float(pms_audio_result[6] or 0.3),
                    float(pms_audio_result[7] or 0.1),
                    float(pms_audio_result[8] or 0.2),
                ])
                logger.info(f"PMS 오디오 프로파일 (DB): {user_taste_vector}")
            else:
                # PMS 오디오 피처가 없으면 M1 Audio Predictor로 예측
                logger.info("PMS 오디오 피처 없음, M1 Audio Predictor로 예측")
                try:
                    from M1.spotify_recommender import AudioFeaturePredictor

                    m1_model_path = BASE_DIR.parent / "M1" / "audio_predictor.pkl"
                    predictor = AudioFeaturePredictor(model_type='Ridge')

                    if m1_model_path.exists():
                        predictor.load(str(m1_model_path))

                        # PMS 트랙으로 데이터프레임 생성
                        pms_df = pms_tracks[['track_title', 'artist', 'album']].copy()
                        pms_df.columns = ['track_name', 'artists', 'album_name']
                        pms_df['duration_ms'] = 200000
                        pms_df['popularity'] = 50
                        pms_df['track_genre'] = pms_tracks['track_genre']

                        # M1로 예측
                        predicted_pms = predictor.predict(pms_df)

                        # 평균 벡터 계산
                        user_taste_vector = np.array([
                            predicted_pms['predicted_danceability'].mean() if 'predicted_danceability' in predicted_pms else 0.5,
                            predicted_pms['predicted_energy'].mean() if 'predicted_energy' in predicted_pms else 0.5,
                            hash(str(pms_tracks['artist'].iloc[0])) % 12,  # key
                            predicted_pms['predicted_loudness'].mean() if 'predicted_loudness' in predicted_pms else -6.0,
                            hash(str(pms_tracks['artist'].iloc[0])) % 2,   # mode
                            predicted_pms['predicted_speechiness'].mean() if 'predicted_speechiness' in predicted_pms else 0.1,
                            predicted_pms['predicted_acousticness'].mean() if 'predicted_acousticness' in predicted_pms else 0.3,
                            predicted_pms['predicted_instrumentalness'].mean() if 'predicted_instrumentalness' in predicted_pms else 0.1,
                            predicted_pms['predicted_liveness'].mean() if 'predicted_liveness' in predicted_pms else 0.2,
                        ])
                        logger.info(f"PMS 오디오 프로파일 (M1 예측): {user_taste_vector}")
                    else:
                        # M1 모델 없으면 CatBoost 사용
                        pms_input = pms_tracks[['artist', 'album', 'track_genre']].fillna('unknown').astype(str)
                        pms_input.columns = FEATURES
                        pms_predictions = self.model.predict(pms_input)
                        user_taste_vector = pms_predictions.mean(axis=0)
                        logger.info(f"PMS 오디오 프로파일 (CatBoost): {user_taste_vector}")

                except Exception as e:
                    logger.error(f"PMS M1 예측 실패, CatBoost 사용: {e}")
                    pms_input = pms_tracks[['artist', 'album', 'track_genre']].fillna('unknown').astype(str)
                    pms_input.columns = FEATURES
                    pms_predictions = self.model.predict(pms_input)
                    user_taste_vector = pms_predictions.mean(axis=0)
            
            # EMS 오디오 피처 행렬
            ems_audio_matrix = self.df[TARGET_COLUMNS].values.astype(float)
            
            # 유클리드 거리 계산
            distances = cdist([user_taste_vector], ems_audio_matrix, metric='euclidean')[0]
            
            # PMS 아티스트 목록 (아티스트 유사도 보너스)
            pms_artists = set(pms_tracks['artist'].str.lower().tolist())
            
            # 아티스트 보너스 적용 (동일 아티스트면 거리 감소)
            for i, row in self.df.iterrows():
                ems_artist = str(row.get('artists', '')).lower()
                if ems_artist in pms_artists:
                    distances[i] *= 0.5  # 동일 아티스트는 50% 거리 감소
            
            # 상위 N개 선택 (중복 제거 후 top_k 보장을 위해 여유분 조회)
            top_indices = np.argsort(distances)[:top_k * 2]
            recommended_tracks = self.df.iloc[top_indices].copy()
            recommended_tracks['distance'] = distances[top_indices]

            # 중복 제거 (track_id 기준)
            before_dedup = len(recommended_tracks)
            recommended_tracks = recommended_tracks.drop_duplicates(subset=['track_id'], keep='first')

            # 아티스트+제목 기준 중복 제거 (같은 곡이 다른 track_id로 존재할 수 있음)
            recommended_tracks['artist_title_key'] = recommended_tracks['artists'].str.lower() + '|' + recommended_tracks['track_name'].str.lower()
            recommended_tracks = recommended_tracks.drop_duplicates(subset=['artist_title_key'], keep='first')
            recommended_tracks = recommended_tracks.drop(columns=['artist_title_key'])

            after_dedup = len(recommended_tracks)
            if before_dedup != after_dedup:
                logger.info(f"[M3] 중복 제거: {before_dedup}곡 → {after_dedup}곡")

            # top_k개로 제한
            recommended_tracks = recommended_tracks.head(top_k)

            # Distance를 Score로 변환 (거리가 작을수록 점수가 높음)
            max_dist = float(distances.max()) if len(distances) > 0 and distances.max() > 0 else 1.0
            min_dist = float(distances.min()) if len(distances) > 0 else 0.0
            dist_range = max_dist - min_dist
            
            logger.info(f"M3 거리 범위: min={min_dist:.4f}, max={max_dist:.4f}, range={dist_range:.4f}")
            
            # 결과 포맷팅
            recommendations = []
            total_tracks = len(recommended_tracks) if recommended_tracks is not None else 0
            
            for rank, (idx, row) in enumerate(recommended_tracks.iterrows(), 1):
                dist = float(row['distance'])
                
                # 거리 범위가 유의미하면 거리 기반, 아니면 순위 기반
                if dist_range > 0.01:
                    # 거리 기반 점수 (0.5 ~ 1.0)
                    normalized = (dist - min_dist) / dist_range
                    score = 1.0 - (normalized * 0.5)
                else:
                    # 순위 기반 점수 (0.70 ~ 0.95) - 랜덤 요소 추가
                    import random
                    base_score = 0.95 - ((rank - 1) / max(total_tracks - 1, 1)) * 0.25
                    jitter = random.uniform(-0.02, 0.02)  # ±2% 랜덤
                    score = max(0.68, min(0.97, base_score + jitter))
                
                # 아티스트 매칭 보너스
                ems_artist = str(row.get('artists', '')).lower()
                if ems_artist in pms_artists:
                    score = min(0.99, score + 0.05)  # 아티스트 매칭시 +5%
                
                recommendations.append({
                    "track_id": str(row.get('track_id', '')),
                    "track_name": row.get('track_name', ''),
                    "artist": row.get('artists', ''),
                    "album_name": row.get('album_name', ''),
                    "genre": row.get('track_genre', 'unknown'),
                    "distance": dist,
                    "score": round(score, 4),
                    "recommendation_score": round(score, 4)
                })
            
            return {
                "success": True,
                "user_id": user_id,
                "model_used": os.path.basename(model_path),
                "count": len(recommendations),
                "recommendations": recommendations
            }
            
        except Exception as e:
            logger.error(f"추천 생성 오류: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "message": f"추천 생성 오류: {str(e)}",
                "recommendations": []
            }
    
    def analyze_and_save_gms(
        self,
        db,
        user_id: int,
        top_k: int = 50
    ) -> Dict:
        """분석 후 GMS 플레이리스트에 저장"""
        from sqlalchemy import text
        
        # 추천 생성
        result = self.get_recommendations(db, user_id, top_k)
        
        if not result.get("success"):
            return result
        
        recommendations = result.get("recommendations", [])
        
        if not recommendations:
            return {
                "success": False,
                "message": "추천할 트랙이 없습니다"
            }
        
        try:
            # GMS 플레이리스트 생성
            db.execute(text("""
                INSERT INTO playlists (user_id, title, description, space_type, status_flag, source_type)
                VALUES (:uid, 'AI 추천 리스트 (M3)', 'CatBoost 모델 기반 추천', 'GMS', 'PTP', 'System')
            """), {"uid": user_id})
            
            gms_id = db.execute(text("SELECT LAST_INSERT_ID()")).scalar()
            
            # 트랙 저장
            for rec in recommendations:
                metadata = json.dumps({
                    "kaggle_id": rec.get('track_id', ''),
                    "genre": rec.get('genre', 'unknown'),
                    "distance": rec.get('distance', 0),
                    "model": "M3"
                })
                
                db.execute(text("""
                    INSERT INTO tracks (title, artist, album, external_metadata)
                    VALUES (:title, :artist, :album, :meta)
                """), {
                    "title": rec['track_name'],
                    "artist": rec['artist'],
                    "album": rec['album_name'],
                    "meta": metadata
                })
                
                tid = db.execute(text("SELECT LAST_INSERT_ID()")).scalar()
                
                db.execute(text("""
                    INSERT INTO playlist_tracks (playlist_id, track_id)
                    VALUES (:pid, :tid)
                """), {"pid": gms_id, "tid": tid})
            
            db.commit()
            
            return {
                "success": True,
                "message": f"사용자 {user_id} 분석 및 GMS 생성 완료",
                "user_id": user_id,
                "playlist_id": gms_id,
                "model_used": result.get("model_used"),
                "count": len(recommendations)
            }
            
        except Exception as e:
            db.rollback()
            logger.error(f"GMS 저장 오류: {e}")
            return {
                "success": False,
                "message": f"GMS 저장 오류: {str(e)}"
            }


# 싱글톤 인스턴스
_m3_service: Optional[M3RecommendationService] = None

def get_m3_service() -> M3RecommendationService:
    """M3 서비스 싱글톤"""
    global _m3_service
    if _m3_service is None:
        _m3_service = M3RecommendationService()
    return _m3_service
