"""
M1 Service Layer
DB 연동 및 추천 로직 통합 서비스
"""
from .spotify_recommender import AudioFeaturePredictor, UserPreferenceProfile, HybridRecommender, PreferenceClassifier
from .search_enhancer import SearchBasedEnhancer, IntegratedRecommender
import pandas as pd
import numpy as np
import os
import json
import shutil
from sqlalchemy.orm import Session
from sqlalchemy import text
from datetime import datetime


class M1RecommendationService:
    """M1 추천 서비스 - DB 연동 및 ML 파이프라인 통합"""
    
    def __init__(self, model_path: str = None):
        self.predictor = AudioFeaturePredictor(model_type='Ridge')
        self.model_loaded = False
        
        if model_path and os.path.exists(model_path):
            try:
                self.predictor.load(model_path)
                self.model_loaded = True
                print(f"[M1] 모델 로드 완료: {model_path}")
            except Exception as e:
                print(f"[M1] 모델 로드 실패: {e}")
        
        self.enhancer = SearchBasedEnhancer()
        self.preference_classifier = PreferenceClassifier()
    
    def is_model_loaded(self) -> bool:
        return self.model_loaded
    
    # ==================== 1단계: 사용자 정보 조회 및 폴더 생성 ====================
    
    def get_user_info(self, db: Session, user_id: int) -> dict:
        """사용자 정보 조회"""
        query = text("""
            SELECT user_id, email, nickname, created_at
            FROM users
            WHERE user_id = :user_id
        """)
        result = db.execute(query, {"user_id": user_id}).fetchone()
        
        if not result:
            return None
        
        return {
            "user_id": result[0],
            "email": result[1],
            "nickname": result[2],
            "created_at": str(result[3]) if result[3] else None
        }
    
    def get_email_prefix(self, email: str) -> str:
        """이메일에서 @ 앞자리 추출"""
        if not email or '@' not in email:
            return "unknown"
        return email.split('@')[0]
    
    def create_user_model_folder(self, email: str) -> str:
        """사용자 이메일로 모델 폴더 생성"""
        email_prefix = self.get_email_prefix(email)
        base_path = os.path.dirname(__file__)
        user_folder = os.path.join(base_path, "user_models", email_prefix)
        
        if not os.path.exists(user_folder):
            os.makedirs(user_folder)
            print(f"[M1] 사용자 폴더 생성: {user_folder}")
        
        return user_folder
    
    # ==================== 2단계: 모델 파일 복사 ====================
    
    def copy_base_model_to_user(self, email: str) -> str:
        """기본 모델을 사용자 폴더로 복사"""
        email_prefix = self.get_email_prefix(email)
        base_path = os.path.dirname(__file__)
        
        # 원본 모델 경로
        source_model = os.path.join(base_path, "audio_predictor.pkl")
        
        # 사용자 폴더
        user_folder = self.create_user_model_folder(email)
        
        # 사용자 모델 경로 (이메일 앞자리 이름)
        user_model_path = os.path.join(user_folder, f"{email_prefix}.pkl")
        
        # 복사 (이미 있으면 스킵)
        if not os.path.exists(user_model_path):
            if os.path.exists(source_model):
                shutil.copy2(source_model, user_model_path)
                print(f"[M1] 모델 복사 완료: {user_model_path}")
            else:
                print(f"[M1] 원본 모델 없음: {source_model}")
                return None
        else:
            print(f"[M1] 사용자 모델 이미 존재: {user_model_path}")
        
        return user_model_path
    
    # ==================== 3-4단계: PMS 트랙으로 모델 추가학습 ====================
    
    def train_user_model(self, db: Session, user_id: int, email: str) -> dict:
        """
        사용자 PMS 트랙으로 모델 추가학습
        - PMS 트랙 가져오기
        - external_metadata에서 audio features 추출
        - 기본 모델 + PMS 데이터로 재학습
        - 파일명에 _ 붙여 저장
        """
        email_prefix = self.get_email_prefix(email)
        base_path = os.path.dirname(__file__)
        user_folder = os.path.join(base_path, "user_models", email_prefix)
        os.makedirs(user_folder, exist_ok=True)

        # 사용자 모델 경로
        user_model_path = os.path.join(user_folder, f"{email_prefix}.pkl")
        trained_model_path = os.path.join(user_folder, f"{email_prefix}_.pkl")

        # 1. PMS 트랙 가져오기
        pms_tracks = self.get_user_preferences_from_db(db, user_id)

        if pms_tracks.empty:
            return {
                "success": False,
                "message": "PMS 트랙이 없습니다.",
                "track_count": 0
            }

        print(f"[M1] PMS 트랙 {len(pms_tracks)}곡 로드 완료")

        # 2. external_metadata에서 audio features 추출
        audio_features = ['tempo', 'energy', 'danceability', 'valence', 'acousticness', 'instrumentalness']
        pms_with_audio = self._extract_audio_features_from_metadata(pms_tracks)

        # audio features가 있는 트랙만 필터링
        has_audio_features = pms_with_audio[audio_features].notna().any(axis=1)
        trainable_tracks = pms_with_audio[has_audio_features]

        print(f"[M1] Audio features 있는 트랙: {len(trainable_tracks)}/{len(pms_tracks)}곡")

        # 3. 새 모델 생성 및 학습
        user_predictor = AudioFeaturePredictor(model_type='Ridge')

        # 기본 모델 로드 (있으면)
        if os.path.exists(user_model_path):
            user_predictor.load(user_model_path)
            print(f"[M1] 기본 모델 로드: {user_model_path}")

        # 4. PMS 트랙으로 실제 학습 수행
        if len(trainable_tracks) >= 10:
            # audio features가 충분하면 실제 학습
            print(f"[M1] PMS 트랙으로 모델 학습 시작 ({len(trainable_tracks)}곡)")
            try:
                user_predictor.train(trainable_tracks, test_size=0.2)
                print(f"[M1] 모델 학습 완료!")
            except Exception as e:
                print(f"[M1] 학습 중 오류 (프로필 기반으로 대체): {e}")
        else:
            print(f"[M1] Audio features 트랙 부족 ({len(trainable_tracks)}곡), 프로필 기반 학습")

        # 5. 사용자 프로필 생성 (예측 포함)
        pms_with_features = user_predictor.predict(pms_tracks)
        user_profile = UserPreferenceProfile()
        user_profile.build_profile(pms_with_features)

        # 6. 학습된 모델 저장 (파일명에 _ 붙임)
        user_predictor.save(trained_model_path)
        print(f"[M1] 모델 저장 완료: {trained_model_path}")

        # 사용자 프로필도 저장
        profile_path = os.path.join(user_folder, f"{email_prefix}_profile.json")
        with open(profile_path, 'w', encoding='utf-8') as f:
            json.dump({
                "user_id": user_id,
                "email": email,
                "track_count": len(pms_tracks),
                "trainable_tracks": len(trainable_tracks),
                "feature_stats": user_profile.feature_stats,
                "genre_distribution": user_profile.genre_distribution,
                "trained_at": datetime.now().isoformat()
            }, f, ensure_ascii=False, indent=2)

        print(f"[M1] 모델 추가학습 완료: {trained_model_path}")

        return {
            "success": True,
            "message": f"모델 추가학습 완료 ({len(trainable_tracks)}/{len(pms_tracks)}곡 학습)",
            "track_count": len(pms_tracks),
            "trainable_tracks": len(trainable_tracks),
            "model_path": trained_model_path,
            "profile_path": profile_path
        }

    def _extract_audio_features_from_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """external_metadata에서 audio features 추출"""
        audio_features = ['tempo', 'energy', 'danceability', 'valence', 'acousticness', 'instrumentalness']
        result = df.copy()

        # audio features 컬럼 초기화
        for feat in audio_features:
            if feat not in result.columns:
                result[feat] = None

        # external_metadata에서 추출
        if 'external_metadata' in result.columns:
            for idx, row in result.iterrows():
                metadata = row.get('external_metadata')
                if metadata:
                    try:
                        if isinstance(metadata, str):
                            metadata = json.loads(metadata)

                        # metadata가 dict인지 확인
                        if not isinstance(metadata, dict):
                            continue

                        # audio_features 키에서 추출
                        audio_data = metadata.get('audio_features', metadata.get('audioFeatures', {}))
                        if audio_data and isinstance(audio_data, dict):
                            for feat in audio_features:
                                if feat in audio_data and audio_data[feat] is not None:
                                    result.at[idx, feat] = float(audio_data[feat])
                    except (json.JSONDecodeError, TypeError, ValueError):
                        pass

        return result
    
    # ==================== 6단계: 추가학습 모델로 EMS 트랙 평가 ====================
    
    def evaluate_with_user_model(self, db: Session, user_id: int, email: str, ems_tracks_df: pd.DataFrame) -> pd.DataFrame:
        """
        사용자 추가학습 모델로 EMS 트랙 평가
        - email_.pkl 모델 로드
        - EMS 트랙에 대해 점수 계산
        - 점수순 정렬하여 반환
        """
        email_prefix = self.get_email_prefix(email)
        base_path = os.path.dirname(__file__)
        
        # 추가학습 모델 경로
        trained_model_path = os.path.join(base_path, "user_models", email_prefix, f"{email_prefix}_.pkl")
        profile_path = os.path.join(base_path, "user_models", email_prefix, f"{email_prefix}_profile.json")
        
        # 추가학습 모델 확인
        if not os.path.exists(trained_model_path):
            print(f"[M1] 추가학습 모델 없음: {trained_model_path}")
            return pd.DataFrame()
        
        # 추가학습 모델 로드
        user_predictor = AudioFeaturePredictor(model_type='Ridge')
        user_predictor.load(trained_model_path)
        print(f"[M1] 추가학습 모델 로드: {trained_model_path}")
        
        # 사용자 프로필 로드
        user_profile = None
        if os.path.exists(profile_path):
            with open(profile_path, 'r', encoding='utf-8') as f:
                profile_data = json.load(f)
                user_profile = UserPreferenceProfile()
                user_profile.feature_stats = profile_data.get('feature_stats', {})
                user_profile.genre_distribution = profile_data.get('genre_distribution', {})
            print(f"[M1] 사용자 프로필 로드: {profile_path}")
        
        # EMS 트랙 특성 예측
        ems_with_features = user_predictor.predict(ems_tracks_df)
        
        # 점수 계산
        if user_profile and user_profile.feature_stats:
            # 사용자 프로필 기반 유사도 점수 계산
            scores = []
            for idx, row in ems_with_features.iterrows():
                score = self._calculate_similarity_score(row, user_profile)
                scores.append(score)
            ems_with_features['score'] = scores
        else:
            # 프로필 없으면 기본 점수 (0.5)
            ems_with_features['score'] = 0.5
        
        # 점수순 정렬
        result = ems_with_features.sort_values('score', ascending=False)
        
        print(f"[M1] EMS 트랙 {len(result)}곡 평가 완료")
        
        return result
    
    def _calculate_similarity_score(self, track_row, user_profile: UserPreferenceProfile) -> float:
        """트랙과 사용자 프로필 간 유사도 점수 계산"""
        total_score = 0.0
        total_weight = 0.0
        
        # 오디오 특성 유사도 (가중치 70%)
        audio_features = ['danceability', 'energy', 'valence', 'acousticness', 
                         'instrumentalness', 'speechiness', 'liveness']
        
        audio_score = 0.0
        audio_count = 0
        
        for feature in audio_features:
            predicted_col = f'predicted_{feature}'
            # 프로필에는 predicted_ prefix로 저장됨
            profile_key = predicted_col
            if predicted_col in track_row.index and profile_key in user_profile.feature_stats:
                track_val = float(track_row[predicted_col])
                user_mean = float(user_profile.feature_stats[profile_key].get('mean', 0.5))
                user_std = float(user_profile.feature_stats[profile_key].get('std', 0.2))
                
                # 차이 기반 점수 (0~1)
                if user_std > 0.01:
                    diff = abs(track_val - user_mean)
                    # 차이가 작을수록 높은 점수
                    feature_score = max(0, 1 - (diff / max(user_std * 2, 0.5)))
                else:
                    feature_score = max(0, 1 - abs(track_val - user_mean) * 2)
                
                audio_score += feature_score
                audio_count += 1
        
        if audio_count > 0:
            total_score += (audio_score / audio_count) * 0.7
            total_weight += 0.7
        
        # 장르 유사도 (가중치 30%)
        if 'track_genre' in track_row.index and user_profile.genre_distribution:
            track_genre = str(track_row['track_genre']).lower().strip()
            genre_score = float(user_profile.genre_distribution.get(track_genre, 0.0))
            total_score += genre_score * 0.3
            total_weight += 0.3
        
        # 최종 점수
        if total_weight > 0:
            final_score = total_score / total_weight
        else:
            final_score = 0.5  # 기본값
        
        return round(min(1.0, max(0.0, final_score)), 2)
    
    def _extract_metadata(self, external_metadata) -> dict:
        """external_metadata JSON 파싱"""
        if pd.isna(external_metadata) or not external_metadata:
            return {'genre': 'unknown', 'popularity': 50}
        try:
            if isinstance(external_metadata, str):
                return json.loads(external_metadata)
            return external_metadata
        except:
            return {'genre': 'unknown', 'popularity': 50}
    
    def _prepare_tracks_df(self, result) -> pd.DataFrame:
        """DB 결과를 ML 모델 형식으로 변환"""
        df = pd.DataFrame(result.fetchall(), columns=['track_id', 'title', 'artist', 'album', 'duration', 'external_metadata'])
        
        # 컬럼명 변환 (모델이 기대하는 형식)
        df = df.rename(columns={
            'title': 'track_name',
            'artist': 'artists',
            'album': 'album_name'
        })
        
        # duration: 초 → 밀리초
        df['duration_ms'] = df['duration'] * 1000
        
        # external_metadata에서 장르, popularity 추출
        metadata = df['external_metadata'].apply(self._extract_metadata)
        df['track_genre'] = metadata.apply(lambda x: x.get('genre', 'unknown'))
        df['popularity'] = metadata.apply(lambda x: x.get('popularity', 50))
        
        return df
    
    def get_user_preferences_from_db(self, db: Session, user_id: int) -> pd.DataFrame:
        """PMS에서 사용자 선호 트랙 조회"""
        query = text("""
            SELECT t.track_id, t.title, t.artist, t.album, t.duration, t.external_metadata
            FROM tracks t
            JOIN playlist_tracks pt ON t.track_id = pt.track_id
            JOIN playlists p ON pt.playlist_id = p.playlist_id
            WHERE p.user_id = :user_id AND p.space_type = 'PMS'
        """)
        result = db.execute(query, {"user_id": user_id})
        return self._prepare_tracks_df(result)
    
    def get_ems_tracks_from_db(self, db: Session, user_id: int = None, limit: int = None) -> pd.DataFrame:
        """
        EMS에서 후보 트랙 조회 (PMS/GMS에 이미 있는 곡 제외)

        Args:
            db: 데이터베이스 세션
            user_id: 사용자 ID (PMS/GMS 중복 제외용)
            limit: 최대 곡 수 (None이면 전체 조회)
        """
        if user_id and limit:
            # PMS와 GMS에 이미 있는 곡 제외 (아티스트+제목 기준 중복도 체크)
            query = text("""
                SELECT DISTINCT t.track_id, t.title, t.artist, t.album, t.duration, t.external_metadata
                FROM tracks t
                JOIN playlist_tracks pt ON t.track_id = pt.track_id
                JOIN playlists p ON pt.playlist_id = p.playlist_id
                WHERE p.space_type = 'EMS'
                AND t.track_id NOT IN (
                    SELECT t2.track_id FROM tracks t2
                    JOIN playlist_tracks pt2 ON t2.track_id = pt2.track_id
                    JOIN playlists p2 ON pt2.playlist_id = p2.playlist_id
                    WHERE p2.user_id = :user_id AND p2.space_type IN ('PMS', 'GMS')
                )
                AND CONCAT(LOWER(t.artist), '|', LOWER(t.title)) NOT IN (
                    SELECT CONCAT(LOWER(t3.artist), '|', LOWER(t3.title)) FROM tracks t3
                    JOIN playlist_tracks pt3 ON t3.track_id = pt3.track_id
                    JOIN playlists p3 ON pt3.playlist_id = p3.playlist_id
                    WHERE p3.user_id = :user_id AND p3.space_type IN ('PMS', 'GMS')
                )
                ORDER BY RAND()
                LIMIT :limit
            """)
            result = db.execute(query, {"user_id": user_id, "limit": limit})
            print(f"[M1] EMS 조회: user_id={user_id}, PMS/GMS 중복 제외")
        elif limit:
            query = text("""
                SELECT DISTINCT t.track_id, t.title, t.artist, t.album, t.duration, t.external_metadata
                FROM tracks t
                JOIN playlist_tracks pt ON t.track_id = pt.track_id
                JOIN playlists p ON pt.playlist_id = p.playlist_id
                WHERE p.space_type = 'EMS'
                ORDER BY RAND()
                LIMIT :limit
            """)
            result = db.execute(query, {"limit": limit})
        else:
            query = text("""
                SELECT DISTINCT t.track_id, t.title, t.artist, t.album, t.duration, t.external_metadata
                FROM tracks t
                JOIN playlist_tracks pt ON t.track_id = pt.track_id
                JOIN playlists p ON pt.playlist_id = p.playlist_id
                WHERE p.space_type = 'EMS'
            """)
            result = db.execute(query)
        return self._prepare_tracks_df(result)
    
    def get_random_ems_tracks(self, db: Session, limit: int = 100, user_id: int = None) -> pd.DataFrame:
        """
        EMS 전체에서 랜덤하게 트랙 추출 (중복 제거, PMS/GMS 제외)
        - SQL RAND()로 DB에서 직접 랜덤 추출
        - 아티스트별 편중 없이 균등 분포
        - user_id가 있으면 해당 사용자의 PMS/GMS 곡 제외
        """
        if user_id:
            query = text("""
                SELECT DISTINCT t.track_id, t.title, t.artist, t.album, t.duration, t.external_metadata
                FROM tracks t
                JOIN playlist_tracks pt ON t.track_id = pt.track_id
                JOIN playlists p ON pt.playlist_id = p.playlist_id
                WHERE p.space_type = 'EMS'
                AND t.track_id NOT IN (
                    SELECT t2.track_id FROM tracks t2
                    JOIN playlist_tracks pt2 ON t2.track_id = pt2.track_id
                    JOIN playlists p2 ON pt2.playlist_id = p2.playlist_id
                    WHERE p2.user_id = :user_id AND p2.space_type IN ('PMS', 'GMS')
                )
                AND CONCAT(LOWER(t.artist), '|', LOWER(t.title)) NOT IN (
                    SELECT CONCAT(LOWER(t3.artist), '|', LOWER(t3.title)) FROM tracks t3
                    JOIN playlist_tracks pt3 ON t3.track_id = pt3.track_id
                    JOIN playlists p3 ON pt3.playlist_id = p3.playlist_id
                    WHERE p3.user_id = :user_id AND p3.space_type IN ('PMS', 'GMS')
                )
                ORDER BY RAND()
                LIMIT :limit
            """)
            result = db.execute(query, {"user_id": user_id, "limit": limit})
            print(f"[M1] EMS 랜덤 추출: user_id={user_id}, PMS/GMS 중복 제외")
        else:
            query = text("""
                SELECT DISTINCT t.track_id, t.title, t.artist, t.album, t.duration, t.external_metadata
                FROM tracks t
                JOIN playlist_tracks pt ON t.track_id = pt.track_id
                JOIN playlists p ON pt.playlist_id = p.playlist_id
                WHERE p.space_type = 'EMS'
                ORDER BY RAND()
                LIMIT :limit
            """)
            result = db.execute(query, {"limit": limit})
        df = self._prepare_tracks_df(result)
        print(f"[M1] EMS 랜덤 {len(df)}곡 추출 완료")
        return df
    
    def get_tracks_by_ids(self, db: Session, track_ids: list) -> pd.DataFrame:
        """특정 track_id 목록으로 트랙 조회"""
        if not track_ids:
            return pd.DataFrame()
        
        # IN 절을 위한 플레이스홀더 생성
        placeholders = ','.join([f':id_{i}' for i in range(len(track_ids))])
        query = text(f"""
            SELECT track_id, title, artist, album, duration, external_metadata
            FROM tracks
            WHERE track_id IN ({placeholders})
        """)
        
        # 파라미터 딕셔너리 생성
        params = {f'id_{i}': tid for i, tid in enumerate(track_ids)}
        result = db.execute(query, params)
        return self._prepare_tracks_df(result)
    
    def get_recommendations(self, db: Session, user_id: int, ems_limit: int = 100, track_ids: list = None) -> pd.DataFrame:
        """
        M1 추천 파이프라인 실행

        1. PMS에서 사용자 프로필 생성 (없으면 전체 EMS 기반 프로필)
        2. EMS에서 후보 트랙 추출 (ems_limit 개) 또는 track_ids로 특정 트랙만
        3. IntegratedRecommender로 4-Factor 점수 계산
        4. final_score >= 0.5 필터링 (PMS 없으면 0.5, 있으면 0.7)

        Args:
            db: 데이터베이스 세션
            user_id: 사용자 ID
            ems_limit: EMS에서 분석할 곡 수 (기본값: 100)
            track_ids: 특정 트랙 ID 목록 (데모 페이지에서 동일 트랙 비교용)
        """
        has_pms = True
        
        # 1. 사용자 프로필 생성 (PMS 기반)
        pms_df = self.get_user_preferences_from_db(db, user_id)
        if pms_df.empty:
            print(f"[M1] 사용자 {user_id}의 PMS 데이터 없음 - 전체 EMS 기반 프로필 생성")
            has_pms = False
            # PMS 없으면 전체 EMS에서 랜덤 샘플링하여 프로필 생성
            pms_df = self.get_random_ems_tracks(db, limit=50)
            if pms_df.empty:
                print(f"[M1] EMS 데이터도 없음")
                return pd.DataFrame()
        
        # 오디오 특성 예측 및 프로필 생성
        pms_enhanced = self.predictor.predict(pms_df)
        user_profile = UserPreferenceProfile()
        user_profile.build_profile(pms_enhanced)
        
        # 2. EMS 후보 트랙 조회 (track_ids가 제공되면 해당 트랙만, 아니면 ems_limit 적용)
        if track_ids and len(track_ids) > 0:
            # 데모 페이지에서 특정 트랙 ID로 조회 (동일 트랙 비교용)
            ems_df = self.get_tracks_by_ids(db, track_ids)
            print(f"[M1] 특정 트랙 ID로 조회: {len(ems_df)}곡 (요청: {len(track_ids)}개)")
        else:
            ems_df = self.get_ems_tracks_from_db(db, user_id, limit=ems_limit)
            if ems_df.empty:
                # 사용자 EMS 없으면 전체 EMS에서 추출
                print(f"[M1] 사용자 {user_id}의 EMS 데이터 없음 - 전체 EMS 사용")
                ems_df = self.get_random_ems_tracks(db, limit=ems_limit)
            print(f"[M1] EMS에서 {len(ems_df)}곡 분석 (설정: {ems_limit}곡)")

        if ems_df.empty:
            print(f"[M1] EMS 데이터 없음")
            return pd.DataFrame()
        
        # 3. 통합 추천 파이프라인 실행
        recommender = IntegratedRecommender(
            self.predictor, 
            user_profile, 
            self.enhancer,
            self.preference_classifier if self.preference_classifier.is_trained else None
        )
        recommendations = recommender.recommend_with_search(ems_df)
        
        # 4. GMS 품질 필터링
        threshold = 0.7
        gms_pass = recommendations[recommendations['final_score'] >= threshold].sort_values(
            by='final_score', ascending=False
        )
        
        # 최소 20곡 보장 (threshold 미달이면 top 20 강제 포함)
        if len(gms_pass) < 20:
            print(f"[M1] threshold 미달 - top 20으로 보완")
            gms_pass = recommendations.sort_values(by='final_score', ascending=False).head(20)

        gms_pass['recommendation_score'] = gms_pass['final_score']

        # 중복 제거 (track_id 기준)
        before_dedup = len(gms_pass)
        gms_pass = gms_pass.drop_duplicates(subset=['track_id'], keep='first')
        
        # 아티스트+제목 기준 중복 제거 (같은 곡이 다른 track_id로 존재할 수 있음)
        # 컬럼명: artist 또는 artists (DB 조회 후 rename됨)
        artist_col = 'artists' if 'artists' in gms_pass.columns else 'artist'
        gms_pass['artist_title_key'] = gms_pass[artist_col].str.lower() + '|' + gms_pass['track_name'].str.lower()
        gms_pass = gms_pass.drop_duplicates(subset=['artist_title_key'], keep='first')
        gms_pass = gms_pass.drop(columns=['artist_title_key'])
        
        after_dedup = len(gms_pass)
        if before_dedup != after_dedup:
            print(f"[M1] 중복 제거: {before_dedup}곡 → {after_dedup}곡")

        print(f"[M1] 사용자 {user_id}: {len(ems_df)}개 후보 중 {len(gms_pass)}개 GMS 통과 (threshold: {threshold})")

        # NaN/Infinity 처리 (JSON Serialization Error 방지)
        gms_pass = gms_pass.fillna(0)
        gms_pass = gms_pass.replace([np.inf, -np.inf], 0)

        return gms_pass

    def save_gms_playlist(self, db: Session, user_id: int, recommendations_df: pd.DataFrame) -> int:
        """추천 결과를 GMS 플레이리스트로 DB에 저장"""
        try:
            # 기존 GMS 플레이리스트는 유지 (사용자가 직접 승인/거절로 관리)
            
            # 1. GMS 플레이리스트 생성
            insert_playlist = text("""
                INSERT INTO playlists (user_id, title, description, space_type, status_flag, source_type)
                VALUES (:user_id, :title, :description, 'GMS', 'PFP', 'System')
            """)
            result = db.execute(insert_playlist, {
                "user_id": user_id,
                "title": f"AI Gateway (GMS) - {datetime.now().strftime('%m/%d %H:%M')}",
                "description": "AI 추천 플레이리스트 (M1 모델)"
            })
            playlist_id = result.lastrowid
            
            # 2. 트랙 연결 및 점수 저장
            order_idx = 0
            for _, row in recommendations_df.iterrows():
                track_id = int(row['track_id'])
                score = float(row['recommendation_score']) * 100  # 0-1 → 0-100
                
                # playlist_tracks 연결
                db.execute(text("""
                    INSERT IGNORE INTO playlist_tracks (playlist_id, track_id, order_index)
                    VALUES (:playlist_id, :track_id, :order_index)
                """), {"playlist_id": playlist_id, "track_id": track_id, "order_index": order_idx})
                order_idx += 1
                
                # AI 점수 기록
                db.execute(text("""
                    INSERT INTO track_scored_id (track_id, user_id, ai_score) 
                    VALUES (:track_id, :user_id, :score)
                    ON DUPLICATE KEY UPDATE ai_score = :score
                """), {"track_id": track_id, "user_id": user_id, "score": score})
            
            db.commit()
            print(f"[M1] GMS 플레이리스트 생성: ID={playlist_id}, 트랙={order_idx}개")
            
            return playlist_id
        except Exception as e:
            db.rollback()
            print(f"[M1] GMS 저장 실패: {e}")
            raise

    def delete_tracks_from_playlist(self, db: Session, playlist_id: int, track_ids: list) -> dict:
        """플레이리스트에서 트랙 삭제"""
        if not track_ids:
            return {"status": "success", "deleted_count": 0}
        
        placeholders = ','.join([f':tid_{i}' for i in range(len(track_ids))])
        params = {'playlist_id': playlist_id}
        for i, tid in enumerate(track_ids):
            params[f'tid_{i}'] = tid
        
        delete_query = text(f"""
            DELETE FROM playlist_tracks 
            WHERE playlist_id = :playlist_id AND track_id IN ({placeholders})
        """)
        db.execute(delete_query, params)
        db.commit()
        
        return {"status": "success", "deleted_count": len(track_ids)}
    
    def retrain_with_feedback(self, db: Session, user_id: int, deleted_track_ids: list) -> dict:
        """
        피드백 기반 모델 재학습
        
        - 좋아요: PMS 트랙 (기존)
        - 싫어요: 삭제된 트랙
        """
        if not deleted_track_ids:
            return {"status": "skipped", "message": "삭제된 트랙 없음"}
        
        # 1. 좋아요 트랙 (PMS)
        likes_df = self.get_user_preferences_from_db(db, user_id)
        if likes_df.empty:
            return {"status": "skipped", "message": "PMS 데이터 없음"}
        
        likes_enhanced = self.predictor.predict(likes_df)
        
        # 2. 싫어요 트랙 (삭제된 트랙)
        query = text("""
            SELECT track_id, title, artist, album, duration, external_metadata 
            FROM tracks WHERE track_id IN :ids
        """)
        result = db.execute(query, {"ids": tuple(deleted_track_ids)})
        dislikes_df = self._prepare_tracks_df(result)
        
        if dislikes_df.empty:
            return {"status": "skipped", "message": "삭제된 트랙 정보 없음"}
        
        dislikes_enhanced = self.predictor.predict(dislikes_df)
        
        # 3. PreferenceClassifier 재학습
        audio_cols = [f"predicted_{c}" for c in self.predictor.audio_features]
        self.preference_classifier.train(likes_enhanced, dislikes_enhanced, audio_cols, validate=True)
        
        return {
            "status": "success",
            "message": f"{len(deleted_track_ids)}개 싫어요 데이터로 모델 재학습 완료",
            "metrics": self.preference_classifier.metrics
        }
    
    def get_user_profile(self, db: Session, user_id: int) -> dict:
        """사용자 음악 취향 프로필 조회"""
        pms_df = self.get_user_preferences_from_db(db, user_id)
        
        if pms_df.empty:
            return {"status": "no_data", "message": "PMS 데이터 없음"}
        
        pms_enhanced = self.predictor.predict(pms_df)
        user_profile = UserPreferenceProfile()
        user_profile.build_profile(pms_enhanced)
        
        return user_profile.get_summary()
