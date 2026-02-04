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
    
    def get_ems_tracks_from_db(self, db: Session, user_id: int) -> pd.DataFrame:
        """EMS에서 후보 트랙 조회"""
        query = text("""
            SELECT t.track_id, t.title, t.artist, t.album, t.duration, t.external_metadata
            FROM tracks t
            JOIN playlist_tracks pt ON t.track_id = pt.track_id
            JOIN playlists p ON pt.playlist_id = p.playlist_id
            WHERE p.user_id = :user_id AND p.space_type = 'EMS'
        """)
        result = db.execute(query, {"user_id": user_id})
        return self._prepare_tracks_df(result)
    
    def get_recommendations(self, db: Session, user_id: int) -> pd.DataFrame:
        """
        M1 추천 파이프라인 실행
        
        1. PMS에서 사용자 프로필 생성
        2. EMS에서 후보 트랙 추출
        3. IntegratedRecommender로 4-Factor 점수 계산
        4. final_score >= 0.7 필터링
        """
        # 1. 사용자 프로필 생성 (PMS 기반)
        pms_df = self.get_user_preferences_from_db(db, user_id)
        if pms_df.empty:
            print(f"[M1] 사용자 {user_id}의 PMS 데이터 없음")
            return pd.DataFrame()
        
        # 오디오 특성 예측 및 프로필 생성
        pms_enhanced = self.predictor.predict(pms_df)
        user_profile = UserPreferenceProfile()
        user_profile.build_profile(pms_enhanced)
        
        # 2. EMS 후보 트랙 조회
        ems_df = self.get_ems_tracks_from_db(db, user_id)
        if ems_df.empty:
            print(f"[M1] 사용자 {user_id}의 EMS 데이터 없음")
            return pd.DataFrame()
        
        # 3. 통합 추천 파이프라인 실행
        recommender = IntegratedRecommender(
            self.predictor, 
            user_profile, 
            self.enhancer,
            self.preference_classifier if self.preference_classifier.is_trained else None
        )
        recommendations = recommender.recommend_with_search(ems_df)
        
        # 4. GMS 품질 필터링 (0.7 이상)
        gms_pass = recommendations[recommendations['final_score'] >= 0.7].sort_values(
            by='final_score', ascending=False
        )
        gms_pass['recommendation_score'] = gms_pass['final_score']
        
        print(f"[M1] 사용자 {user_id}: {len(ems_df)}개 후보 중 {len(gms_pass)}개 GMS 통과")
        
        return gms_pass
    
    def save_gms_playlist(self, db: Session, user_id: int, recommendations_df: pd.DataFrame) -> int:
        """추천 결과를 GMS 플레이리스트로 DB에 저장"""
        
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
    
    def delete_tracks_from_playlist(self, db: Session, playlist_id: int, track_ids: list) -> dict:
        """플레이리스트에서 트랙 삭제"""
        if not track_ids:
            return {"status": "success", "deleted_count": 0}
        
        delete_query = text("""
            DELETE FROM playlist_tracks 
            WHERE playlist_id = :playlist_id AND track_id IN :track_ids
        """)
        db.execute(delete_query, {
            "playlist_id": playlist_id,
            "track_ids": tuple(track_ids)
        })
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
