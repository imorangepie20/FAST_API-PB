from .spotify_recommender import AudioFeaturePredictor, UserPreferenceProfile, HybridRecommender, PreferenceClassifier
from .search_enhancer import SearchBasedEnhancer, IntegratedRecommender
import pandas as pd
import numpy as np
import os
import pickle
from sqlalchemy.orm import Session
from sqlalchemy import text

class SpotifyRecommendationService:
    def __init__(self, model_path: str):
        self.predictor = AudioFeaturePredictor(model_type='Ridge')
        if os.path.exists(model_path):
            self.predictor.load(model_path)
        self.enhancer = SearchBasedEnhancer()
        self.preference_classifier = PreferenceClassifier()

    def get_tracks_from_db(self, db: Session):
        query = text("SELECT track_id, title, artist, album, duration, external_metadata FROM tracks")
        result = db.execute(query)
        df = pd.DataFrame(result.fetchall(), columns=['track_id', 'title', 'artist', 'album', 'duration', 'external_metadata'])

        # 컬럼명 변환 (모델이 기대하는 형식으로)
        df = df.rename(columns={
            'title': 'track_name',
            'artist': 'artists',
            'album': 'album_name'
        })
        # duration은 초 단위 → 밀리초로 변환
        df['duration_ms'] = df['duration'] * 1000

        # external_metadata에서 장르 추출
        import json
        df['track_genre'] = df['external_metadata'].apply(
            lambda x: json.loads(x).get('genre', 'unknown') if pd.notna(x) else 'unknown'
        )
        df['popularity'] = df['external_metadata'].apply(
            lambda x: json.loads(x).get('popularity', 50) if pd.notna(x) else 50
        )

        return df

    def get_user_preferences_from_db(self, db: Session, user_id: int):
        # Get user's preference tracks from PMS (Personal Music Space)
        query = text("""
            SELECT t.track_id, t.title, t.artist, t.album, t.duration, t.external_metadata
            FROM tracks t
            JOIN playlist_tracks pt ON t.track_id = pt.track_id
            JOIN playlists p ON pt.playlist_id = p.playlist_id
            WHERE p.user_id = :user_id AND p.space_type = 'PMS'
        """)
        result = db.execute(query, {"user_id": user_id})
        df = pd.DataFrame(result.fetchall(), columns=['track_id', 'title', 'artist', 'album', 'duration', 'external_metadata'])

        # 컬럼명 변환
        df = df.rename(columns={
            'title': 'track_name',
            'artist': 'artists',
            'album': 'album_name'
        })
        df['duration_ms'] = df['duration'] * 1000

        # external_metadata에서 장르 추출
        import json
        df['track_genre'] = df['external_metadata'].apply(
            lambda x: json.loads(x).get('genre', 'unknown') if pd.notna(x) else 'unknown'
        )
        df['popularity'] = df['external_metadata'].apply(
            lambda x: json.loads(x).get('popularity', 50) if pd.notna(x) else 50
        )

        return df

    def get_recommendations(self, db: Session, user_id: int):
        # Step 3: Use the real ML model to recommend from EMS
        
        # 1. Get User Profile (from PMS tracks)
        pms_df = self.get_user_preferences_from_db(db, user_id)
        if pms_df.empty:
            return pd.DataFrame() # No preferences found
            
        # Predict audio features for PMS to build DNA profile
        pms_enhanced = self.predictor.predict(pms_df)
        user_profile = UserPreferenceProfile()
        user_profile.build_profile(pms_enhanced)
        
        # 2. Get EMS Candidate Tracks
        query = text("""
            SELECT t.track_id, t.title, t.artist, t.album, t.duration, t.external_metadata
            FROM tracks t
            JOIN playlist_tracks pt ON t.track_id = pt.track_id
            JOIN playlists p ON pt.playlist_id = p.playlist_id
            WHERE p.user_id = :user_id AND p.space_type = 'EMS'
        """)
        result = db.execute(query, {"user_id": user_id})
        ems_df = pd.DataFrame(result.fetchall(), columns=['track_id', 'title', 'artist', 'album', 'duration', 'external_metadata'])

        # 컬럼명 변환
        ems_df = ems_df.rename(columns={
            'title': 'track_name',
            'artist': 'artists',
            'album': 'album_name'
        })
        ems_df['duration_ms'] = ems_df['duration'] * 1000

        # external_metadata에서 장르 추출
        import json
        ems_df['track_genre'] = ems_df['external_metadata'].apply(
            lambda x: json.loads(x).get('genre', 'unknown') if pd.notna(x) else 'unknown'
        )
        ems_df['popularity'] = ems_df['external_metadata'].apply(
            lambda x: json.loads(x).get('popularity', 50) if pd.notna(x) else 50
        )
        
        if ems_df.empty:
            return pd.DataFrame()
            
        # 3. Predict & Score (Integrated Pipeline matching the 4-factor report)
        ems_enhanced = self.predictor.predict(ems_df)
        recommender = IntegratedRecommender(self.predictor, user_profile, self.enhancer)
        recommendations = recommender.recommend_with_search(ems_df)
        
        # 4. Filter by GMS Quality Threshold (0.7) on 'final_score'
        gms_pass = recommendations[recommendations['final_score'] >= 0.7].sort_values(by='final_score', ascending=False)
        
        # Rename column for consistency with previous use if needed, but 'final_score' is what user wants
        gms_pass['recommendation_score'] = gms_pass['final_score']
        
        return gms_pass

    def save_gms_playlist(self, db: Session, user_id: int, recommendations_df: pd.DataFrame):
        """Save recommendations as GMS playlist in DB"""
        from datetime import datetime
        
        # 1. Create GMS playlist
        insert_playlist = text("""
            INSERT INTO playlists (user_id, title, description, space_type, status_flag, source_type)
            VALUES (:user_id, :title, :description, 'GMS', 'PFP', 'System')
        """)
        result = db.execute(insert_playlist, {
            "user_id": user_id,
            "title": f"Verified Gateway (GMS) - {datetime.now().strftime('%m/%d %H:%M')}",
            "description": "AI 추천 플레이리스트"
        })
        playlist_id = result.lastrowid
        
        # 2. Insert tracks into playlist_tracks & track_scored_id
        order_idx = 0
        for _, row in recommendations_df.iterrows():
            track_id = int(row['track_id'])
            score = float(row['recommendation_score'])

            # GMS Playlist link
            db.execute(text("""
                INSERT IGNORE INTO playlist_tracks (playlist_id, track_id, order_index)
                VALUES (:playlist_id, :track_id, :order_index)
            """), {"playlist_id": playlist_id, "track_id": track_id, "order_index": order_idx})
            order_idx += 1
            
            # Record final AI Score
            db.execute(text("""
                INSERT INTO track_scored_id (track_id, user_id, ai_score) 
                VALUES (:track_id, :user_id, :score)
                ON DUPLICATE KEY UPDATE ai_score = :score
            """), {"track_id": track_id, "user_id": user_id, "score": score})
        
        db.commit()
        return playlist_id

    def delete_tracks_from_playlist(self, db: Session, playlist_id: int, track_ids: list):
        """Delete specific tracks from a playlist"""
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

    def retrain_with_feedback(self, db: Session, user_id: int, deleted_track_ids: list):
        # Step 5: Retrain based on deleted (disliked) tracks
        # 1. Get liked tracks
        likes_df = self.get_user_preferences_from_db(db, user_id)
        likes_enhanced = self.predictor.predict(likes_df)
        
        # 2. Get deleted tracks
        query = text("SELECT track_id, title, artist, album, duration, external_metadata FROM tracks WHERE track_id IN :ids")
        result = db.execute(query, {"ids": tuple(deleted_track_ids)})
        dislikes_df = pd.DataFrame(result.fetchall(), columns=['track_id', 'title', 'artist', 'album', 'duration', 'external_metadata'])

        # 컬럼명 변환
        dislikes_df = dislikes_df.rename(columns={
            'title': 'track_name',
            'artist': 'artists',
            'album': 'album_name'
        })
        dislikes_df['duration_ms'] = dislikes_df['duration'] * 1000

        # external_metadata에서 장르 추출
        import json
        dislikes_df['track_genre'] = dislikes_df['external_metadata'].apply(
            lambda x: json.loads(x).get('genre', 'unknown') if pd.notna(x) else 'unknown'
        )
        dislikes_df['popularity'] = dislikes_df['external_metadata'].apply(
            lambda x: json.loads(x).get('popularity', 50) if pd.notna(x) else 50
        )
        
        dislikes_enhanced = self.predictor.predict(dislikes_df)
        
        # 3. Train preference classifier with validation
        audio_cols = [f"predicted_{c}" for c in self.predictor.audio_features]
        self.preference_classifier.train(likes_enhanced, dislikes_enhanced, audio_cols, validate=True)
        
        return {
            "status": "success", 
            "message": f"Model retrained with {len(deleted_track_ids)} dislikes.",
            "metrics": self.preference_classifier.metrics
        }
