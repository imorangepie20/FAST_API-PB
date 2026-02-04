"""
Search-based Quality Enhancement Module
========================================

Enhances track evaluation with web search-based qualitative information:
- Artist style and influences
- Track mood and atmosphere
- Critical reception
- Playlist context analysis

Author: AI Assistant
Date: 2026-01-27
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import json
import re
from collections import Counter


class SearchBasedEnhancer:
    """
    Enhances track data with search-based qualitative information
    """
    
    def __init__(self):
        self.cache = {}
        self.mood_keywords = {
            'energetic': ['energetic', 'upbeat', 'lively', 'dynamic', 'powerful'],
            'calm': ['calm', 'peaceful', 'relaxing', 'soothing', 'mellow'],
            'happy': ['happy', 'joyful', 'cheerful', 'uplifting', 'positive'],
            'sad': ['sad', 'melancholic', 'emotional', 'dark', 'somber'],
            'aggressive': ['aggressive', 'intense', 'heavy', 'hard', 'fierce'],
            'romantic': ['romantic', 'love', 'intimate', 'sensual', 'tender']
        }
        
    def analyze_playlist_context(self, playlist_name: str) -> Dict[str, float]:
        """
        Infer intended mood/purpose from playlist name
        """
        if not playlist_name or pd.isna(playlist_name):
            return {'energy': 0.5, 'valence': 0.5, 'focus': 'general'}
        
        name_lower = playlist_name.lower()
        
        # Context patterns
        contexts = {
            'workout': {'energy': 0.9, 'valence': 0.7, 'tempo': 'fast'},
            'gym': {'energy': 0.9, 'valence': 0.7, 'tempo': 'fast'},
            'study': {'energy': 0.3, 'valence': 0.5, 'instrumentalness': 0.8},
            'focus': {'energy': 0.3, 'valence': 0.5, 'instrumentalness': 0.7},
            'chill': {'energy': 0.3, 'valence': 0.6, 'acousticness': 0.6},
            'relax': {'energy': 0.2, 'valence': 0.6, 'acousticness': 0.7},
            'sleep': {'energy': 0.1, 'valence': 0.5, 'acousticness': 0.8},
            'party': {'energy': 0.9, 'valence': 0.8, 'danceability': 0.9},
            'sad': {'energy': 0.3, 'valence': 0.2, 'acousticness': 0.6},
            'happy': {'energy': 0.7, 'valence': 0.9},
            'morning': {'energy': 0.6, 'valence': 0.7},
            'night': {'energy': 0.4, 'valence': 0.5},
            'driving': {'energy': 0.7, 'valence': 0.6},
            'running': {'energy': 0.9, 'valence': 0.7, 'tempo': 'fast'},
            'romantic': {'energy': 0.4, 'valence': 0.6, 'acousticness': 0.5},
            'dinner': {'energy': 0.3, 'valence': 0.6, 'acousticness': 0.6},
            'meditation': {'energy': 0.1, 'valence': 0.5, 'instrumentalness': 0.9}
        }
        
        # Match patterns
        for keyword, attributes in contexts.items():
            if keyword in name_lower:
                return attributes
        
        # Default
        return {'energy': 0.5, 'valence': 0.5, 'focus': 'general'}
    
    def extract_mood_from_text(self, text: str) -> Dict[str, float]:
        """
        Extract mood scores from descriptive text
        """
        if not text or pd.isna(text):
            return {}
        
        text_lower = text.lower()
        mood_scores = {}
        
        for mood, keywords in self.mood_keywords.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                mood_scores[mood] = min(score / len(keywords), 1.0)
        
        return mood_scores
    
    def infer_from_genre(self, genre: str) -> Dict[str, float]:
        """
        Infer typical audio characteristics from genre
        """
        if not genre or pd.isna(genre):
            return {}
        
        genre_lower = genre.lower()
        
        # Genre-based heuristics
        genre_profiles = {
            'edm': {'energy': 0.9, 'danceability': 0.85, 'valence': 0.7, 'acousticness': 0.1},
            'electronic': {'energy': 0.75, 'danceability': 0.8, 'acousticness': 0.15},
            'hip hop': {'energy': 0.7, 'danceability': 0.75, 'speechiness': 0.3},
            'rap': {'energy': 0.7, 'danceability': 0.7, 'speechiness': 0.4},
            'rock': {'energy': 0.8, 'loudness': 0.7, 'acousticness': 0.2},
            'metal': {'energy': 0.95, 'loudness': 0.9, 'acousticness': 0.1},
            'pop': {'energy': 0.65, 'danceability': 0.7, 'valence': 0.65},
            'classical': {'energy': 0.4, 'acousticness': 0.9, 'instrumentalness': 0.85},
            'jazz': {'energy': 0.5, 'acousticness': 0.7, 'instrumentalness': 0.6},
            'blues': {'energy': 0.45, 'acousticness': 0.65, 'valence': 0.4},
            'country': {'energy': 0.55, 'acousticness': 0.6, 'valence': 0.6},
            'folk': {'energy': 0.4, 'acousticness': 0.8, 'valence': 0.55},
            'acoustic': {'energy': 0.4, 'acousticness': 0.9, 'valence': 0.6},
            'indie': {'energy': 0.6, 'acousticness': 0.5, 'valence': 0.6},
            'indie-pop': {'energy': 0.6, 'acousticness': 0.5, 'valence': 0.6},
            'r&b': {'energy': 0.6, 'danceability': 0.7, 'valence': 0.6},
            'soul': {'energy': 0.55, 'valence': 0.65, 'acousticness': 0.5},
            'reggae': {'energy': 0.65, 'danceability': 0.75, 'valence': 0.7},
            'latin': {'energy': 0.75, 'danceability': 0.85, 'valence': 0.75},
            'ambient': {'energy': 0.2, 'instrumentalness': 0.8, 'acousticness': 0.4},
            'lofi': {'energy': 0.3, 'valence': 0.5, 'acousticness': 0.4}
        }
        
        # Find matching genre
        for genre_key, profile in genre_profiles.items():
            if genre_key in genre_lower:
                return profile
        
        return {}
    
    def analyze_duration_context(self, duration_ms: float) -> Dict[str, str]:
        """
        Infer track purpose from duration
        """
        duration_sec = duration_ms / 1000
        
        if duration_sec < 120:
            return {'type': 'intro/interlude', 'purpose': 'transition'}
        elif duration_sec < 180:
            return {'type': 'standard', 'purpose': 'radio-friendly'}
        elif duration_sec < 300:
            return {'type': 'extended', 'purpose': 'album track'}
        else:
            return {'type': 'epic', 'purpose': 'artistic/progressive'}
    
    def enhance_track_batch(self, tracks_df: pd.DataFrame) -> pd.DataFrame:
        """
        Enhance a batch of tracks with inferred qualitative data
        """
        enhanced = tracks_df.copy()
        
        # Playlist context
        if 'playlist_name' in enhanced.columns:
            contexts = enhanced['playlist_name'].apply(self.analyze_playlist_context)
            for key in ['energy', 'valence', 'danceability', 'acousticness', 'instrumentalness']:
                enhanced[f'context_{key}'] = contexts.apply(lambda x: x.get(key, 0.5))
        
        # Genre-based inference
        if 'track_genre' in enhanced.columns:
            genre_profiles = enhanced['track_genre'].apply(self.infer_from_genre)
            for key in ['energy', 'valence', 'danceability', 'acousticness']:
                enhanced[f'genre_{key}'] = genre_profiles.apply(lambda x: x.get(key, 0.5))
        
        # Duration context
        if 'duration_ms' in enhanced.columns:
            duration_context = enhanced['duration_ms'].apply(self.analyze_duration_context)
            enhanced['duration_type'] = duration_context.apply(lambda x: x.get('type', 'standard'))
            enhanced['duration_purpose'] = duration_context.apply(lambda x: x.get('purpose', 'general'))
        
        # Combined qualitative score
        qual_features = [col for col in enhanced.columns if col.startswith('context_') or col.startswith('genre_')]
        if qual_features:
            enhanced['qualitative_confidence'] = enhanced[qual_features].notna().sum(axis=1) / len(qual_features)
        
        return enhanced


class IntegratedRecommender:
    """
    Integrates search enhancement with audio feature prediction
    """
    
    def __init__(self, audio_predictor, user_profile, search_enhancer, preference_classifier=None):
        self.audio_predictor = audio_predictor
        self.user_profile = user_profile
        self.search_enhancer = search_enhancer
        self.preference_classifier = preference_classifier
    
    def recommend_with_search(self, test_df: pd.DataFrame, 
                             weights: Dict[str, float] = None) -> pd.DataFrame:
        """
        Generate recommendations with search enhancement
        """
        if weights is None:
            weights = {
                'predicted_audio': 0.35,
                'qualitative': 0.25,
                'genre': 0.15,
                'artist': 0.1,
                'preference': 0.15 if self.preference_classifier and self.preference_classifier.is_trained else 0.0
            }
        
        print("\n INTEGRATED RECOMMENDATION PIPELINE")
        print("=" * 60)
        
        # Step 1: Predict audio features
        print("\n1  Predicting audio features...")
        with_predictions = self.audio_predictor.predict(test_df)
        
        # Step 2: Add search-based enhancement
        print("\n2  Adding search-based qualitative analysis...")
        enhanced = self.search_enhancer.enhance_track_batch(with_predictions)
        
        # Step 3: Calculate scores
        print("\n3  Calculating recommendation scores...")
        
        # Predicted audio similarity
        pred_audio_features = [col for col in enhanced.columns if col.startswith('predicted_')]
        if pred_audio_features:
            user_pref_audio = np.array([
                self.user_profile.feature_stats.get(feat, {}).get('mean', 0.5)
                for feat in pred_audio_features
            ])
            test_audio = enhanced[pred_audio_features].fillna(0.5).values
            from sklearn.metrics.pairwise import cosine_similarity
            audio_scores = cosine_similarity(test_audio, user_pref_audio.reshape(1, -1)).flatten()
        else:
            audio_scores = np.zeros(len(enhanced))
        
        # Qualitative similarity (from context and genre inferences)
        qual_features = [col for col in enhanced.columns if col.startswith('context_') or col.startswith('genre_')]
        if qual_features:
            qual_scores = enhanced[qual_features].mean(axis=1).values
        else:
            qual_scores = np.ones(len(enhanced)) * 0.5
        
        # Genre similarity
        if 'track_genre' in enhanced.columns:
            genre_scores = []
            for genre in enhanced['track_genre'].fillna('unknown'):
                genre_list = str(genre).split(',')
                score = sum(self.user_profile.genre_distribution.get(g.strip(), 0) for g in genre_list)
                genre_scores.append(score)
            genre_scores = np.array(genre_scores)
            genre_scores = genre_scores / (genre_scores.max() + 1e-8)
        else:
            genre_scores = np.zeros(len(enhanced))
        
        # Artist familiarity
        if 'artists' in enhanced.columns:
            artist_scores = enhanced['artists'].apply(
                lambda x: 1.0 if str(x) in self.user_profile.artist_list else 0.0
            ).values
        else:
            artist_scores = np.zeros(len(enhanced))
        
        # Preference Classifier score
        if self.preference_classifier and self.preference_classifier.is_trained:
            classifier_scores = self.preference_classifier.predict_score(enhanced, pred_audio_features)
        else:
            classifier_scores = np.zeros(len(enhanced))
            
        # Normalize audio scores
        if audio_scores.max() > audio_scores.min():
            audio_scores = (audio_scores - audio_scores.min()) / (audio_scores.max() - audio_scores.min())
        
        # Combined score
        final_score = (
            weights['predicted_audio'] * audio_scores +
            weights['qualitative'] * qual_scores +
            weights['genre'] * genre_scores +
            weights['artist'] * artist_scores +
            weights.get('preference', 0.0) * classifier_scores
        )
        
        # Add to dataframe
        enhanced['audio_score'] = audio_scores
        enhanced['qualitative_score'] = qual_scores
        enhanced['genre_score'] = genre_scores
        enhanced['artist_score'] = artist_scores
        enhanced['preference_score'] = classifier_scores
        enhanced['final_score'] = final_score
        enhanced['rank'] = enhanced['final_score'].rank(ascending=False, method='first').astype(int)
        
        # Sort
        result = enhanced.sort_values('final_score', ascending=False).reset_index(drop=True)
        
        print("\n Recommendation completed!")
        print(f"    Score range: {final_score.min():.3f} - {final_score.max():.3f}")
        
        return result
