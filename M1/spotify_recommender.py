"""
Spotify Track Recommender with Search-based Quality Enhancement
================================================================

System Overview:
1. Audio Feature Prediction Model (trained on 110K tracks)
2. Search-based Qualitative Analysis
3. Hybrid Recommendation Engine

Author: AI Assistant
Date: 2026-01-27
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.svm import SVR
import lightgbm as lgb
import pickle
import json
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class AudioFeaturePredictor:
    """
    Predicts audio features from textual metadata
    Supports multiple regression model types selected via model_type parameter
    """

    # 지원하는 모델 타입 목록
    SUPPORTED_MODELS = {
        'LightGBM': 'LightGBM Regressor',
        'LinearRegression': 'Linear Regression',
        'Ridge': 'Ridge Regression',
        'RandomForest': 'Random Forest Regressor',
        'GradientBoosting': 'Gradient Boosting Regressor',
        'SVR': 'Support Vector Regressor',
    }

    def __init__(self, model_type: str = 'Ridge'):
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(f"지원하지 않는 모델: {model_type}. 지원 목록: {list(self.SUPPORTED_MODELS.keys())}")

        self.model_type = model_type
        self.models = {}
        self.scalers = {}
        self.genre_encoder = MultiLabelBinarizer()
        self.artist_vectorizer = TfidfVectorizer(max_features=500)
        self.album_vectorizer = TfidfVectorizer(max_features=300)

        # Audio features to predict
        self.audio_features = [
            'danceability', 'energy', 'valence',
            'acousticness', 'instrumentalness', 'speechiness',
            'liveness', 'tempo', 'loudness'
        ]
        
    def prepare_features(self, df: pd.DataFrame, fit: bool = False) -> np.ndarray:
        """
        Convert textual metadata to numerical features
        """
        features_list = []
        
        # 1. Genre encoding (multi-hot)
        if 'track_genre' in df.columns:
            genres = df['track_genre'].fillna('unknown').str.split(',').tolist()
            if fit:
                genre_features = self.genre_encoder.fit_transform(genres)
            else:
                genre_features = self.genre_encoder.transform(genres)
            features_list.append(genre_features)
        
        # 2. Artist TF-IDF
        if 'artists' in df.columns:
            artists = df['artists'].fillna('unknown')
            if fit:
                artist_features = self.artist_vectorizer.fit_transform(artists).toarray()
            else:
                artist_features = self.artist_vectorizer.transform(artists).toarray()
            features_list.append(artist_features)
        
        # 3. Album TF-IDF
        if 'album_name' in df.columns:
            albums = df['album_name'].fillna('unknown')
            if fit:
                album_features = self.album_vectorizer.fit_transform(albums).toarray()
            else:
                album_features = self.album_vectorizer.transform(albums).toarray()
            features_list.append(album_features)
        
        # 4. Duration (normalized)
        if 'duration_ms' in df.columns:
            duration = df['duration_ms'].fillna(df['duration_ms'].median()).values.reshape(-1, 1)
            duration_normalized = duration / 300000  # normalize by 5 minutes
            features_list.append(duration_normalized)
        
        # 5. Popularity (if available)
        if 'popularity' in df.columns:
            popularity = df['popularity'].fillna(50).values.reshape(-1, 1) / 100
            features_list.append(popularity)
        
        # Combine all features
        X = np.hstack(features_list)
        return X
    
    def _create_model(self):
        """
        Step 4에서 선정된 모델 타입에 따라 회귀 모델 인스턴스를 생성
        """
        from sklearn.linear_model import LinearRegression, Ridge
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.svm import SVR

        if self.model_type == 'LightGBM':
            return lgb.LGBMRegressor(
                n_estimators=100, max_depth=5, learning_rate=0.1,
                random_state=42, n_jobs=-1, verbose=-1
            )
        elif self.model_type == 'LinearRegression':
            return LinearRegression()
        elif self.model_type == 'Ridge':
            return Ridge(alpha=1.0)
        elif self.model_type == 'RandomForest':
            return RandomForestRegressor(
                n_estimators=50, max_depth=10, random_state=42, n_jobs=-1
            )
        elif self.model_type == 'GradientBoosting':
            return GradientBoostingRegressor(
                n_estimators=50, max_depth=5, random_state=42
            )
        elif self.model_type == 'SVR':
            return SVR(kernel='rbf', C=1.0)

    def _needs_scaling(self) -> bool:
        """
        트리 기반 모델은 StandardScaler가 불필요
        거리/기울기 기반 모델만 스케일링 적용
        """
        return self.model_type in ['LinearRegression', 'Ridge', 'SVR']

    def train(self, df: pd.DataFrame, test_size: float = 0.2):
        """
        Train prediction models for each audio feature
        """
        print(f"[INFO] Training Audio Feature Prediction Models ({self.model_type})...")
        print(f" Training data: {len(df)} tracks")
        use_scaling = self._needs_scaling()
        if use_scaling:
            print(f" StandardScaler 적용 ({self.model_type}은 스케일링 필요)")
        else:
            print(f" StandardScaler 미적용 ({self.model_type}은 트리 기반 모델)")

        # Prepare features
        X = self.prepare_features(df, fit=True)

        results = {}

        for feature in self.audio_features:
            if feature not in df.columns:
                print(f"  Skipping {feature} (not in dataset)")
                continue

            print(f"\n Training {feature} predictor...")

            # Prepare target
            y = df[feature].fillna(df[feature].median()).values

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

            # 스케일링이 필요한 모델만 적용
            if use_scaling:
                scaler = StandardScaler()
                X_train_fit = scaler.fit_transform(X_train)
                X_test_fit = scaler.transform(X_test)
                self.scalers[feature] = scaler
            else:
                X_train_fit = X_train
                X_test_fit = X_test

            # 모델 생성 및 학습
            model = self._create_model()
            model.fit(X_train_fit, y_train)

            # Evaluate
            train_score = model.score(X_train_fit, y_train)
            test_score = model.score(X_test_fit, y_test)

            print(f"    Train R²: {train_score:.3f}")
            print(f"    Test R²: {test_score:.3f}")

            # Save model
            self.models[feature] = model

            results[feature] = {
                'train_r2': train_score,
                'test_r2': test_score
            }

        print("\n Training completed!")
        return results
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict audio features for tracks without them
        """
        X = self.prepare_features(df, fit=False)

        predictions = df.copy()

        for feature, model in self.models.items():
            if feature in self.scalers:
                X_input = self.scalers[feature].transform(X)
            else:
                X_input = X
            predictions[f'predicted_{feature}'] = model.predict(X_input)

        return predictions
    
    def save(self, path: str):
        """Save trained models"""
        with open(path, 'wb') as f:
            pickle.dump({
                'model_type': self.model_type,
                'models': self.models,
                'scalers': self.scalers,
                'genre_encoder': self.genre_encoder,
                'artist_vectorizer': self.artist_vectorizer,
                'album_vectorizer': self.album_vectorizer,
                'audio_features': self.audio_features
            }, f)
        print(f" Models saved to {path} (model_type: {self.model_type})")

    def load(self, path: str):
        """Load trained models"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.model_type = data.get('model_type', 'LightGBM')
            self.models = data['models']
            self.scalers = data.get('scalers', {})
            self.genre_encoder = data['genre_encoder']
            self.artist_vectorizer = data['artist_vectorizer']
            self.album_vectorizer = data['album_vectorizer']
            self.audio_features = data['audio_features']
        print(f" Models loaded from {path} (model_type: {self.model_type})")


class UserPreferenceProfile:
    """
    Builds user preference profile from liked tracks
    """
    
    def __init__(self):
        self.preference_tracks = None
        self.feature_stats = {}
        self.genre_distribution = {}
        self.artist_list = []
        
    def build_profile(self, df: pd.DataFrame):
        """
        Analyze user's preference tracks
        """
        print(f" Building user preference profile from {len(df)} tracks...")
        
        self.preference_tracks = df.copy()
        
        # Audio feature statistics
        audio_cols = [col for col in df.columns if col in [
            'danceability', 'energy', 'valence', 'acousticness',
            'instrumentalness', 'speechiness', 'liveness', 'tempo', 'loudness'
        ] or col.startswith('predicted_')]
        
        for col in audio_cols:
            if col in df.columns:
                self.feature_stats[col] = {
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'median': df[col].median(),
                    'q25': df[col].quantile(0.25),
                    'q75': df[col].quantile(0.75)
                }
        
        # Genre distribution
        if 'track_genre' in df.columns:
            genres = df['track_genre'].fillna('unknown').str.split(',').explode()
            self.genre_distribution = genres.value_counts(normalize=True).to_dict()
        
        # Artist list
        if 'artists' in df.columns:
            self.artist_list = df['artists'].unique().tolist()
        
        print(" User profile created!")
        print(f"    Audio feature stats: {len(self.feature_stats)} features")
        print(f"    Favorite genres: {list(self.genre_distribution.keys())[:5]}")
        print(f"    Unique artists: {len(self.artist_list)}")
        
        return self.feature_stats
    
    def get_summary(self) -> Dict:
        """Get profile summary"""
        return {
            'num_tracks': len(self.preference_tracks),
            'feature_stats': self.feature_stats,
            'top_genres': dict(list(self.genre_distribution.items())[:10]),
            'num_artists': len(self.artist_list)
        }

class PreferenceClassifier:
    """
    Learns user preferences from explicit feedback (likes/dislikes)
    Includes professional validation metrics and feature importance
    """
    def __init__(self):
        from sklearn.ensemble import GradientBoostingClassifier
        self.model = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.metrics = {}
        self.cv_results = None
        self.feature_names = []

    def prepare_data(self, df: pd.DataFrame, audio_features: List[str]) -> np.ndarray:
        return df[audio_features].fillna(0.5).values

    def train(self, likes_df: pd.DataFrame, dislikes_df: pd.DataFrame, audio_features: List[str], validate: bool = True):
        """
        Train the classifier on likes and dislikes
        """
        self.feature_names = audio_features
        if len(likes_df) < 2 or len(dislikes_df) < 2:
            print(" Not enough data for feedback-based training (minimum 2 likes and 2 dislikes required).")
            return

        likes_X = self.prepare_data(likes_df, audio_features)
        dislikes_X = self.prepare_data(dislikes_df, audio_features)
        
        X = np.vstack([likes_X, dislikes_X])
        y = np.array([1] * len(likes_X) + [0] * len(dislikes_X))
        
        # Shuffle
        indices = np.arange(len(X))
        np.random.seed(42)
        np.random.shuffle(indices)
        X, y = X[indices], y[indices]

        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
        if validate:
            self.evaluate(X_scaled, y)
            self.perform_cv(X_scaled, y)
            
        print(f" Feedback model trained on {len(X)} samples.")

    def evaluate(self, X_scaled: np.ndarray, y: np.ndarray):
        """Calculate detailed performance metrics"""
        y_pred = self.model.predict(X_scaled)
        y_prob = self.model.predict_proba(X_scaled)[:, 1]
        
        self.metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'f1': f1_score(y, y_pred),
            'auc_roc': roc_auc_score(y, y_prob)
        }
        return self.metrics

    def perform_cv(self, X_scaled: np.ndarray, y: np.ndarray, cv: int = 5):
        """Perform K-Fold Cross Validation"""
        if len(y) < cv:
            print(f" Skipping CV: Data size ({len(y)}) smaller than requested folds ({cv})")
            return None
            
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        self.cv_results = cross_validate(self.model, X_scaled, y, cv=skf, scoring=scoring)
        return self.cv_results

    def get_feature_importance(self) -> pd.DataFrame:
        """Get importance of each audio feature"""
        if not self.is_trained:
            return pd.DataFrame()
            
        importance = self.model.feature_importances_
        df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        return df

    def predict_score(self, df: pd.DataFrame, audio_features: List[str]) -> np.ndarray:
        if not self.is_trained:
            return np.ones(len(df)) * 0.5
        
        X = self.prepare_data(df, audio_features)
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]

    def save(self, path: str):
        """Save preference model"""
        if not self.is_trained:
            print(" Model not trained yet. Nothing to save.")
            return
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'is_trained': self.is_trained,
                'metrics': self.metrics,
                'feature_names': self.feature_names
            }, f)
        print(f" Preference model saved to {path}")

    def load(self, path: str):
        """Load preference model"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.scaler = data['scaler']
            self.is_trained = data['is_trained']
            self.metrics = data.get('metrics', {})
            self.feature_names = data.get('feature_names', [])
        print(f" Preference model loaded from {path}")


class HybridRecommender:
    """
    Hybrid recommendation system combining:
    1. Predicted audio features
    2. Metadata similarity
    3. Search-based qualitative scores (placeholder for integration)
    """
    
    def __init__(self, predictor: AudioFeaturePredictor, user_profile: UserPreferenceProfile):
        self.predictor = predictor
        self.user_profile = user_profile
        
    def calculate_audio_similarity(self, test_tracks: pd.DataFrame) -> np.ndarray:
        """
        Calculate similarity based on audio features
        """
        audio_features = [col for col in test_tracks.columns 
                         if col.startswith('predicted_') or col in self.predictor.audio_features]
        
        if not audio_features:
            return np.zeros(len(test_tracks))
        
        # User preference vector (mean of liked tracks)
        pref_vector = np.array([
            self.user_profile.feature_stats.get(feat, {}).get('mean', 0.5)
            for feat in audio_features
        ]).reshape(1, -1)
        
        # Test tracks vectors
        test_vectors = test_tracks[audio_features].fillna(0.5).values
        
        # Cosine similarity
        similarities = cosine_similarity(test_vectors, pref_vector).flatten()
        
        return similarities
    
    def calculate_genre_similarity(self, test_tracks: pd.DataFrame) -> np.ndarray:
        """
        Calculate genre-based similarity
        """
        if 'track_genre' not in test_tracks.columns:
            return np.zeros(len(test_tracks))
        
        scores = []
        for genre in test_tracks['track_genre'].fillna('unknown'):
            genre_list = genre.split(',') if isinstance(genre, str) else []
            score = sum(self.user_profile.genre_distribution.get(g, 0) for g in genre_list)
            scores.append(score)
        
        return np.array(scores)
    
    def calculate_artist_familiarity(self, test_tracks: pd.DataFrame) -> np.ndarray:
        """
        Check if artist is in user's preference list
        """
        if 'artists' not in test_tracks.columns:
            return np.zeros(len(test_tracks))
        
        scores = []
        for artist in test_tracks['artists'].fillna('unknown'):
            score = 1.0 if artist in self.user_profile.artist_list else 0.0
            scores.append(score)
        
        return np.array(scores)
    
    def recommend(self, test_tracks: pd.DataFrame, 
                  weights: Dict[str, float] = None) -> pd.DataFrame:
        """
        Generate recommendations with hybrid scoring
        """
        if weights is None:
            weights = {
                'audio': 0.5,
                'genre': 0.3,
                'artist': 0.2
            }
        
        print(f"\n Evaluating {len(test_tracks)} test tracks...")
        
        # Calculate component scores
        audio_sim = self.calculate_audio_similarity(test_tracks)
        genre_sim = self.calculate_genre_similarity(test_tracks)
        artist_sim = self.calculate_artist_familiarity(test_tracks)
        
        # Normalize scores to 0-1
        audio_sim = (audio_sim - audio_sim.min()) / (audio_sim.max() - audio_sim.min() + 1e-8)
        genre_sim = genre_sim / (genre_sim.max() + 1e-8)
        
        # Combined score
        final_score = (
            weights['audio'] * audio_sim +
            weights['genre'] * genre_sim +
            weights['artist'] * artist_sim
        )
        
        # Add scores to dataframe
        results = test_tracks.copy()
        results['audio_similarity'] = audio_sim
        results['genre_similarity'] = genre_sim
        results['artist_familiarity'] = artist_sim
        results['recommendation_score'] = final_score
        results['rank'] = results['recommendation_score'].rank(ascending=False, method='first').astype(int)
        
        # Sort by score
        results = results.sort_values('recommendation_score', ascending=False).reset_index(drop=True)
        
        print(" Evaluation completed!")
        print(f"    Score range: {final_score.min():.3f} - {final_score.max():.3f}")
        print(f"    Top track: {results.iloc[0]['track_name'] if 'track_name' in results else 'N/A'}")
        
        return results
    
    def explain_recommendation(self, track_data: pd.Series) -> str:
        """
        Generate explanation for a recommendation
        """
        explanation = []
        explanation.append(f" Track: {track_data.get('track_name', 'Unknown')}")
        explanation.append(f" Artist: {track_data.get('artists', 'Unknown')}")
        explanation.append(f" Overall Score: {track_data.get('recommendation_score', 0):.3f}")
        explanation.append(f"\nScore Breakdown:")
        explanation.append(f"  • Audio Similarity: {track_data.get('audio_similarity', 0):.3f}")
        explanation.append(f"  • Genre Match: {track_data.get('genre_similarity', 0):.3f}")
        explanation.append(f"  • Artist Familiarity: {track_data.get('artist_familiarity', 0):.3f}")
        
        return "\n".join(explanation)


def main_pipeline(
    training_data_path: str,
    preference_data_path: str,
    test_data_path: str,
    output_path: str = 'outputs/recommendations.csv'
):
    """
    Complete pipeline for track recommendation
    """
    print("=" * 60)
    print(" SPOTIFY TRACK RECOMMENDER WITH SEARCH ENHANCEMENT")
    print("=" * 60)
    
    # Step 1: Load data
    print("\n Loading data...")
    train_df = pd.read_csv(training_data_path)
    preference_df = pd.read_csv(preference_data_path)
    test_df = pd.read_csv(test_data_path)
    
    print(f"    Training data: {len(train_df)} tracks")
    print(f"    Preference data: {len(preference_df)} tracks")
    print(f"    Test data: {len(test_df)} tracks")
    
    # Step 2: Train audio feature predictor
    print("\n" + "=" * 60)
    predictor = AudioFeaturePredictor(model_type='Ridge')
    training_results = predictor.train(train_df)
    predictor.save('outputs/audio_predictor.pkl')
    
    # Step 3: Predict features for preference tracks (if needed)
    print("\n" + "=" * 60)
    print(" Predicting audio features for preference tracks...")
    preference_enhanced = predictor.predict(preference_df)
    
    # Step 4: Build user profile
    print("\n" + "=" * 60)
    user_profile = UserPreferenceProfile()
    user_profile.build_profile(preference_enhanced)
    
    # Step 5: Predict features for test tracks
    print("\n" + "=" * 60)
    print(" Predicting audio features for test tracks...")
    test_enhanced = predictor.predict(test_df)
    
    # Step 6: Generate recommendations
    print("\n" + "=" * 60)
    recommender = HybridRecommender(predictor, user_profile)
    recommendations = recommender.recommend(test_enhanced)
    
    # Step 7: Save results
    recommendations.to_csv(output_path, index=False)
    print(f"\n Results saved to: {output_path}")
    
    return recommendations, predictor, user_profile, recommender
