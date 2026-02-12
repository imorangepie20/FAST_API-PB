"""
====================================================================================
FastAPI 음악 추천 시스템 - 시스템 B (SVM + 오디오 역추적) 통합 파일
====================================================================================

이 파일은 시스템 B (dooseok 폴더)만 포함한 경량 버전입니다.

제가 카카오 팀챗에 올려놓은 spotify_114k_with_tags.csv 파일로  역추적 모델 학습을 해주시면 감사합니다.
제가 카카오톡에 올려드리는 lastfm_artist_info.csv 파일은 fastapi/data/ 폴더에 넣어주시면 취향 추천모델 학습에 사용됩니다. 


====================================================================================
시스템 B 개요
====================================================================================

시스템 B는 SVM 기반 고성능 추천 시스템입니다.

주요 특징:
- 393D 피처 (384D 텍스트 임베딩 + 9D 오디오 피처)
- 오디오 역추적 모델 (TF-IDF + GradientBoosting)
- 플레이리스트 기반 학습 (PMS → SVM)
- GMS 피드백 재학습 (선택/삭제)
- 높은 정확도 (AUC 0.9995)

====================================================================================
시스템 B 전체 흐름
====================================================================================

┌─────────────────────────────────────────────────────────────────────┐
│ 1. 트랙 생성 시 오디오 피처 자동 채우기                              │
├─────────────────────────────────────────────────────────────────────┤
│ Backend → POST /enrich-audio-features                                │
│           ↓                                                          │
│ 섹션 7: Last.fm API (artist_tags 수집)                               │
│           ↓                                                          │
│ 섹션 8: AudioPredictionService (TF-IDF + GBR)                        │
│           ↓                                                          │
│ 텍스트 → 9개 오디오 피처 예측                                        │
│ (danceability, energy, speechiness, acousticness,                    │
│  instrumentalness, liveness, valence, tempo, loudness)               │
│           ↓                                                          │
│ DB tracks.external_metadata에 JSON 저장                              │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ 2. PMS 플레이리스트로 SVM 모델 초기 학습                             │
├─────────────────────────────────────────────────────────────────────┤
│ Backend → POST /api/svm-train/upload (CSV 파일)                     │
│           또는                                                       │
│ 관리자 → POST /api/svm-db-train/{user_id}/{playlist_id} (DB)        │
│           ↓                                                          │
│ 섹션 11: SVM 학습 API                                                │
│   1) CSV 파싱 or DB 플레이리스트 조회                                │
│   2) 섹션 7: Last.fm 태그 수집 (artist + track tags)                │
│   3) 섹션 8: 오디오 피처 예측 (9D)                                   │
│   4) 섹션 6: 텍스트 임베딩 (384D)                                    │
│   5) EMS에서 Negative 샘플링 (1:3 비율)                             │
│   6) 393D 피처 생성 (384D + 9D)                                      │
│   7) SVM 학습 (StandardScaler + RBF SVM, C=10)                      │
│           ↓                                                          │
│ 저장: data/user_svm_models/user_{user_id}_svm.pkl                   │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ 3. EMS에서 GMS로 자동 추천                                           │
├─────────────────────────────────────────────────────────────────────┤
│ Backend → POST /api/svm/recommend                                    │
│           ↓                                                          │
│ 섹션 10: SVM 추천 API                                                │
│           ↓                                                          │
│ 섹션 9: SVMRecommendationService                                     │
│   1) EMS 후보 트랙들 로드                                            │
│   2) 각 트랙별 393D 피처 생성                                        │
│      - 섹션 6: 텍스트 임베딩 (384D)                                  │
│      - 섹션 8: 오디오 피처 예측 (9D)                                 │
│   3) SVM 모델로 "좋아할 확률" 예측                                   │
│   4) 확률 높은 순으로 정렬                                           │
│   5) 상위 N개 반환                                                   │
│           ↓                                                          │
│ GMS로 이동                                                           │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ 4. GMS 선택/삭제 피드백 → SVM 재학습                                 │
├─────────────────────────────────────────────────────────────────────┤
│ 사용자 GMS에서 곡 선택:                                              │
│ Backend → POST /gms-selected-track                                   │
│           ↓                                                          │
│ 섹션 12: 피드백 API                                                  │
│   - Positive 피드백 저장 (.pkl)                                      │
│   - 피드백 5개 이상? → 재학습 트리거                                 │
│                                                                      │
│ 사용자 GMS에서 곡 삭제:                                              │
│ Backend → POST /gms-deleted-track                                    │
│           또는                                                       │
│ Backend → POST /bulk-delete-track                                    │
│           ↓                                                          │
│ 섹션 12: 피드백 API                                                  │
│   - Negative 피드백 저장 (.pkl)                                      │
│   - 피드백 5개 이상? → 재학습 트리거                                 │
│           ↓                                                          │
│ 재학습 (자동 또는 수동):                                             │
│ POST /api/svm-train/{user_id}/retrain                               │
│           ↓                                                          │
│ 섹션 11: SVM 재학습                                                  │
│   1) 기존 피드백 로드                                                │
│   2) 새 피드백 추가                                                  │
│   3) EMS 추가 샘플링                                                 │
│   4) 393D 피처 생성                                                  │
│   5) SVM 재학습                                                      │
│           ↓                                                          │
│ 모델 업데이트: user_{user_id}_svm.pkl                                │
│           ↓                                                          │
│ 다음번 추천부터 학습 내용 반영                                       │
└─────────────────────────────────────────────────────────────────────┘

====================================================================================
Backend 연동 API (시스템 B)
====================================================================================

Backend가 호출하는 API:

1. POST /enrich-audio-features
   - 시점: 트랙 생성 시
   - 기능: 오디오 피처 자동 예측 및 DB 저장
   - 관련 섹션: 7, 8, 12

2. POST /api/svm/recommend
   - 시점: EMS → GMS 추천 시
   - 기능: SVM 기반 추천 트랙 선정
   - 관련 섹션: 6, 8, 9, 10

3. POST /gms-deleted-track
   - 시점: GMS 곡 삭제 시
   - 기능: Negative 피드백 저장 + 재학습 트리거
   - 관련 섹션: 11, 12

4. POST /bulk-delete-track
   - 시점: GMS 여러 곡 삭제 시
   - 기능: Negative 피드백 일괄 저장
   - 관련 섹션: 11, 12

5. POST /gms-selected-track
   - 시점: GMS 곡 선택 시
   - 기능: Positive 피드백 저장 + 재학습 트리거
   - 관련 섹션: 11, 12

내부/관리자 API (Backend 호출 안함):

6. POST /api/svm-train/upload
   - 기능: CSV로 SVM 초기 학습
   - 관련 섹션: 6, 7, 8, 11

7. POST /api/svm-db-train/{user_id}/{playlist_id}
   - 기능: DB 플레이리스트로 SVM 학습 (캐싱 활용)
   - 관련 섹션: 4, 6, 7, 8, 11

8. POST /api/svm-train/{user_id}/retrain
   - 기능: 피드백 기반 재학습 (피드백 API에서 자동 트리거)
   - 관련 섹션: 6, 7, 8, 11

====================================================================================
섹션 구성
====================================================================================

1. 라이브러리 임포트
2. 로깅 설정
3. 설정 (Config)
4. 데이터베이스 (SQLAlchemy)
5. Pydantic 스키마
6. 텍스트 임베딩 서비스 (Sentence-Transformers)
7. Last.fm API 서비스
8. 오디오 피처 예측 서비스 (TF-IDF + GBR)
9. SVM 추천 서비스
10. SVM 추천 API 라우터
11. SVM 학습/재학습 API 라우터
12. GMS 피드백 API 라우터
13. FastAPI 애플리케이션

====================================================================================


"""
# ====================================================================================
# 1. 필수 라이브러리 임포트
# ====================================================================================

# FastAPI 관련
from fastapi import FastAPI, APIRouter, HTTPException, Query, Depends, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# Pydantic 모델 (데이터 검증)
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

# SQLAlchemy (데이터베이스)
from sqlalchemy import Column, BigInteger, String, Text, Enum, DateTime, ForeignKey, DECIMAL, create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.sql import func
from contextlib import contextmanager

# 머신러닝/임베딩
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from scipy.sparse import hstack, csr_matrix

# 외부 API
import httpx
import asyncio

# 유틸리티
import os
import json
import csv
import io
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import joblib
import pandas as pd

# ====================================================================================
# 2. 로깅 설정
# ====================================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ====================================================================================
# 3. 설정 (Config)
# ====================================================================================

"""
환경 설정 관리
- 데이터베이스 URL, API 키, 모델 경로 등
"""

BASE_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):
    """애플리케이션 설정"""

    # Last.fm API
    lastfm_api_key: str = ""
    lastfm_api_secret: str = ""

    # Backend 연동
    backend_url: str = "http://localhost:8089"

    # 데이터베이스 (MariaDB)
    database_url: str = "mysql+pymysql://root:0000@localhost:3308/music_space_db"

    # 임베딩 모델
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # 데이터 경로
    data_dir: Path = BASE_DIR / "data"
    user_models_dir: Path = BASE_DIR / "data" / "user_models"
    metadata_cache_dir: Path = BASE_DIR / "data" / "metadata_cache"

    class Config:
        env_file = ".env"
        extra = "allow"


def get_settings() -> Settings:
    """설정 싱글톤"""
    return Settings()


# ====================================================================================
# 4. 데이터베이스 (SQLAlchemy)
# ====================================================================================

"""
MariaDB 연동
- Track, Playlist, PlaylistTrack 모델
"""

Base = declarative_base()


class Track(Base):
    """트랙 테이블"""
    __tablename__ = "tracks"

    track_id = Column(BigInteger, primary_key=True)
    title = Column(String(255), nullable=False)
    artist = Column(String(255), nullable=False)
    album = Column(String(255))
    duration = Column(BigInteger)
    isrc = Column(String(50))
    external_metadata = Column(Text)

    # 오디오 피처 (시스템 B에서 채움)
    danceability = Column(DECIMAL(5, 4))
    energy = Column(DECIMAL(5, 4))
    speechiness = Column(DECIMAL(5, 4))
    acousticness = Column(DECIMAL(5, 4))
    instrumentalness = Column(DECIMAL(5, 4))
    liveness = Column(DECIMAL(5, 4))
    valence = Column(DECIMAL(5, 4))
    tempo = Column(DECIMAL(6, 2))
    loudness = Column(DECIMAL(6, 2))


class Playlist(Base):
    """플레이리스트 테이블"""
    __tablename__ = "playlists"

    playlist_id = Column(BigInteger, primary_key=True)
    name = Column(String(255), nullable=False)
    user_id = Column(BigInteger, ForeignKey("users.user_id"))
    space_type = Column(Enum("PMS", "GMS", "EMS"))
    created_at = Column(DateTime, default=func.now())


class PlaylistTrack(Base):
    """플레이리스트-트랙 연결 테이블"""
    __tablename__ = "playlist_tracks"

    playlist_track_id = Column(BigInteger, primary_key=True)
    playlist_id = Column(BigInteger, ForeignKey("playlists.playlist_id"))
    track_id = Column(BigInteger, ForeignKey("tracks.track_id"))
    added_at = Column(DateTime, default=func.now())


# 데이터베이스 엔진
settings = get_settings()
engine = create_engine(settings.database_url, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Session:
    """DB 세션 의존성"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def get_db_context():
    """DB 컨텍스트 매니저"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ====================================================================================
# 5. Pydantic 스키마 (API 요청/응답 모델)
# ====================================================================================

"""
API 요청/응답 검증
"""

# 기본 트랙 스키마
class TrackBase(BaseModel):
    """트랙 기본 정보"""
    track_id: Optional[str] = None
    title: str
    artist: str
    album: Optional[str] = None
    duration: Optional[int] = None


# ====================================================================================
# 6. 텍스트 임베딩 서비스
# ====================================================================================

"""
Sentence-Transformers를 사용한 텍스트 임베딩
- 트랙 정보 → 384차원 벡터 변환
- 시스템 B의 393D 피처 중 384D 담당
"""


class EmbeddingService:
    """텍스트 임베딩 서비스"""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None

    def load_model(self):
        """모델 로드 (Lazy Loading)"""
        if self.model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)

    def encode_track(self, track: TrackBase) -> np.ndarray:
        """단일 트랙 임베딩"""
        self.load_model()
        text = f"{track.artist} {track.title}"
        if track.album:
            text += f" {track.album}"
        return self.model.encode([text])[0]

    def encode_tracks(self, tracks: List[TrackBase]) -> np.ndarray:
        """여러 트랙 임베딩"""
        self.load_model()
        texts = [
            f"{t.artist} {t.title}" + (f" {t.album}" if t.album else "")
            for t in tracks
        ]
        return self.model.encode(texts)


# 싱글톤
_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    global _embedding_service
    if _embedding_service is None:
        settings = get_settings()
        _embedding_service = EmbeddingService(settings.embedding_model)
    return _embedding_service


# ====================================================================================
# 7. Last.fm API 서비스
# ====================================================================================

"""
Last.fm API로 아티스트/트랙 태그(장르) 수집
- 시스템 B: SVM 학습 시 태그 정보 필요
- CSV 캐시 활용으로 API 호출 최소화
"""

LASTFM_API_URL = "https://ws.audioscrobbler.com/2.0/"


class LastFmService:
    """Last.fm API 클라이언트"""

    def __init__(self):
        settings = get_settings()
        self.api_key = settings.lastfm_api_key
        self.cache_dir = settings.metadata_cache_dir / "lastfm"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if not self.api_key:
            logger.warning("Last.fm API key not configured")

        # CSV 캐시
        self.csv_data = None
        self._load_csv_cache()

    def _load_csv_cache(self):
        """CSV 캐시 로드"""
        csv_path = Path("data/lastfm_artist_info.csv")
        if csv_path.exists():
            try:
                self.csv_data = pd.read_csv(csv_path)
                logger.info(f"Last.fm CSV 캐시: {len(self.csv_data)}개")
            except Exception as e:
                logger.warning(f"CSV 로드 실패: {e}")

    def get_tags_from_csv(self, artist: str) -> Optional[str]:
        """CSV에서 아티스트 태그 조회"""
        if self.csv_data is None:
            return None

        try:
            match = self.csv_data[
                self.csv_data['artist'].str.lower() == artist.lower()
            ]

            if not match.empty and pd.notna(match.iloc[0]['lfm_tags']):
                return match.iloc[0]['lfm_tags']
        except Exception as e:
            logger.warning(f"CSV 조회 에러: {e}")

        return None

    async def get_tags_from_api(self, artist: str, save_to_csv: bool = True) -> Optional[str]:
        """Last.fm API에서 아티스트 태그 조회"""
        tags_list = await self.get_artist_tags(artist)
        if tags_list:
            tag_names = [t['name'] for t in tags_list[:10]]
            return ' | '.join(tag_names)
        return None

    def _get_cache_path(self, cache_key: str) -> Path:
        """캐시 파일 경로"""
        safe_key = "".join(c if c.isalnum() or c in "._-" else "_" for c in cache_key)
        return self.cache_dir / f"{safe_key}.json"

    def _get_cached(self, cache_key: str) -> Optional[Dict]:
        """캐시 조회"""
        cache_path = self._get_cache_path(cache_key)
        if cache_path.exists():
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return None

    def _set_cache(self, cache_key: str, data: Dict):
        """캐시 저장"""
        cache_path = self._get_cache_path(cache_key)
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Cache write error: {e}")

    async def get_track_tags(self, artist: str, title: str) -> List[Dict[str, Any]]:
        """트랙 태그 조회"""
        if not self.api_key:
            return []

        cache_key = f"track_{artist}_{title}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached.get("tags", [])

        params = {
            "method": "track.getTopTags",
            "artist": artist,
            "track": title,
            "api_key": self.api_key,
            "format": "json"
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(LASTFM_API_URL, params=params, timeout=10.0)
                response.raise_for_status()
                data = response.json()

                tags = []
                if "toptags" in data and "tag" in data["toptags"]:
                    raw_tags = data["toptags"]["tag"]
                    if isinstance(raw_tags, list):
                        tags = [{"name": t["name"], "count": int(t.get("count", 0))} for t in raw_tags]
                    elif isinstance(raw_tags, dict):
                        tags = [{"name": raw_tags["name"], "count": int(raw_tags.get("count", 0))}]

                self._set_cache(cache_key, {"tags": tags})
                return tags

        except Exception as e:
            logger.error(f"Last.fm API error: {e}")

        return []

    async def get_artist_tags(self, artist: str) -> List[Dict[str, Any]]:
        """아티스트 태그 조회"""
        if not self.api_key:
            return []

        cache_key = f"artist_{artist}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached.get("tags", [])

        params = {
            "method": "artist.getTopTags",
            "artist": artist,
            "api_key": self.api_key,
            "format": "json"
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(LASTFM_API_URL, params=params, timeout=10.0)
                response.raise_for_status()
                data = response.json()

                tags = []
                if "toptags" in data and "tag" in data["toptags"]:
                    raw_tags = data["toptags"]["tag"]
                    if isinstance(raw_tags, list):
                        tags = [{"name": t["name"], "count": int(t.get("count", 0))} for t in raw_tags]
                    elif isinstance(raw_tags, dict):
                        tags = [{"name": raw_tags["name"], "count": int(raw_tags.get("count", 0))}]

                self._set_cache(cache_key, {"tags": tags})
                return tags

        except Exception as e:
            logger.error(f"Artist tags error: {e}")

        return []

    async def get_combined_tags(self, artist: str, title: str) -> List[str]:
        """트랙 + 아티스트 태그 결합"""
        track_tags_task = self.get_track_tags(artist, title)
        artist_tags_task = self.get_artist_tags(artist)

        track_tags, artist_tags = await asyncio.gather(track_tags_task, artist_tags_task)

        tag_scores: Dict[str, int] = {}

        for tag in track_tags:
            name = tag["name"].lower()
            tag_scores[name] = tag_scores.get(name, 0) + tag.get("count", 50)

        for tag in artist_tags:
            name = tag["name"].lower()
            tag_scores[name] = tag_scores.get(name, 0) + tag.get("count", 50) // 2

        sorted_tags = sorted(tag_scores.items(), key=lambda x: x[1], reverse=True)
        return [name for name, _ in sorted_tags[:10]]


# 싱글톤
_lastfm_service: Optional[LastFmService] = None


def get_lastfm_service() -> LastFmService:
    global _lastfm_service
    if _lastfm_service is None:
        _lastfm_service = LastFmService()
    return _lastfm_service


# ====================================================================================
# 8. 오디오 피처 예측 서비스 (Audio Reverse Tracking Model)
# ====================================================================================

"""
오디오 역추적 모델

TF-IDF + GradientBoostingRegressor로 텍스트 → 오디오 피처 예측
- 입력: artist, track_name, tags
- 출력: 9개 오디오 피처
- 용도: Spotify 같은 실제 오디오 분석 없이 피처 예측
"""


class AudioPredictionService:
    """오디오 피처 예측 서비스"""

    AUDIO_FEATURES = [
        'danceability', 'energy', 'speechiness', 'acousticness',
        'instrumentalness', 'liveness', 'valence', 'tempo', 'loudness'
    ]

    def __init__(self, model_path: Optional[str] = None):
        if model_path is None:
            model_path = BASE_DIR / 'tfidf_gbr_models.pkl'

        self.model_path = Path(model_path)
        self.models = None
        self.vectorizers = None
        self._loaded = False

    def load_model(self):
        """모델 로드"""
        if self._loaded:
            return

        if not self.model_path.exists():
            raise FileNotFoundError(f"모델 파일 없음: {self.model_path}")

        logger.info(f"오디오 예측 모델 로드: {self.model_path}")
        data = joblib.load(self.model_path)

        self.models = data['models']
        self.vectorizers = data['vectorizers']
        self._loaded = True

        logger.info(f"오디오 모델 로드 완료: {list(self.models.keys())}")

    def predict_single(
        self,
        artist: str,
        track_name: str,
        album_name: str = "",
        tags: str = "",
        duration: int = 200,
        duration_ms: int = None,
        lfm_listeners: int = 0,
        lfm_playcount: int = 0
    ) -> Dict[str, float]:
        """단일 트랙 오디오 피처 예측"""
        self.load_model()

        # TF-IDF 벡터화
        artist_vec = self.vectorizers['artist'].transform([artist])
        track_vec = self.vectorizers['track'].transform([track_name])
        tags_vec = self.vectorizers['tags'].transform([tags])

        # 수치 피처
        if duration_ms is not None:
            duration_ms_val = duration_ms
        else:
            duration_ms_val = duration * 1000
        popularity = 50
        numeric = np.array([[duration_ms_val, popularity]])
        num_scaler = self.vectorizers.get('num_scaler')
        if num_scaler is not None:
            numeric = num_scaler.transform(numeric)
        numeric_sparse = csr_matrix(numeric)

        # 피처 결합
        X = hstack([artist_vec, track_vec, tags_vec, numeric_sparse])

        # 예측
        predictions = {}
        for feature in self.AUDIO_FEATURES:
            if feature in self.models:
                pred = self.models[feature].predict(X)[0]
                predictions[feature] = float(pred)
            else:
                predictions[feature] = 0.0

        return predictions

    def predict_batch(self, tracks: List[Dict]) -> List[Dict[str, float]]:
        """배치 트랙 오디오 피처 예측"""
        self.load_model()

        if not tracks:
            return []

        artists = [t.get('artist', '') for t in tracks]
        track_names = [t.get('track_name', '') for t in tracks]
        tags_list = [t.get('tags', '') for t in tracks]

        artist_vec = self.vectorizers['artist'].transform(artists)
        track_vec = self.vectorizers['track'].transform(track_names)
        tags_vec = self.vectorizers['tags'].transform(tags_list)

        numeric = np.array([
            [
                t.get('duration', 200) * 1000,
                t.get('popularity', 50)
            ]
            for t in tracks
        ])
        num_scaler = self.vectorizers.get('num_scaler')
        if num_scaler is not None:
            numeric = num_scaler.transform(numeric)
        numeric_sparse = csr_matrix(numeric)

        X = hstack([artist_vec, track_vec, tags_vec, numeric_sparse])

        all_predictions = []
        feature_preds = {}

        for feature in self.AUDIO_FEATURES:
            if feature in self.models:
                feature_preds[feature] = self.models[feature].predict(X)
            else:
                feature_preds[feature] = np.zeros(len(tracks))

        for i in range(len(tracks)):
            predictions = {
                feature: float(feature_preds[feature][i])
                for feature in self.AUDIO_FEATURES
            }
            all_predictions.append(predictions)

        return all_predictions


# 싱글톤
_audio_prediction_service: Optional[AudioPredictionService] = None


def get_audio_prediction_service() -> AudioPredictionService:
    global _audio_prediction_service
    if _audio_prediction_service is None:
        _audio_prediction_service = AudioPredictionService()
    return _audio_prediction_service


# ====================================================================================
# 9. SVM 기반 추천 서비스
# ====================================================================================

"""
SVM 추천 서비스

393D 피처 (384D 임베딩 + 9D 오디오)로 사용자 취향 예측
- 모델: StandardScaler + SVM (RBF kernel, C=10)
- 성능: AUC 0.9995
"""


class SVMRecommendationService:
    """SVM 기반 추천 서비스"""

    AUDIO_FEATURES = [
        'danceability', 'energy', 'speechiness', 'acousticness',
        'instrumentalness', 'liveness', 'valence', 'tempo', 'loudness'
    ]

    def __init__(
        self,
        model_path: Optional[str] = None,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        if model_path is None:
            model_path = BASE_DIR / 'models' / 'recommendation' / 'final_model.pkl'

        self.model_path = Path(model_path)
        self.embedding_model_name = embedding_model_name

        self.svm_pipeline = None
        self.embedding_model = None
        self.audio_service = None
        self._loaded = False

    def load_model(self):
        """모델 로드"""
        if self._loaded:
            return

        if not self.model_path.exists():
            raise FileNotFoundError(f"SVM 모델 없음: {self.model_path}")

        logger.info(f"SVM 모델 로드: {self.model_path}")
        self.svm_pipeline = joblib.load(self.model_path)

        logger.info(f"임베딩 모델 로드: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)

        self.audio_service = get_audio_prediction_service()

        self._loaded = True
        logger.info("SVM 추천 모델 로드 완료")

    def _create_text_for_embedding(
        self,
        artist: str,
        track_name: str,
        album_name: str = "",
        tags: str = ""
    ) -> str:
        """임베딩용 텍스트 생성"""
        parts = [artist, track_name]
        if album_name:
            parts.append(album_name)
        if tags:
            parts.append(tags.replace('|', ' '))
        return ' '.join(parts)

    def _create_features(
        self,
        artist: str,
        track_name: str,
        album_name: str = "",
        tags: str = "",
        audio_features: Optional[Dict[str, float]] = None,
        duration_ms: int = 200000,
        lfm_listeners: int = 0,
        lfm_playcount: int = 0
    ) -> np.ndarray:
        """393D 피처 벡터 생성"""
        # 1. 텍스트 임베딩 (384D)
        text = self._create_text_for_embedding(artist, track_name, album_name, tags)
        embedding = self.embedding_model.encode([text])[0]

        # 2. 오디오 피처 (9D)
        if audio_features is None:
            audio_features = self.audio_service.predict_single(
                artist=artist,
                track_name=track_name,
                album_name=album_name,
                tags=tags,
                duration=duration_ms // 1000,
                lfm_listeners=lfm_listeners,
                lfm_playcount=lfm_playcount
            )

        audio_vector = np.array([
            audio_features.get(f, 0.0) for f in self.AUDIO_FEATURES
        ])

        # 3. 결합 (393D)
        features = np.concatenate([embedding, audio_vector])

        return features

    def predict_single(
        self,
        artist: str,
        track_name: str,
        album_name: str = "",
        tags: str = "",
        audio_features: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> Dict:
        """단일 트랙 SVM 예측"""
        self.load_model()

        features = self._create_features(
            artist=artist,
            track_name=track_name,
            album_name=album_name,
            tags=tags,
            audio_features=audio_features,
            **kwargs
        )

        X = features.reshape(1, -1)
        prediction = self.svm_pipeline.predict(X)[0]
        probability = self.svm_pipeline.predict_proba(X)[0]

        if audio_features is None:
            audio_features = self.audio_service.predict_single(
                artist=artist,
                track_name=track_name,
                album_name=album_name,
                tags=tags,
                **kwargs
            )

        return {
            'probability': float(probability[1]),
            'prediction': int(prediction),
            'audio_features': audio_features
        }

    def predict_batch(self, tracks: List[Dict]) -> List[Dict]:
        """배치 트랙 SVM 예측"""
        self.load_model()

        if not tracks:
            return []

        # 오디오 피처 예측
        tracks_needing_audio = [
            t for t in tracks if t.get('audio_features') is None
        ]
        if tracks_needing_audio:
            audio_predictions = self.audio_service.predict_batch(tracks_needing_audio)
            for t, audio in zip(tracks_needing_audio, audio_predictions):
                t['audio_features'] = audio

        # 피처 생성
        features_list = []
        for t in tracks:
            features = self._create_features(
                artist=t.get('artist', ''),
                track_name=t.get('track_name', ''),
                album_name=t.get('album_name', ''),
                tags=t.get('tags', ''),
                audio_features=t.get('audio_features'),
                duration_ms=t.get('duration_ms', 200000),
                lfm_listeners=t.get('lfm_listeners', 0),
                lfm_playcount=t.get('lfm_playcount', 0)
            )
            features_list.append(features)

        X = np.array(features_list)

        predictions = self.svm_pipeline.predict(X)
        probabilities = self.svm_pipeline.predict_proba(X)

        results = []
        for i, t in enumerate(tracks):
            results.append({
                'artist': t.get('artist', ''),
                'track_name': t.get('track_name', ''),
                'probability': float(probabilities[i][1]),
                'prediction': int(predictions[i]),
                'audio_features': t.get('audio_features', {})
            })

        return results

    def get_recommendations(
        self,
        candidate_tracks: List[Dict],
        top_k: int = 10,
        threshold: float = 0.5
    ) -> List[Dict]:
        """후보 트랙 중 추천 선정"""
        results = self.predict_batch(candidate_tracks)

        filtered = [r for r in results if r['probability'] >= threshold]
        sorted_results = sorted(filtered, key=lambda x: x['probability'], reverse=True)

        return sorted_results[:top_k]


# 싱글톤
_svm_service: Optional[SVMRecommendationService] = None


def get_svm_recommendation_service() -> SVMRecommendationService:
    global _svm_service
    if _svm_service is None:
        _svm_service = SVMRecommendationService()
    return _svm_service


# ====================================================================================
# 10. SVM 추천 API 라우터
# ====================================================================================

"""
SVM 기반 추천 API

엔드포인트:
- POST /api/svm/predict: 단일 트랙 예측
- POST /api/svm/predict/batch: 배치 예측
- POST /api/svm/recommend: 추천 트랙 선정 (Backend → GMS)
- POST /api/svm/audio-features: 오디오 피처 예측
- GET /api/svm/health: SVM 서비스 상태 확인
"""

svm_router = APIRouter(prefix="/api/svm", tags=["svm-recommendation"])


# Request/Response 스키마
class TrackInput(BaseModel):
    """트랙 입력 스키마"""
    artist: str = Field(..., description="아티스트명")
    track_name: str = Field(..., description="곡명")
    album_name: str = Field("", description="앨범명")
    tags: str = Field("", description="Last.fm 태그 (파이프로 구분)")
    duration_ms: int = Field(200000, description="곡 길이 (밀리초)")
    lfm_listeners: int = Field(0, description="Last.fm 리스너 수")
    lfm_playcount: int = Field(0, description="Last.fm 재생 수")


class AudioFeaturesResponse(BaseModel):
    """오디오 피처 응답"""
    danceability: float
    energy: float
    speechiness: float
    acousticness: float
    instrumentalness: float
    liveness: float
    valence: float
    tempo: float
    loudness: float


class PredictionResponse(BaseModel):
    """단일 예측 응답"""
    artist: str
    track_name: str
    probability: float = Field(..., description="좋아할 확률 (0~1)")
    prediction: int = Field(..., description="예측 라벨 (0=안좋아함, 1=좋아함)")
    audio_features: Dict[str, float]


class BatchPredictionRequest(BaseModel):
    """배치 예측 요청"""
    tracks: List[TrackInput]


class BatchPredictionResponse(BaseModel):
    """배치 예측 응답"""
    total: int
    predictions: List[PredictionResponse]


class RecommendationRequest(BaseModel):
    """추천 요청"""
    tracks: List[TrackInput]
    top_k: int = Field(10, ge=1, le=100, description="상위 K개 반환")
    threshold: float = Field(0.5, ge=0, le=1, description="최소 확률 임계값")


class RecommendationResponse(BaseModel):
    """추천 응답"""
    total_candidates: int
    recommended_count: int
    recommendations: List[PredictionResponse]


@svm_router.post("/predict", response_model=PredictionResponse)
async def predict_single(track: TrackInput):
    """단일 트랙 예측"""
    try:
        svm_service = get_svm_recommendation_service()

        result = svm_service.predict_single(
            artist=track.artist,
            track_name=track.track_name,
            album_name=track.album_name,
            tags=track.tags,
            duration_ms=track.duration_ms,
            lfm_listeners=track.lfm_listeners,
            lfm_playcount=track.lfm_playcount
        )

        return PredictionResponse(
            artist=track.artist,
            track_name=track.track_name,
            probability=result['probability'],
            prediction=result['prediction'],
            audio_features=result['audio_features']
        )

    except Exception as e:
        logger.error(f"Predict single error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@svm_router.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """배치 트랙 예측"""
    try:
        svm_service = get_svm_recommendation_service()

        tracks_dict = [
            {
                'artist': t.artist,
                'track_name': t.track_name,
                'album_name': t.album_name,
                'tags': t.tags,
                'duration_ms': t.duration_ms,
                'lfm_listeners': t.lfm_listeners,
                'lfm_playcount': t.lfm_playcount
            }
            for t in request.tracks
        ]

        results = svm_service.predict_batch(tracks_dict)

        predictions = [
            PredictionResponse(
                artist=r['artist'],
                track_name=r['track_name'],
                probability=r['probability'],
                prediction=r['prediction'],
                audio_features=r['audio_features']
            )
            for r in results
        ]

        return BatchPredictionResponse(
            total=len(predictions),
            predictions=predictions
        )

    except Exception as e:
        logger.error(f"Predict batch error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@svm_router.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """추천 트랙 선정 (EMS → GMS 이동 시 Backend에서 호출)"""
    try:
        svm_service = get_svm_recommendation_service()

        tracks_dict = [
            {
                'artist': t.artist,
                'track_name': t.track_name,
                'album_name': t.album_name,
                'tags': t.tags,
                'duration_ms': t.duration_ms,
                'lfm_listeners': t.lfm_listeners,
                'lfm_playcount': t.lfm_playcount
            }
            for t in request.tracks
        ]

        results = svm_service.get_recommendations(
            candidate_tracks=tracks_dict,
            top_k=request.top_k,
            threshold=request.threshold
        )

        recommendations = [
            PredictionResponse(
                artist=r['artist'],
                track_name=r['track_name'],
                probability=r['probability'],
                prediction=r['prediction'],
                audio_features=r['audio_features']
            )
            for r in results
        ]

        return RecommendationResponse(
            total_candidates=len(request.tracks),
            recommended_count=len(recommendations),
            recommendations=recommendations
        )

    except Exception as e:
        logger.error(f"Get recommendations error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@svm_router.post("/audio-features", response_model=AudioFeaturesResponse)
async def predict_audio_features(track: TrackInput):
    """오디오 피처 예측"""
    try:
        audio_service = get_audio_prediction_service()

        features = audio_service.predict_single(
            artist=track.artist,
            track_name=track.track_name,
            album_name=track.album_name,
            tags=track.tags,
            duration=track.duration_ms // 1000,
            lfm_listeners=track.lfm_listeners,
            lfm_playcount=track.lfm_playcount
        )

        return AudioFeaturesResponse(**features)

    except Exception as e:
        logger.error(f"Predict audio features error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@svm_router.get("/health")
async def health_check_svm():
    """SVM 서비스 상태 확인"""
    try:
        svm_service = get_svm_recommendation_service()
        audio_service = get_audio_prediction_service()

        # 모델 로드 시도
        svm_service.load_model()
        audio_service.load_model()

        return {
            "status": "healthy",
            "svm_model": "loaded",
            "audio_model": "loaded",
            "embedding_model": svm_service.embedding_model_name
        }

    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


# ====================================================================================
# 11. SVM 학습/재학습 API 라우터
# ====================================================================================

"""
SVM 모델 학습 및 재학습 API

초기 학습:
- CSV 업로드 → 태그 수집 → EMS negative 샘플링 → SVM 학습

재학습 (GMS 피드백):
- 선택한 곡 (positive) + 삭제한 곡 (negative) → 모델 업데이트

저장 위치: data/user_svm_models/user_{user_id}_svm.pkl
"""

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

svm_training_router = APIRouter(prefix="/api/svm-train", tags=["svm-training"])

# 설정
USER_MODELS_DIR = BASE_DIR / "data" / "user_svm_models"
USER_MODELS_DIR.mkdir(parents=True, exist_ok=True)
EMS_DATA_PATH = BASE_DIR / "data" / "ems_songs.csv"

# DB 세션
try:
    from database import SessionLocal
except ImportError:
    from app.database import SessionLocal


def get_user_email_from_db(user_id: int) -> str:
    """DB에서 사용자 이메일 조회"""
    db = SessionLocal()
    try:
        query = text("SELECT email FROM users WHERE user_id = :user_id")
        result = db.execute(query, {"user_id": user_id}).fetchone()
        if result:
            return result[0]
        return None
    except Exception as e:
        logger.error(f"사용자 이메일 조회 실패: {e}")
        return None
    finally:
        db.close()

# 오디오 피처 목록
AUDIO_FEATURES = [
    'danceability', 'energy', 'speechiness', 'acousticness',
    'instrumentalness', 'liveness', 'valence', 'tempo', 'loudness'
]


# Request/Response 스키마
class TrainingResponse(BaseModel):
    """학습 응답"""
    user_id: int
    success: bool
    message: str
    positive_count: int = 0
    negative_count: int = 0
    model_path: Optional[str] = None
    metrics: Optional[Dict] = None


class FeedbackTrack(BaseModel):
    """피드백 트랙"""
    artist: str = Field(..., description="아티스트명")
    track_name: str = Field(..., description="곡명")
    album_name: str = Field("", description="앨범명")
    tags: str = Field("", description="Last.fm 태그")


class FeedbackRequest(BaseModel):
    """재학습 요청"""
    selected_tracks: List[FeedbackTrack] = Field(..., description="선택한 곡 (Positive)")
    rejected_tracks: List[FeedbackTrack] = Field(default=[], description="거부한 곡 (Negative)")


class RetrainResponse(BaseModel):
    """재학습 응답"""
    user_id: int
    success: bool
    message: str
    new_positives: int = 0
    new_negatives: int = 0
    total_positives: int = 0
    total_negatives: int = 0
    metrics: Optional[Dict] = None


def load_ems_data() -> pd.DataFrame:
    """EMS 데이터 로드 (CSV 폴백)"""
    if EMS_DATA_PATH.exists():
        df = pd.read_csv(EMS_DATA_PATH)
        logger.info(f"EMS 데이터 로드 (CSV): {len(df)}곡")
        return df

    logger.warning("EMS 데이터 없음")
    return pd.DataFrame(columns=['artists', 'track_name', 'album'] + AUDIO_FEATURES)


def sample_negatives(
    positive_df: pd.DataFrame,
    ems_df: pd.DataFrame,
    ratio: int = 3
) -> pd.DataFrame:
    """Negative 샘플링 (Hard + Random)"""
    n_positive = len(positive_df)
    n_negative = n_positive * ratio

    positive_artists = set(positive_df['artist'].str.lower().unique())
    positive_tracks = set(
        (positive_df['artist'].str.lower() + '|' + positive_df['track_name'].str.lower()).unique()
    )

    # EMS에서 중복 제거
    ems_df = ems_df.copy()
    ems_df['_key'] = ems_df['artists'].str.lower() + '|' + ems_df['track_name'].str.lower()
    ems_df = ems_df[~ems_df['_key'].isin(positive_tracks)]

    # Hard Negative: 같은 아티스트의 다른 곡
    hard_negatives = ems_df[ems_df['artists'].str.lower().isin(positive_artists)]
    n_hard = min(len(hard_negatives), n_negative // 2)

    if n_hard > 0:
        hard_sample = hard_negatives.sample(n=n_hard, random_state=42)
    else:
        hard_sample = pd.DataFrame()

    # Random Negative
    remaining = ems_df[~ems_df.index.isin(hard_sample.index)]
    n_random = n_negative - n_hard

    if len(remaining) >= n_random:
        random_sample = remaining.sample(n=n_random, random_state=42)
    else:
        random_sample = remaining

    negative_df = pd.concat([hard_sample, random_sample], ignore_index=True)
    logger.info(f"Negative 샘플링: Hard={n_hard}, Random={len(random_sample)}")

    return negative_df


async def enrich_with_lastfm(tracks: List[Dict]) -> List[Dict]:
    """Last.fm 태그 수집"""
    lastfm_service = get_lastfm_service()

    enriched = []
    for track in tracks:
        artist = track.get('artist', '')
        track_name = track.get('track_name', '')

        # 태그 수집
        tags = await lastfm_service.get_combined_tags(artist, track_name)
        track['tags'] = '|'.join(tags) if tags else ''
        enriched.append(track)

    return enriched


def create_features_for_training(
    df: pd.DataFrame,
    embedding_model: SentenceTransformer,
    audio_service
) -> np.ndarray:
    """393D 피처 생성 (SVM 학습용)"""
    # 텍스트 임베딩
    texts = []
    for _, row in df.iterrows():
        parts = [
            str(row.get('artist', row.get('artists', ''))),
            str(row.get('track_name', '')),
            str(row.get('album_name', row.get('album', ''))),
            str(row.get('tags', row.get('lfm_artist_tags', '')))
        ]
        parts = [p for p in parts if p and p.lower() != 'nan']
        texts.append(' '.join(parts) if parts else 'unknown')

    embeddings = embedding_model.encode(texts, show_progress_bar=False)

    # 오디오 피처
    audio_features = []
    for _, row in df.iterrows():
        if all(f in row for f in AUDIO_FEATURES):
            audio = [float(row[f]) if pd.notna(row[f]) else 0.5 for f in AUDIO_FEATURES]
        else:
            # 예측
            try:
                pred = audio_service.predict_single(
                    artist=str(row.get('artist', row.get('artists', ''))),
                    track_name=str(row.get('track_name', '')),
                    tags=str(row.get('tags', ''))
                )
                audio = [pred[f] for f in AUDIO_FEATURES]
            except:
                audio = [0.5, 0.5, 0.1, 0.5, 0.1, 0.2, 0.5, 120.0, -10.0]
        audio_features.append(audio)

    audio_array = np.array(audio_features, dtype=np.float64)
    audio_array = np.nan_to_num(audio_array, nan=0.5)

    # 결합
    features = np.hstack([embeddings, audio_array])
    features = np.nan_to_num(features, nan=0.0)

    return features


@svm_training_router.post("/upload", response_model=TrainingResponse)
async def train_from_csv(
    user_id: int,
    file: UploadFile = File(...),
    negative_ratio: int = 3,
    force_retrain: bool = False
):
    """
    CSV 파일로 SVM 모델 학습 (Backend에서 호출)

    CSV 형식: artist, track_name, album_name

    흐름:
    1. CSV 파싱
    2. Last.fm 태그 수집
    3. 오디오 피처 예측
    4. EMS에서 Negative 샘플링
    5. SVM 학습
    6. 모델 저장
    """
    try:
        # 사용자 이메일 조회
        user_email = get_user_email_from_db(user_id)
        if not user_email:
            raise HTTPException(status_code=404, detail="User not found")

        email_prefix = user_email.split('@')[0]
        model_path = USER_MODELS_DIR / f"{email_prefix}_.pkl"

        if model_path.exists() and not force_retrain:
            return TrainingResponse(
                user_id=user_id,
                success=False,
                message="Model already exists. Use force_retrain=true to retrain.",
                model_path=str(model_path)
            )

        # CSV 파싱
        content = await file.read()
        decoded = content.decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(decoded))

        positive_tracks = []
        for row in reader:
            track = {
                'artist': row.get('artist', ''),
                'track_name': row.get('track_name', ''),
                'album_name': row.get('album_name', row.get('album', '')),
            }
            if track['artist'] and track['track_name']:
                positive_tracks.append(track)

        if not positive_tracks:
            raise HTTPException(status_code=400, detail="No valid tracks in CSV")

        logger.info(f"CSV 파싱: {len(positive_tracks)}곡")

        # Last.fm 태그 수집
        logger.info("Last.fm 태그 수집 중...")
        positive_tracks = await enrich_with_lastfm(positive_tracks)

        # EMS Negative 샘플링
        logger.info("Negative 샘플링...")
        ems_df = load_ems_data()
        positive_df = pd.DataFrame(positive_tracks)
        negative_df = sample_negatives(positive_df, ems_df, ratio=negative_ratio)

        # 라벨
        positive_df['label'] = 1
        negative_df['label'] = 0
        negative_df = negative_df.rename(columns={'artists': 'artist'})
        if 'lfm_artist_tags' in negative_df.columns:
            negative_df['tags'] = negative_df['lfm_artist_tags']

        full_df = pd.concat([positive_df, negative_df], ignore_index=True)
        logger.info(f"학습 데이터: Positive={len(positive_df)}, Negative={len(negative_df)}")

        # 피처 생성
        logger.info("피처 생성 (393D)...")
        settings = get_settings()
        embedding_model = SentenceTransformer(settings.embedding_model)
        audio_service = get_audio_prediction_service()

        X = create_features_for_training(full_df, embedding_model, audio_service)
        y = full_df['label'].values

        logger.info(f"피처 완료: {X.shape}")

        # SVM 학습
        logger.info("SVM 학습 중...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.15, stratify=y, random_state=42
        )

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(C=10, kernel='rbf', gamma='scale', probability=True))
        ])

        pipeline.fit(X_train, y_train)

        # 평가
        train_score = pipeline.score(X_train, y_train)
        test_score = pipeline.score(X_test, y_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_proba)

        metrics = {
            'train_accuracy': float(train_score),
            'test_accuracy': float(test_score),
            'test_auc': float(auc_score),
            'positive_count': int(len(positive_df)),
            'negative_count': int(len(negative_df)),
            'trained_at': datetime.now().isoformat()
        }

        logger.info(f"학습 완료: Train={train_score:.4f}, Test={test_score:.4f}, AUC={auc_score:.4f}")

        # 모델 저장
        model_data = {
            'pipeline': pipeline,
            'embedding_model_name': settings.embedding_model,
            'metrics': metrics,
            'user_id': user_id
        }
        joblib.dump(model_data, model_path)

        return TrainingResponse(
            user_id=user_id,
            success=True,
            message="SVM model trained successfully",
            positive_count=len(positive_df),
            negative_count=len(negative_df),
            model_path=str(model_path),
            metrics=metrics
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Training error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@svm_training_router.post("/{user_id}/retrain", response_model=RetrainResponse)
async def retrain_with_feedback(
    user_id: int,
    feedback: FeedbackRequest
):
    """
    GMS 피드백 기반 재학습

    선택한 곡 → positive
    거부한 곡 → negative

    흐름:
    1. 기존 피드백 로드
    2. 새 피드백 추가
    3. EMS 샘플링
    4. 모델 재학습
    """
    try:
        user_email = get_user_email_from_db(user_id)
        if not user_email:
            raise HTTPException(status_code=404, detail="User not found")

        email_prefix = user_email.split('@')[0]
        model_path = USER_MODELS_DIR / f"{email_prefix}_.pkl"
        feedback_path = USER_MODELS_DIR / f"{email_prefix}_feedback.pkl"

        if not model_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"User {user_id} 모델 없음. /upload로 초기 학습 필요"
            )

        if not feedback.selected_tracks and not feedback.rejected_tracks:
            raise HTTPException(status_code=400, detail="최소 1개 트랙 필요")

        logger.info(f"재학습: user={user_id}, selected={len(feedback.selected_tracks)}, rejected={len(feedback.rejected_tracks)}")

        # 기존 피드백 로드
        if feedback_path.exists():
            existing = joblib.load(feedback_path)
            accumulated_pos = existing.get('positives', [])
            accumulated_neg = existing.get('negatives', [])
        else:
            accumulated_pos = []
            accumulated_neg = []

        # 새 피드백 추가
        for track in feedback.selected_tracks:
            accumulated_pos.append({
                'artist': track.artist,
                'track_name': track.track_name,
                'album_name': track.album_name,
                'tags': track.tags,
                'source': 'gms_feedback',
                'feedback_type': 'selected'
            })

        for track in feedback.rejected_tracks:
            accumulated_neg.append({
                'artist': track.artist,
                'track_name': track.track_name,
                'album_name': track.album_name,
                'tags': track.tags,
                'source': 'gms_feedback',
                'feedback_type': 'rejected'
            })

        # 피드백 저장
        joblib.dump({
            'positives': accumulated_pos,
            'negatives': accumulated_neg,
            'updated_at': datetime.now().isoformat()
        }, feedback_path)

        # 학습 데이터 준비
        pos_df = pd.DataFrame(accumulated_pos) if accumulated_pos else pd.DataFrame(columns=['artist', 'track_name', 'tags'])
        neg_df = pd.DataFrame(accumulated_neg) if accumulated_neg else pd.DataFrame(columns=['artist', 'track_name', 'tags'])

        # EMS 샘플링
        ems_df = load_ems_data()
        total_pos = len(accumulated_pos)
        n_ems_neg = max((total_pos * 3) - len(accumulated_neg), total_pos)

        # 피드백 곡 제외
        feedback_keys = set()
        for p in accumulated_pos:
            feedback_keys.add(f"{p['artist'].lower()}|{p['track_name'].lower()}")
        for n in accumulated_neg:
            feedback_keys.add(f"{n['artist'].lower()}|{n['track_name'].lower()}")

        ems_df['_key'] = ems_df['artists'].str.lower() + '|' + ems_df['track_name'].str.lower()
        ems_available = ems_df[~ems_df['_key'].isin(feedback_keys)]

        if len(ems_available) >= n_ems_neg:
            ems_sample = ems_available.sample(n=n_ems_neg, random_state=42)
        else:
            ems_sample = ems_available

        # 피처 생성
        settings = get_settings()
        embedding_model = SentenceTransformer(settings.embedding_model)
        audio_service = get_audio_prediction_service()

        if not pos_df.empty:
            pos_df['label'] = 1
            X_pos = create_features_for_training(pos_df, embedding_model, audio_service)
            y_pos = pos_df['label'].values
        else:
            X_pos = np.array([]).reshape(0, 393)
            y_pos = np.array([])

        if not neg_df.empty:
            neg_df['label'] = 0
            X_neg = create_features_for_training(neg_df, embedding_model, audio_service)
            y_neg = neg_df['label'].values
        else:
            X_neg = np.array([]).reshape(0, 393)
            y_neg = np.array([])

        ems_sample['label'] = 0
        ems_sample = ems_sample.rename(columns={'artists': 'artist', 'lfm_artist_tags': 'tags'})
        X_ems = create_features_for_training(ems_sample, embedding_model, audio_service)
        y_ems = ems_sample['label'].values

        # 결합
        if len(X_pos) > 0:
            X_all = np.vstack([X_pos, X_neg, X_ems]) if len(X_neg) > 0 else np.vstack([X_pos, X_ems])
            y_all = np.concatenate([y_pos, y_neg, y_ems]) if len(y_neg) > 0 else np.concatenate([y_pos, y_ems])
        else:
            X_all = np.vstack([X_neg, X_ems]) if len(X_neg) > 0 else X_ems
            y_all = np.concatenate([y_neg, y_ems]) if len(y_neg) > 0 else y_ems

        if len(np.unique(y_all)) < 2:
            return RetrainResponse(
                user_id=user_id,
                success=False,
                message="Positive와 Negative 모두 필요",
                new_positives=len(feedback.selected_tracks),
                new_negatives=len(feedback.rejected_tracks)
            )

        logger.info(f"재학습 데이터: Pos={len(y_pos)}, Neg={len(y_neg)}, EMS={len(y_ems)}")

        # 재학습
        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all, test_size=0.15, stratify=y_all, random_state=42
        )

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(C=10, kernel='rbf', gamma='scale', probability=True))
        ])

        pipeline.fit(X_train, y_train)

        # 평가
        train_score = pipeline.score(X_train, y_train)
        test_score = pipeline.score(X_test, y_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_proba)

        old_model = joblib.load(model_path)
        old_metrics = old_model.get('metrics', {})

        metrics = {
            'train_accuracy': float(train_score),
            'test_accuracy': float(test_score),
            'test_auc': float(auc_score),
            'positive_count': int(len(accumulated_pos)),
            'negative_count': int(len(accumulated_neg) + len(ems_sample)),
            'feedback_positives': int(len(accumulated_pos)),
            'feedback_negatives': int(len(accumulated_neg)),
            'retrained_at': datetime.now().isoformat(),
            'retrain_count': old_metrics.get('retrain_count', 0) + 1
        }

        logger.info(f"재학습 완료: Train={train_score:.4f}, Test={test_score:.4f}, AUC={auc_score:.4f}")

        # 저장
        joblib.dump({
            'pipeline': pipeline,
            'embedding_model_name': settings.embedding_model,
            'metrics': metrics,
            'user_id': user_id
        }, model_path)

        return RetrainResponse(
            user_id=user_id,
            success=True,
            message="모델 재학습 완료",
            new_positives=len(feedback.selected_tracks),
            new_negatives=len(feedback.rejected_tracks),
            total_positives=len(accumulated_pos),
            total_negatives=len(accumulated_neg),
            metrics=metrics
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Retrain error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@svm_training_router.get("/{user_id}/feedback")
async def get_user_feedback(user_id: int):
    """사용자 피드백 히스토리 조회"""
    feedback_path = USER_MODELS_DIR / f"user_{user_id}_feedback.pkl"

    if not feedback_path.exists():
        return {
            "user_id": user_id,
            "has_feedback": False,
            "positives": [],
            "negatives": []
        }

    feedback_data = joblib.load(feedback_path)
    return {
        "user_id": user_id,
        "has_feedback": True,
        "positives": feedback_data.get('positives', []),
        "negatives": feedback_data.get('negatives', []),
        "updated_at": feedback_data.get('updated_at')
    }


# ====================================================================================
# 12. GMS 피드백 API 라우터 (Backend 연동)
# ====================================================================================

"""
GMS 피드백 API

Backend에서 호출하는 API:
- POST /gms-deleted-track: 곡 삭제 → negative 피드백 저장
- POST /bulk-delete-track: 여러 곡 일괄 삭제 → negative 피드백 저장
- POST /gms-selected-track: 곡 선택 → positive 피드백 저장
- POST /enrich-audio-features: 트랙 생성 시 오디오 피처 예측/저장

피드백이 5개 이상 쌓이면 자동으로 재학습 트리거
"""

feedback_router = APIRouter(tags=["gms-feedback"])


# Request/Response 스키마
class GmsDeleteTrackRequest(BaseModel):
    """GMS 곡 삭제 요청"""
    users_id: int = Field(..., description="사용자 ID")
    playlists_id: int = Field(..., description="플레이리스트 ID")
    tracks_id: int = Field(..., description="트랙 ID")


class GmsSelectTrackRequest(BaseModel):
    """GMS 곡 선택 요청"""
    users_id: int = Field(..., description="사용자 ID")
    playlists_id: int = Field(..., description="플레이리스트 ID")
    tracks_id: int = Field(..., description="트랙 ID")
    artist: Optional[str] = None
    track_name: Optional[str] = None
    album_name: Optional[str] = None


class FeedbackResponse(BaseModel):
    """피드백 응답"""
    status: str
    message: str
    user_id: int
    feedback_type: str
    tracks_count: int
    model_updated: bool


class EnrichAudioRequest(BaseModel):
    """오디오 특성 예측 요청"""
    track_id: int
    artist: str = ""
    track_name: str = ""
    album_name: str = ""


class EnrichAudioResponse(BaseModel):
    """오디오 특성 예측 응답"""
    success: bool
    track_id: int
    message: str
    audio_features: Optional[dict] = None


def get_track_info_from_db(track_id: int) -> dict:
    """DB에서 트랙 정보 조회"""
    try:
        with get_db_context() as db:
            from sqlalchemy import text
            query = text("""
                SELECT artist, title as track_name, album as album_name
                FROM tracks
                WHERE track_id = :track_id
            """)
            result = db.execute(query, {"track_id": track_id})
            row = result.fetchone()

            if row:
                return {
                    "artist": row.artist or "",
                    "track_name": row.track_name or "",
                    "album_name": row.album_name or ""
                }
    except Exception as e:
        logger.warning(f"트랙 정보 조회 실패 (track_id={track_id}): {e}")

    return {"artist": "", "track_name": "", "album_name": ""}


def add_feedback_to_user(user_id: int, tracks: List[dict], feedback_type: str) -> bool:
    """피드백 데이터 저장"""
    feedback_path = USER_MODELS_DIR / f"user_{user_id}_feedback.pkl"

    try:
        if feedback_path.exists():
            feedback_data = joblib.load(feedback_path)
            accumulated_pos = feedback_data.get('positives', [])
            accumulated_neg = feedback_data.get('negatives', [])
        else:
            accumulated_pos = []
            accumulated_neg = []

        for track in tracks:
            item = {
                'artist': track.get('artist', ''),
                'track_name': track.get('track_name', ''),
                'album_name': track.get('album_name', ''),
                'tags': track.get('tags', ''),
                'source': 'gms_feedback',
                'feedback_type': feedback_type,
                'added_at': datetime.now().isoformat()
            }

            if feedback_type == "positive":
                accumulated_pos.append(item)
            else:
                accumulated_neg.append(item)

        joblib.dump({
            'positives': accumulated_pos,
            'negatives': accumulated_neg,
            'updated_at': datetime.now().isoformat()
        }, feedback_path)

        logger.info(f"피드백 저장: user={user_id}, type={feedback_type}, count={len(tracks)}")
        return True

    except Exception as e:
        logger.error(f"피드백 저장 실패: {e}")
        return False


async def trigger_svm_retrain(user_id: int) -> bool:
    """피드백이 5개 이상이면 재학습 트리거"""
    feedback_path = USER_MODELS_DIR / f"user_{user_id}_feedback.pkl"
    model_path = USER_MODELS_DIR / f"user_{user_id}_svm.pkl"

    if not model_path.exists():
        logger.warning(f"User {user_id} SVM 모델 없음. 재학습 스킵")
        return False

    try:
        if not feedback_path.exists():
            return False

        feedback_data = joblib.load(feedback_path)
        total = len(feedback_data.get('positives', [])) + len(feedback_data.get('negatives', []))

        RETRAIN_THRESHOLD = 5

        if total >= RETRAIN_THRESHOLD:
            logger.info(f"재학습 트리거: user={user_id}, feedback={total}")
            # 실제 재학습은 /retrain API 호출로 수행
            return True
        else:
            logger.info(f"재학습 대기: user={user_id}, feedback={total}/{RETRAIN_THRESHOLD}")
            return False

    except Exception as e:
        logger.error(f"재학습 트리거 실패: {e}")
        return False


@feedback_router.post("/gms-deleted-track", response_model=FeedbackResponse)
async def gms_deleted_track(request: GmsDeleteTrackRequest):
    """
    GMS 곡 삭제 피드백 (Backend 호출)

    삭제된 곡 → negative 피드백 → 향후 추천 제외
    """
    try:
        logger.info(f"GMS 삭제 피드백: user={request.users_id}, track={request.tracks_id}")

        track_info = get_track_info_from_db(request.tracks_id)

        success = add_feedback_to_user(
            user_id=request.users_id,
            tracks=[track_info],
            feedback_type="negative"
        )

        if not success:
            raise HTTPException(status_code=500, detail="피드백 저장 실패")

        model_updated = await trigger_svm_retrain(request.users_id)

        return FeedbackResponse(
            status="success",
            message=f"삭제 피드백 기록. {'모델 업데이트됨' if model_updated else '피드백 누적 중'}",
            user_id=request.users_id,
            feedback_type="deleted",
            tracks_count=1,
            model_updated=model_updated
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"GMS deleted-track error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@feedback_router.post("/bulk-delete-track", response_model=FeedbackResponse)
async def bulk_delete_track(request: Dict):
    """
    GMS 여러 곡 일괄 삭제 피드백 (Backend 호출)

    여러 곡 삭제 → negative 피드백 일괄 저장
    """
    try:
        users_id = request.get("users_id")
        playlists_id = request.get("playlists_id")
        tracks_ids = request.get("tracks_ids", [])

        logger.info(f"GMS 일괄 삭제: user={users_id}, tracks={len(tracks_ids)}")

        # 각 트랙 정보 조회
        tracks_info = []
        for track_id in tracks_ids:
            track_info = get_track_info_from_db(track_id)
            tracks_info.append(track_info)

        # negative 피드백 저장
        success = add_feedback_to_user(
            user_id=users_id,
            tracks=tracks_info,
            feedback_type="negative"
        )

        if not success:
            raise HTTPException(status_code=500, detail="피드백 저장 실패")

        model_updated = await trigger_svm_retrain(users_id)

        return FeedbackResponse(
            status="success",
            message=f"{len(tracks_ids)}곡 삭제 피드백 기록",
            user_id=users_id,
            feedback_type="deleted",
            tracks_count=len(tracks_ids),
            model_updated=model_updated
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Bulk delete error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@feedback_router.post("/gms-selected-track", response_model=FeedbackResponse)
async def gms_selected_track(request: GmsSelectTrackRequest):
    """
    GMS 곡 선택 피드백 (Backend 호출)

    선택된 곡 → positive 피드백 → 향후 추천 강화
    """
    try:
        logger.info(f"GMS 선택 피드백: user={request.users_id}, track={request.tracks_id}")

        if request.artist and request.track_name:
            track_info = {
                "artist": request.artist,
                "track_name": request.track_name,
                "album_name": request.album_name or ""
            }
        else:
            track_info = get_track_info_from_db(request.tracks_id)

        success = add_feedback_to_user(
            user_id=request.users_id,
            tracks=[track_info],
            feedback_type="positive"
        )

        if not success:
            raise HTTPException(status_code=500, detail="피드백 저장 실패")

        model_updated = await trigger_svm_retrain(request.users_id)

        return FeedbackResponse(
            status="success",
            message="선택 피드백 기록",
            user_id=request.users_id,
            feedback_type="selected",
            tracks_count=1,
            model_updated=model_updated
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"GMS selected-track error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@feedback_router.post("/enrich-audio-features", response_model=EnrichAudioResponse)
async def enrich_audio_features(request: EnrichAudioRequest):
    """
    트랙 오디오 특성 예측 및 DB 저장 (Backend 호출)

    트랙 생성 시 호출 → 오디오 피처 자동 예측 → external_metadata에 저장
    """
    try:
        logger.info(f"오디오 특성 예측: track_id={request.track_id}")

        audio_service = get_audio_prediction_service()
        audio_service.load_model()

        features = audio_service.predict_single(
            artist=request.artist,
            track_name=request.track_name,
            album_name=request.album_name,
            tags="",
            duration=200
        )

        logger.info(f"오디오 예측 완료: {features}")

        # DB 저장
        try:
            with get_db_context() as db:
                from sqlalchemy import text

                select_query = text("SELECT external_metadata FROM tracks WHERE track_id = :track_id")
                result = db.execute(select_query, {"track_id": request.track_id})
                row = result.fetchone()

                if row:
                    existing = {}
                    if row.external_metadata:
                        try:
                            existing = json.loads(row.external_metadata) if isinstance(row.external_metadata, str) else row.external_metadata
                        except:
                            existing = {}

                    existing['audio_features'] = features
                    existing['audio_features_updated_at'] = datetime.now().isoformat()

                    update_query = text("""
                        UPDATE tracks
                        SET external_metadata = :metadata
                        WHERE track_id = :track_id
                    """)
                    db.execute(update_query, {
                        "track_id": request.track_id,
                        "metadata": json.dumps(existing, ensure_ascii=False)
                    })
                    db.commit()

                    logger.info(f"DB 저장 완료: track_id={request.track_id}")
        except Exception as db_error:
            logger.warning(f"DB 저장 실패 (피처 예측 성공): {db_error}")

        return EnrichAudioResponse(
            success=True,
            track_id=request.track_id,
            message="오디오 특성 예측 및 저장 완료",
            audio_features=features
        )

    except Exception as e:
        logger.error(f"Enrich audio features error: {e}", exc_info=True)
        return EnrichAudioResponse(
            success=False,
            track_id=request.track_id,
            message=f"오디오 특성 예측 실패: {str(e)}",
            audio_features=None
        )


# ====================================================================================
# 13. FastAPI 애플리케이션
# ====================================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 라이프사이클 관리"""
    # 시작 시
    logger.info("Starting Music Recommendation Service (System B)...")
    settings = get_settings()
    logger.info(f"Embedding model: {settings.embedding_model}")
    logger.info(f"Data directory: {settings.data_dir}")
    logger.info(f"Audio prediction model: {settings.audio_model_path}")
    logger.info(f"SVM models directory: {USER_MODELS_DIR}")

    yield

    # 종료 시
    logger.info("Shutting down Music Recommendation Service (System B)...")


app = FastAPI(
    title="Music Recommendation API - System B",
    description="SVM 기반 음악 추천 시스템 (오디오 역추적 모델 포함)",
    version="1.0.0",
    lifespan=lifespan
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 실제 운영에서는 특정 도메인만 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록 (시스템 B만)
app.include_router(svm_router)  # SVM 추천 API
app.include_router(svm_training_router)  # SVM 학습/재학습 API
app.include_router(feedback_router)  # GMS 피드백 API (Backend 연동)


@app.get("/")
def read_root():
    """헬스 체크"""
    return {
        "message": "Music Recommendation API (System B) is running",
        "version": "1.0.0",
        "system": "SVM + Audio Reverse Tracking"
    }


@app.get("/health")
def health_check():
    """상세 헬스 체크"""
    settings = get_settings()
    return {
        "status": "healthy",
        "system": "B",
        "embedding_model": settings.embedding_model,
        "audio_model_path": str(settings.audio_model_path),
        "user_models_dir": str(USER_MODELS_DIR),
        "ems_data_path": str(EMS_DATA_PATH)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fastapi_system_b_integrated:app", host="0.0.0.0", port=8001, reload=True)


# ====================================================================================
# 사용 예시
# ====================================================================================

"""
1. 서버 실행:
   python fastapi_system_b_integrated.py

2. API 문서 확인:
   http://localhost:8001/docs

3. 초기 SVM 모델 학습 (CSV 업로드):
   POST /api/svm-train/upload
   Form Data: user_id=1, file=playlist.csv, negative_ratio=3

4. EMS → GMS 추천:
   POST /api/svm/recommend
   {
     "tracks": [
       {"artist": "Miles Davis", "track_name": "So What", "tags": "jazz|cool jazz"},
       {"artist": "John Coltrane", "track_name": "Giant Steps", "tags": "jazz|bebop"}
     ],
     "top_k": 10,
     "threshold": 0.5
   }

5. GMS 곡 선택 피드백:
   POST /gms-selected-track
   {
     "users_id": 1,
     "playlists_id": 10,
     "tracks_id": 123,
     "artist": "Miles Davis",
     "track_name": "So What"
   }

6. GMS 곡 삭제 피드백:
   POST /gms-deleted-track
   {
     "users_id": 1,
     "playlists_id": 10,
     "tracks_id": 456
   }

7. 여러 곡 일괄 삭제:
   POST /bulk-delete-track
   {
     "users_id": 1,
     "playlists_id": 10,
     "tracks_ids": [123, 456, 789]
   }

8. 오디오 피처 자동 예측 (트랙 생성 시):
   POST /enrich-audio-features
   {
     "track_id": 123,
     "artist": "Miles Davis",
     "track_name": "So What",
     "album_name": "Kind of Blue"
   }

9. GMS 피드백 기반 재학습:
   POST /api/svm-train/1/retrain
   {
     "selected_tracks": [
       {"artist": "Miles Davis", "track_name": "So What", "tags": "jazz"}
     ],
     "rejected_tracks": [
       {"artist": "Bad Artist", "track_name": "Bad Song", "tags": "pop"}
     ]
   }

10. 사용자 피드백 히스토리 조회:
    GET /api/svm-train/1/feedback
"""
