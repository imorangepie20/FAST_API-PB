"""
벡터 기반 자연어 검색 서비스 (ChromaDB)

114k 곡의 아티스트 태그를 벡터화하여 의미 기반 검색을 수행합니다.
"""
import os
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any
import pandas as pd

try:
    import chromadb
    from chromadb.config import Settings
    from chromadb.utils import embedding_functions
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    chromadb = None
    embedding_functions = None

# SentenceTransformer는 chromadb.utils.embedding_functions를 통해 사용

logger = logging.getLogger(__name__)

# 벡터 DB 경로 (이 파일 기준 절대 경로: LLM/L2/app/services/llm/ -> LLM/L2/data/)
CHROMA_DB_PATH = Path(__file__).parent.parent.parent.parent / "data" / "chroma_db"
COLLECTION_NAME = "music_tags"

# 임베딩 모델 - 영어 전용 (빠르고 정확)
# 한국어 쿼리는 LLM이 영어로 번역 후 검색
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)


class VectorSearchService:
    """ChromaDB 기반 벡터 검색 서비스"""

    def __init__(self):
        self.client = None
        self.collection = None
        self.embedding_model = None
        self._initialized = False

        if not CHROMADB_AVAILABLE:
            logger.warning("ChromaDB가 설치되지 않았습니다. pip install chromadb")
            return

        self._initialize()

    def _initialize(self):
        """ChromaDB 클라이언트 및 컬렉션 초기화"""
        try:
            # 디렉토리 생성
            CHROMA_DB_PATH.mkdir(parents=True, exist_ok=True)

            # ChromaDB 클라이언트 생성 (영구 저장)
            self.client = chromadb.PersistentClient(
                path=str(CHROMA_DB_PATH),
                settings=Settings(anonymized_telemetry=False)
            )

            # 다국어 임베딩 함수 생성
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=EMBEDDING_MODEL
            )

            # 컬렉션 가져오기 또는 생성 (다국어 임베딩 함수 사용)
            self.collection = self.client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"description": "Music tracks with artist tags for semantic search"},
                embedding_function=self.embedding_function
            )

            # 임베딩 모델 참조 저장 (호환성)
            self.embedding_model = self.embedding_function

            self._initialized = True
            logger.info(f"VectorSearchService 초기화 완료. 컬렉션 크기: {self.collection.count()}")

        except Exception as e:
            logger.error(f"VectorSearchService 초기화 실패: {e}")
            self._initialized = False

    @property
    def is_ready(self) -> bool:
        """서비스 준비 상태"""
        return self._initialized and self.collection is not None

    @property
    def collection_count(self) -> int:
        """컬렉션 내 문서 수"""
        if not self.is_ready:
            return 0
        return self.collection.count()

    def _embed_text(self, text: str) -> List[float]:
        """텍스트를 벡터로 변환"""
        if not self.embedding_model:
            return []
        return self.embedding_model.encode(text).tolist()

    def add_track(
        self,
        track_id: str,
        artist: str,
        title: str,
        tags: str,
        genre: Optional[str] = None,
        audio_features: Optional[Dict] = None
    ) -> bool:
        """
        트랙을 벡터 DB에 추가

        Args:
            track_id: 고유 ID (문자열)
            artist: 아티스트명
            title: 곡 제목
            tags: 아티스트 태그 (쉼표 또는 파이프 구분)
            genre: 장르 (옵션)
            audio_features: 오디오 특성 (옵션)

        Returns:
            성공 여부
        """
        if not self.is_ready:
            logger.warning("VectorSearchService가 준비되지 않았습니다.")
            return False

        try:
            # 태그 정규화
            if tags:
                tags_normalized = tags.replace("|", ",").replace("  ", " ").strip()
            else:
                tags_normalized = ""

            # 검색용 텍스트 생성 (태그 + 장르)
            search_text = tags_normalized
            if genre:
                search_text = f"{genre}, {search_text}" if search_text else genre

            if not search_text:
                logger.debug(f"태그 없음, 건너뜀: {artist} - {title}")
                return False

            # 메타데이터 구성
            metadata = {
                "artist": artist,
                "title": title,
                "tags": tags_normalized,
                "genre": genre or "",
            }

            # 오디오 특성 추가 (있으면)
            if audio_features:
                for key in ["energy", "valence", "acousticness", "danceability", "tempo"]:
                    if key in audio_features:
                        metadata[key] = float(audio_features[key])

            # 벡터 DB에 추가 (중복 시 업데이트)
            self.collection.upsert(
                ids=[str(track_id)],
                documents=[search_text],
                metadatas=[metadata]
            )

            return True

        except Exception as e:
            logger.error(f"트랙 추가 실패 ({artist} - {title}): {e}")
            return False

    def search(
        self,
        query: str,
        n_results: int = 10,
        filters: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        자연어 쿼리로 유사한 곡 검색

        Args:
            query: 검색 쿼리 (예: "비오는 날 잔잔한 재즈")
            n_results: 반환할 결과 수
            filters: 추가 필터 조건 (예: {"genre": "jazz"})

        Returns:
            유사한 곡 목록
        """
        if not self.is_ready:
            logger.warning("VectorSearchService가 준비되지 않았습니다.")
            return []

        if self.collection_count == 0:
            logger.warning("벡터 DB가 비어있습니다. 먼저 초기화가 필요합니다.")
            return []

        try:
            # 쿼리 필터 구성
            where_filter = None
            if filters:
                # ChromaDB where 필터 형식으로 변환
                conditions = []
                for key, value in filters.items():
                    if key == "genres" and isinstance(value, list):
                        # 장르 목록 중 하나라도 포함되면 매칭
                        # ChromaDB는 $contains 연산자 사용
                        if value:
                            conditions.append({"genre": {"$in": value}})
                    elif key.endswith("_min"):
                        field = key.replace("_min", "")
                        conditions.append({field: {"$gte": value}})
                    elif key.endswith("_max"):
                        field = key.replace("_max", "")
                        conditions.append({field: {"$lte": value}})

                if conditions:
                    if len(conditions) == 1:
                        where_filter = conditions[0]
                    else:
                        where_filter = {"$and": conditions}

            # 벡터 검색 수행
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where_filter,
                include=["documents", "metadatas", "distances"]
            )

            # 결과 포맷팅
            tracks = []
            if results and results["ids"] and results["ids"][0]:
                for i, track_id in enumerate(results["ids"][0]):
                    metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                    distance = results["distances"][0][i] if results["distances"] else 0

                    # 거리를 유사도 점수로 변환 (0~1, 높을수록 유사)
                    similarity = max(0, 1 - distance / 2)

                    tracks.append({
                        "track_id": track_id,
                        "artist": metadata.get("artist", "Unknown"),
                        "title": metadata.get("title", "Unknown"),
                        "tags": metadata.get("tags", ""),
                        "genre": metadata.get("genre", ""),
                        "similarity_score": round(similarity, 3),
                        "audio_features": {
                            "energy": metadata.get("energy"),
                            "valence": metadata.get("valence"),
                            "acousticness": metadata.get("acousticness"),
                        }
                    })

            return tracks

        except Exception as e:
            logger.error(f"검색 실패: {e}")
            return []

    def delete_track(self, track_id: str) -> bool:
        """트랙 삭제"""
        if not self.is_ready:
            return False

        try:
            self.collection.delete(ids=[str(track_id)])
            return True
        except Exception as e:
            logger.error(f"삭제 실패: {e}")
            return False

    def clear_collection(self) -> bool:
        """컬렉션 전체 삭제 (주의!)"""
        if not self.is_ready:
            return False

        try:
            self.client.delete_collection(COLLECTION_NAME)
            self.collection = self.client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"description": "Music tracks with artist tags for semantic search"}
            )
            logger.info("컬렉션 초기화 완료")
            return True
        except Exception as e:
            logger.error(f"컬렉션 삭제 실패: {e}")
            return False


# 싱글톤 인스턴스
_vector_search_service: Optional[VectorSearchService] = None


def get_vector_search_service() -> VectorSearchService:
    """VectorSearchService 싱글톤 반환"""
    global _vector_search_service
    if _vector_search_service is None:
        _vector_search_service = VectorSearchService()
    return _vector_search_service


async def semantic_search(
    query: str,
    n_results: int = 10,
    filters: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    자연어 의미 기반 검색 (async wrapper)

    Args:
        query: 자연어 검색어
        n_results: 결과 수
        filters: LLM 파싱된 필터 조건

    Returns:
        검색 결과
    """
    service = get_vector_search_service()

    if not service.is_ready:
        return {
            "success": False,
            "error": "벡터 검색 서비스가 준비되지 않았습니다.",
            "tracks": [],
            "total": 0
        }

    tracks = service.search(query, n_results, filters)

    return {
        "success": True,
        "query": query,
        "tracks": tracks,
        "total": len(tracks),
        "collection_size": service.collection_count
    }
