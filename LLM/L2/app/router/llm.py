"""
LLM API Router

LLM 기반 기능들의 API 엔드포인트를 제공합니다.
- /api/llm/explain: 추천 이유 설명
- /api/llm/explain-batch: 여러 곡 설명 (일괄)
- /api/llm/parse-query: 자연어 검색 쿼리 파싱
"""
from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.llm import (
    explain_recommendation,
    explain_recommendation_batch,
    analyze_query,
)
from app.services.llm.vector_search import semantic_search, get_vector_search_service

router = APIRouter(prefix="/api/llm", tags=["LLM"])


# ============================================
# Pydantic Models (Request/Response)
# ============================================

class AudioFeatures(BaseModel):
    """오디오 피처"""
    energy: float = Field(0.5, ge=0, le=1, description="에너지 (0-1)")
    valence: float = Field(0.5, ge=0, le=1, description="발랄함 (0-1)")
    acousticness: float = Field(0.5, ge=0, le=1, description="어쿠스틱 (0-1)")
    danceability: Optional[float] = Field(None, ge=0, le=1)
    tempo: Optional[float] = Field(None, ge=0)


class TrackInfo(BaseModel):
    """곡 정보"""
    artist: str = Field(..., description="아티스트명")
    title: str = Field(..., description="곡 제목")
    audio_features: AudioFeatures = Field(default_factory=AudioFeatures)


class UserPreferences(BaseModel):
    """사용자 취향 정보"""
    avg_energy: float = Field(0.5, ge=0, le=1, description="평균 에너지")
    avg_valence: float = Field(0.5, ge=0, le=1, description="평균 발랄함")
    top_genres: list[str] = Field(default=["pop"], description="선호 장르")
    recent_artists: list[str] = Field(default=[], description="최근 들은 아티스트")


class ExplainRequest(BaseModel):
    """추천 설명 요청"""
    user_id: int = Field(..., description="사용자 ID")
    track: TrackInfo = Field(..., description="추천된 곡 정보")
    user_preferences: Optional[UserPreferences] = Field(
        default=None,
        description="사용자 취향 (없으면 기본값 사용)"
    )


class ExplainResponse(BaseModel):
    """추천 설명 응답"""
    explanation: str = Field(..., description="추천 이유 설명")
    match_score: Optional[float] = Field(None, description="취향 매칭 점수 (0-1)")


class ExplainBatchRequest(BaseModel):
    """일괄 추천 설명 요청"""
    user_id: int
    tracks: list[TrackInfo]
    user_preferences: Optional[UserPreferences] = None
    max_tracks: int = Field(5, ge=1, le=10, description="최대 처리 곡 수")


class ExplainBatchResponse(BaseModel):
    """일괄 추천 설명 응답"""
    results: list[dict]
    total: int


# ============================================
# API Endpoints
# ============================================

@router.post("/explain", response_model=ExplainResponse)
async def explain(request: ExplainRequest):
    """
    추천된 곡에 대한 이유 설명 생성

    SVM 모델이 추천한 곡이 왜 사용자에게 맞는지
    LLM이 자연어로 설명을 생성합니다.
    """
    # 사용자 취향 (없으면 기본값)
    user_prefs = request.user_preferences or UserPreferences()

    # 곡 정보를 dict로 변환
    track_dict = {
        "artist": request.track.artist,
        "title": request.track.title,
        "audio_features": request.track.audio_features.model_dump()
    }

    # 사용자 취향을 dict로 변환
    prefs_dict = user_prefs.model_dump()

    # LLM 설명 생성
    result = await explain_recommendation(track_dict, prefs_dict)

    return ExplainResponse(
        explanation=result["explanation"],
        match_score=result.get("match_score")
    )


@router.post("/explain-batch", response_model=ExplainBatchResponse)
async def explain_batch(request: ExplainBatchRequest):
    """
    여러 곡에 대한 추천 이유 일괄 생성

    비용 제한을 위해 최대 10곡까지만 처리합니다.
    """
    user_prefs = request.user_preferences or UserPreferences()

    # 곡 목록을 dict로 변환
    tracks_dict = [
        {
            "artist": track.artist,
            "title": track.title,
            "audio_features": track.audio_features.model_dump()
        }
        for track in request.tracks
    ]

    prefs_dict = user_prefs.model_dump()

    # 일괄 설명 생성
    results = await explain_recommendation_batch(
        tracks_dict,
        prefs_dict,
        max_tracks=request.max_tracks
    )

    return ExplainBatchResponse(
        results=results,
        total=len(results)
    )


# ============================================
# 자연어 검색 (Natural Language Search)
# ============================================

class QueryParseRequest(BaseModel):
    """자연어 검색 요청"""
    query: str = Field(..., description="자연어 검색어", example="비오는 날 카페에서 들을 잔잔한 음악")
    limit: int = Field(10, ge=1, le=50, description="검색 결과 수")


class QueryParseResponse(BaseModel):
    """자연어 검색 파싱 응답"""
    success: bool
    parsed: Optional[dict] = None
    filters: Optional[dict] = None
    error: Optional[str] = None


@router.post("/parse-query", response_model=QueryParseResponse)
async def parse_query(request: QueryParseRequest):
    """
    자연어 검색어를 구조화된 필터 조건으로 변환

    예시 입력: "비오는 날 카페에서 들을 잔잔한 음악"
    예시 출력:
    {
        "parsed": {"english_query": "calm jazz cafe rainy day", "detected_emotion": "melancholic"},
        "filters": {"energy_min": 0, "energy_max": 0.4}
    }
    """
    result = await analyze_query(request.query)

    return QueryParseResponse(
        success=result.get("used_llm", False) or bool(result.get("english_query")),
        parsed={
            "english_query": result.get("english_query"),
            "detected_emotion": result.get("detected_emotion"),
        },
        filters=result.get("audio_filters") if result.get("audio_filters") else None,
        error=None
    )


@router.get("/genres")
async def get_available_genres():
    """사용 가능한 장르 목록 조회 (벡터 DB 메타데이터 기반)"""
    service = get_vector_search_service()
    if not service.is_ready:
        return {"genres": [], "total": 0, "error": "벡터 검색 서비스가 준비되지 않았습니다."}

    try:
        # ChromaDB 컬렉션에서 장르 메타데이터 샘플링
        sample = service.collection.get(
            limit=10000,
            include=["metadatas"]
        )
        genres = set()
        for meta in sample.get("metadatas", []):
            genre = meta.get("genre", "")
            if genre:
                genres.add(genre)
        sorted_genres = sorted(genres)
        return {"genres": sorted_genres, "total": len(sorted_genres)}
    except Exception as e:
        return {"genres": [], "total": 0, "error": str(e)}


@router.get("/health")
async def health_check():
    """LLM 서비스 상태 확인 (Google Gemini)"""
    from app.services.llm.config import get_llm_config

    config = get_llm_config()
    has_api_key = bool(config.GOOGLE_API_KEY)

    return {
        "status": "ok" if has_api_key else "no_api_key",
        "provider": "Google Gemini",
        "model": config.DEFAULT_MODEL,
        "api_key_configured": has_api_key
    }


# ============================================
# 벡터 기반 자연어 검색 (ChromaDB) + LLM 설명
# ============================================

# 검색창 플레이스홀더 예시 (프론트엔드에서 3초마다 순환)
PLACEHOLDER_EXAMPLES = [
    "신나는 운동 음악",
    "비오는 날 카페에서 들을 잔잔한 음악",
    "새벽에 혼자 듣는 감성 발라드",
    "집중해서 공부할 때",
    "친구들과 드라이브할 때",
]

# 검색 도움말
SEARCH_HELP = {
    "기분/감정": ["우울할 때", "신날 때", "설렐 때"],
    "상황/장소": ["카페에서", "운동할 때", "출근길에"],
    "분위기": ["잔잔한", "신나는", "몽환적인"],
    "조합 예시": "비오는 밤 혼자 듣는 재즈"
}


class SemanticSearchRequest(BaseModel):
    """벡터 기반 자연어 검색 요청"""
    query: str = Field(..., description="자연어 검색어", example="비오는 날 카페에서 들을 잔잔한 재즈")
    page: int = Field(1, ge=1, le=1, description="페이지 번호")
    include_explanation: bool = Field(True, description="LLM 추천 이유 포함 여부")
    genres: Optional[list[str]] = Field(None, description="장르 필터 (선택)")
    energy_min: Optional[float] = Field(None, ge=0, le=1)
    energy_max: Optional[float] = Field(None, ge=0, le=1)
    valence_min: Optional[float] = Field(None, ge=0, le=1)
    valence_max: Optional[float] = Field(None, ge=0, le=1)


class TrackWithExplanation(BaseModel):
    """곡 정보 + 추천 이유"""
    track_id: str
    artist: str
    title: str
    genre: str
    tags: str
    similarity_score: float
    explanation: Optional[str] = None  # LLM 생성 추천 이유


class SemanticSearchResponse(BaseModel):
    """벡터 검색 응답"""
    success: bool
    query: Optional[str] = None
    tracks: list[dict] = []
    total: int = 0
    page: int = 1
    max_page: int = 3
    has_more: bool = False
    collection_size: int = 0
    error: Optional[str] = None
    # 쿼리 분석 정보 (디버그/확인용)
    analyzed_query: Optional[str] = None  # 영어로 번역된 쿼리
    detected_emotion: Optional[str] = None  # 감지된 감정
    applied_filters: Optional[dict] = None  # 적용된 오디오 필터


@router.post("/search", response_model=SemanticSearchResponse)
async def search_music(request: SemanticSearchRequest):
    """
    자연어로 음악 검색 (벡터 유사도 + LLM 설명)

    - 한국어 쿼리 → LLM이 영어로 번역 + 감성 분석 (1회 호출)
    - 영어 쿼리로 벡터 검색 (정확도 향상)
    - 감성 분석 결과로 오디오 필터 자동 적용
    - 기본 5곡 + 즉시 설명 포함
    - "더 보기" 클릭 시 추가 5곡 (최대 15곡 = 3페이지)

    예시 검색어:
    - "비오는 날 카페에서 들을 잔잔한 재즈"
    - "신나는 운동 음악"
    - "우울할 때 듣는 감성 발라드"
    """
    from app.services.llm.explainer import generate_contextual_explanation
    from app.services.llm.query_analyzer import analyze_query

    # 1단계: 쿼리 분석 (번역 + 감성 분석)
    query_analysis = await analyze_query(request.query)
    search_query = query_analysis["english_query"]  # 영어로 번역된 쿼리

    # 20곡 반환
    page_size = 20
    offset = 0

    # 2단계: 필터 구성 (요청 필터 + LLM 감성 필터 병합)
    filters = {}

    # 명시적 요청 필터 (우선순위 높음)
    if request.genres:
        filters["genres"] = request.genres
    if request.energy_min is not None:
        filters["energy_min"] = request.energy_min
    if request.energy_max is not None:
        filters["energy_max"] = request.energy_max
    if request.valence_min is not None:
        filters["valence_min"] = request.valence_min
    if request.valence_max is not None:
        filters["valence_max"] = request.valence_max

    # LLM 감성 분석 필터 추가 (명시적 필터가 없는 경우에만)
    llm_filters = query_analysis.get("audio_filters", {})
    for key, value in llm_filters.items():
        if key not in filters and value is not None:
            filters[key] = value

    # 3단계: 벡터 검색 (영어 쿼리 사용)
    result = await semantic_search(
        query=search_query,  # 영어로 번역된 쿼리
        n_results=offset + page_size,
        filters=filters if filters else None
    )

    if not result.get("success"):
        return SemanticSearchResponse(
            success=False,
            error=result.get("error", "검색 실패"),
            query=request.query
        )

    # 현재 페이지의 곡만 추출
    all_tracks = result.get("tracks", [])
    page_tracks = all_tracks[offset:offset + page_size]

    # 4단계: LLM 설명 추가 (요청 시)
    if request.include_explanation and page_tracks:
        try:
            # 원본 한국어 쿼리로 설명 생성 (사용자 친화적)
            explanations = await generate_contextual_explanation(
                tracks=page_tracks,
                search_context=request.query  # 원본 쿼리 사용
            )
            for i, track in enumerate(page_tracks):
                if i < len(explanations):
                    track["explanation"] = explanations[i]
        except Exception as e:
            for track in page_tracks:
                track["explanation"] = None

    return SemanticSearchResponse(
        success=True,
        query=request.query,
        tracks=page_tracks,
        total=len(page_tracks),
        page=request.page,
        max_page=3,
        has_more=request.page < 3 and len(all_tracks) > offset + page_size,
        collection_size=result.get("collection_size", 0),
        # 쿼리 분석 정보
        analyzed_query=search_query,
        detected_emotion=query_analysis.get("detected_emotion"),
        applied_filters=filters if filters else None
    )


@router.get("/search/help")
async def get_search_help():
    """검색 도움말 및 플레이스홀더 예시 조회"""
    return {
        "placeholders": PLACEHOLDER_EXAMPLES,
        "help": SEARCH_HELP,
        "settings": {
            "default_results": 5,
            "max_results": 15,
            "max_pages": 3,
            "includes_explanation": True
        }
    }


@router.get("/search/status")
async def search_status():
    """벡터 검색 서비스 상태 확인"""
    service = get_vector_search_service()

    return {
        "ready": service.is_ready,
        "collection_size": service.collection_count,
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "db_type": "ChromaDB",
        "message": "Ready for semantic search" if service.is_ready else "Service not initialized"
    }
