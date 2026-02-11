"""
LLM Service Package

추천 시스템에 LLM 기능을 추가합니다.
- 추천 이유 설명 (explainer)
- 쿼리 분석: 번역 + 감성 분석 (query_analyzer)
- 벡터 기반 자연어 검색 (vector_search)
- 감성 기반 추천 (emotion) - 추후 추가
"""

from .config import LLMConfig, get_llm_config
from .explainer import explain_recommendation, explain_recommendation_batch
from .query_analyzer import analyze_query, get_query_analyzer
from .vector_search import (
    VectorSearchService,
    get_vector_search_service,
    semantic_search
)

__all__ = [
    "LLMConfig",
    "get_llm_config",
    "explain_recommendation",
    "explain_recommendation_batch",
    "analyze_query",
    "get_query_analyzer",
    "VectorSearchService",
    "get_vector_search_service",
    "semantic_search",
]
