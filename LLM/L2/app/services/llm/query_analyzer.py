"""
LLM 쿼리 분석기 - 번역 + 감성 분석 통합

한국어 쿼리를 영어로 번역하고, 감성 분석을 통해 오디오 특성 필터를 추출합니다.
1회의 LLM 호출로 두 가지를 동시에 처리합니다.
"""
import json
import logging
from typing import Optional, Dict, Any

from .config import get_llm_config

logger = logging.getLogger(__name__)

# Gemini API 설정
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None


# 쿼리 분석 프롬프트
QUERY_ANALYSIS_PROMPT = """You are a music search query analyzer. Analyze the user's query and return a JSON response.

User Query: {query}

Tasks:
1. Translate the query to English music search terms (genres, moods, styles, activities)
2. Analyze the emotional tone and suggest audio feature ranges (0.0 to 1.0)

Return ONLY valid JSON in this exact format:
{{
    "english_query": "translated search terms for music tags",
    "detected_emotion": "emotion name or null",
    "audio_filters": {{
        "energy_min": null or 0.0-1.0,
        "energy_max": null or 0.0-1.0,
        "valence_min": null or 0.0-1.0,
        "valence_max": null or 0.0-1.0
    }}
}}

Audio feature guidelines:
- energy: 0.0 = calm/quiet, 1.0 = energetic/loud
- valence: 0.0 = sad/melancholic, 1.0 = happy/cheerful

Examples:
- "우울할 때 듣는 잔잔한 음악" → energy_max: 0.4, valence_max: 0.4
- "운동할 때 신나는 음악" → energy_min: 0.7, valence_min: 0.6
- "카페에서 일할 때" → energy: 0.3-0.6 (moderate)
- "비오는 날 재즈" → No specific filters, just translate

Only set audio filters when the query clearly implies an emotional state or energy level.
If the query is just about genre/artist style, leave audio_filters as null values."""


class QueryAnalyzer:
    """LLM 기반 쿼리 분석기"""

    def __init__(self):
        self.model = None
        self._initialized = False

        if not GEMINI_AVAILABLE:
            logger.warning("Gemini SDK가 설치되지 않았습니다.")
            return

        # config.py와 동일한 GOOGLE_API_KEY 사용
        config = get_llm_config()
        api_key = config.GOOGLE_API_KEY
        if not api_key:
            logger.warning("GOOGLE_API_KEY가 설정되지 않았습니다.")
            return

        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(config.DEFAULT_MODEL)
            self._initialized = True
            logger.info("QueryAnalyzer 초기화 완료")
        except Exception as e:
            logger.error(f"QueryAnalyzer 초기화 실패: {e}")

    @property
    def is_ready(self) -> bool:
        return self._initialized and self.model is not None

    def _is_english(self, text: str) -> bool:
        """텍스트가 영어인지 간단히 확인"""
        # ASCII 문자 비율로 판단
        ascii_chars = sum(1 for c in text if ord(c) < 128)
        return ascii_chars / len(text) > 0.8 if text else True

    async def analyze(self, query: str) -> Dict[str, Any]:
        """
        쿼리 분석: 번역 + 감성 분석

        Args:
            query: 사용자 검색어 (한국어 또는 영어)

        Returns:
            {
                "original_query": str,
                "english_query": str,
                "detected_emotion": str or None,
                "audio_filters": {
                    "energy_min": float or None,
                    "energy_max": float or None,
                    "valence_min": float or None,
                    "valence_max": float or None
                },
                "used_llm": bool
            }
        """
        result = {
            "original_query": query,
            "english_query": query,
            "detected_emotion": None,
            "audio_filters": {},
            "used_llm": False
        }

        # 영어 쿼리이고 감성 키워드가 없으면 LLM 스킵
        if self._is_english(query) and not self._has_emotion_keywords(query):
            return result

        # LLM 사용 불가시 원본 반환
        if not self.is_ready:
            logger.warning("QueryAnalyzer가 준비되지 않아 원본 쿼리 사용")
            return result

        try:
            prompt = QUERY_ANALYSIS_PROMPT.format(query=query)
            response = self.model.generate_content(prompt)

            # JSON 파싱
            text = response.text.strip()
            # ```json ... ``` 형식 처리
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            text = text.strip()

            parsed = json.loads(text)

            result["english_query"] = parsed.get("english_query", query)
            result["detected_emotion"] = parsed.get("detected_emotion")
            result["used_llm"] = True

            # 오디오 필터 추출
            audio_filters = parsed.get("audio_filters", {})
            if audio_filters:
                for key in ["energy_min", "energy_max", "valence_min", "valence_max"]:
                    value = audio_filters.get(key)
                    if value is not None and isinstance(value, (int, float)):
                        result["audio_filters"][key] = float(value)

            logger.info(f"쿼리 분석 완료: '{query}' → '{result['english_query']}'")
            if result["audio_filters"]:
                logger.info(f"감성 필터: {result['audio_filters']}")

            return result

        except json.JSONDecodeError as e:
            logger.error(f"JSON 파싱 실패: {e}")
            return result
        except Exception as e:
            logger.error(f"쿼리 분석 실패: {e}")
            return result

    def _has_emotion_keywords(self, text: str) -> bool:
        """감성 관련 키워드 포함 여부"""
        emotion_keywords = [
            "sad", "happy", "energetic", "calm", "relaxing", "upbeat",
            "melancholic", "cheerful", "workout", "study", "sleep",
            "party", "chill", "focus", "meditation"
        ]
        text_lower = text.lower()
        return any(kw in text_lower for kw in emotion_keywords)


# 싱글톤 인스턴스
_query_analyzer: Optional[QueryAnalyzer] = None


def get_query_analyzer() -> QueryAnalyzer:
    """QueryAnalyzer 싱글톤 반환"""
    global _query_analyzer
    if _query_analyzer is None:
        _query_analyzer = QueryAnalyzer()
    return _query_analyzer


async def analyze_query(query: str) -> Dict[str, Any]:
    """쿼리 분석 (async wrapper)"""
    analyzer = get_query_analyzer()
    return await analyzer.analyze(query)
