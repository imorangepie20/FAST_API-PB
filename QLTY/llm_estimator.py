"""
LLM Estimator — 3순위 오디오 피처 소스 (Google Search Grounding)

Gemini + Google Search를 활용하여 웹에서 실제 오디오 피처를
검색하고, 찾을 수 없는 값은 음악 지식으로 추정한다.

동작 방식:
1. Gemini가 Google Search로 tunebat.com, songbpm.com 등 검색
2. 검색 결과에서 실제 Spotify 오디오 피처 값 추출
3. 찾을 수 없는 피처는 음악 지식 기반 추정

사용 SDK: google.genai (v1.x, 신규 SDK)
사용 모델: gemini-2.0-flash + Google Search Grounding
Temperature: 0.1 (검색 기반이므로 더 결정적으로)
"""
import os
import json
import logging
from typing import Optional, Dict

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

# Gemini Client 인스턴스 (모듈 수명 동안 유지)
_client = None


def _get_client():
    """google.genai Client를 lazy-load한다."""
    global _client
    if _client is not None:
        return _client

    api_key = os.getenv("GOOGLE_API_KEY", "")
    if not api_key:
        logger.warning("[LLM Estimator] GOOGLE_API_KEY 미설정")
        return None

    try:
        _client = genai.Client(api_key=api_key)
        logger.info("[LLM Estimator] google.genai Client 초기화 완료")
        return _client
    except Exception as e:
        logger.error(f"[LLM Estimator] Client 초기화 실패: {e}")
        return None


# ==================== 프롬프트 ====================

# Google Search 활성화 프롬프트 — 검색을 유도
SEARCH_PROMPT = """Search for the Spotify audio features of this track:

Artist: {artist}
Title: {title}

Search on tunebat.com, songbpm.com, or musicstax.com for the actual Spotify audio features.

Return ONLY a JSON object with these keys (use actual values from search results, estimate if not found):
- "danceability": 0.0-1.0
- "energy": 0.0-1.0
- "valence": 0.0-1.0
- "tempo": BPM number (40-250)
- "acousticness": 0.0-1.0
- "instrumentalness": 0.0-1.0
- "liveness": 0.0-1.0
- "speechiness": 0.0-1.0
- "loudness": dB number (-60 to 5)

JSON only, no other text:"""

# Search 실패 시 순수 추정 프롬프트
FALLBACK_PROMPT = """Estimate Spotify-style audio features for this track:

Artist: {artist}
Title: {title}
Album: {album}
Genre: {genre}
Duration: {duration_sec} seconds

Return ONLY a JSON object with these keys and value ranges:
- "danceability": 0.0-1.0 (how suitable for dancing)
- "energy": 0.0-1.0 (intensity and activity)
- "valence": 0.0-1.0 (musical positiveness)
- "tempo": 40-250 (BPM)
- "acousticness": 0.0-1.0 (acoustic confidence)
- "instrumentalness": 0.0-1.0 (no vocals predicted)
- "liveness": 0.0-1.0 (audience presence)
- "speechiness": 0.0-1.0 (spoken words)
- "loudness": -60 to 5 (dB)

JSON only, no other text:"""


# 응답에서 추출할 피처와 유효 범위
FEATURE_RANGES = {
    "danceability": (0.0, 1.0),
    "energy": (0.0, 1.0),
    "valence": (0.0, 1.0),
    "tempo": (40.0, 250.0),
    "acousticness": (0.0, 1.0),
    "instrumentalness": (0.0, 1.0),
    "liveness": (0.0, 1.0),
    "speechiness": (0.0, 1.0),
    "loudness": (-60.0, 5.0),
}


def _parse_response(text: str) -> Optional[Dict]:
    """
    LLM 응답 텍스트에서 JSON을 추출하고 유효성을 검증한다.

    LLM이 ```json ... ``` 래퍼를 붙이는 경우가 있으므로
    중괄호 기준으로 JSON 부분만 추출한다.

    Returns:
        유효한 피처 dict 또는 None
    """
    if not text:
        return None

    # ```json ... ``` 래퍼 제거
    cleaned = text.strip()
    if "```" in cleaned:
        parts = cleaned.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                cleaned = part
                break

    # { ... } 추출
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        logger.warning("[LLM Estimator] JSON 추출 실패")
        return None

    json_str = cleaned[start : end + 1]

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.warning(f"[LLM Estimator] JSON 파싱 실패: {e}")
        return None

    # 유효성 검증: 범위 밖 값은 클리핑, 타입 오류는 제외
    features = {}
    for key, (lo, hi) in FEATURE_RANGES.items():
        val = data.get(key)
        if val is None:
            continue

        try:
            val = float(val)
        except (ValueError, TypeError):
            continue

        # 범위 클리핑
        val = max(lo, min(hi, val))
        features[key] = round(val, 3)

    if not features:
        logger.warning("[LLM Estimator] 유효한 피처 없음")
        return None

    return features


async def estimate(
    title: str,
    artist: str,
    album: str = "",
    genre: str = "",
    duration_ms: float = 0,
) -> Optional[Dict]:
    """
    Gemini + Google Search로 오디오 피처를 조회/추정한다.

    1차: Google Search Grounding으로 실제 데이터 검색
    2차: Search 실패 시 순수 LLM 추정 fallback

    Args:
        title: 트랙 제목
        artist: 아티스트명
        album: 앨범명 (선택)
        genre: 장르 (선택)
        duration_ms: 길이 밀리초 (선택)

    Returns:
        성공 시: {"danceability": 0.65, ...} dict
        실패 시: None
    """
    if not title or not artist:
        return None

    client = _get_client()
    if client is None:
        return None

    # ========== 1차: Google Search Grounding ==========
    try:
        prompt = SEARCH_PROMPT.format(artist=artist, title=title)

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
                max_output_tokens=500,
                temperature=0.1,
            ),
        )

        features = _parse_response(response.text)

        if features and len(features) >= 5:
            logger.info(
                f"[LLM+Search] '{artist} - {title}' → "
                f"{len(features)}개 피처 (검색 기반)"
            )
            return features
        else:
            logger.info(
                f"[LLM+Search] '{artist} - {title}' → "
                f"검색 결과 부족, fallback 시도"
            )

    except Exception as e:
        logger.warning(f"[LLM+Search] 검색 실패: {e}")

    # ========== 2차: 순수 LLM 추정 (fallback) ==========
    try:
        duration_sec = round(duration_ms / 1000) if duration_ms else 0
        prompt = FALLBACK_PROMPT.format(
            artist=artist,
            title=title,
            album=album or "Unknown",
            genre=genre or "Unknown",
            duration_sec=duration_sec or "Unknown",
        )

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=300,
                temperature=0.2,
            ),
        )

        features = _parse_response(response.text)

        if features:
            logger.info(
                f"[LLM Fallback] '{artist} - {title}' → "
                f"{len(features)}개 피처 (LLM 추정)"
            )
        else:
            logger.warning(
                f"[LLM Fallback] '{artist} - {title}' → 추정 실패"
            )

        return features

    except Exception as e:
        logger.error(f"[LLM Fallback] API 호출 실패: {e}")
        return None
