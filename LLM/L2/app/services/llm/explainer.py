"""
추천 이유 설명 서비스 (Recommendation Explainer) - Google Gemini 버전

SVM 모델의 추천 결과에 대해 LLM이 자연어 설명을 생성합니다.
"""
from typing import Optional
import google.generativeai as genai

from .config import get_llm_config

# LLM 클라이언트 초기화
config = get_llm_config()
model = None

if config.GOOGLE_API_KEY:
    genai.configure(api_key=config.GOOGLE_API_KEY)
    model = genai.GenerativeModel(
        model_name=config.DEFAULT_MODEL,
        system_instruction=config.SYSTEM_PROMPT
    )


# 에너지/발랄함 레벨 해석
def _interpret_level(value: float) -> str:
    """0-1 값을 한국어 레벨로 변환"""
    if value < 0.3:
        return "낮음"
    elif value < 0.6:
        return "중간"
    else:
        return "높음"


def _interpret_energy(value: float) -> str:
    """에너지 값을 설명으로 변환"""
    if value < 0.3:
        return "잔잔하고 차분한"
    elif value < 0.6:
        return "적당한 에너지의"
    else:
        return "신나고 에너지 넘치는"


def _interpret_valence(value: float) -> str:
    """발랄함 값을 설명으로 변환"""
    if value < 0.3:
        return "우울하거나 감성적인"
    elif value < 0.6:
        return "중립적인 분위기의"
    else:
        return "밝고 긍정적인"


EXPLANATION_PROMPT = """사용자 음악 취향:
- 평균 에너지: {user_energy:.2f} ({user_energy_desc})
- 평균 발랄함: {user_valence:.2f} ({user_valence_desc})
- 선호 장르/태그: {top_genres}
- 최근 아티스트: {recent_artists}

추천된 곡:
- 아티스트: {artist}
- 곡명: {title}
- 아티스트 스타일: {artist_tags}
- 에너지: {track_energy:.2f} ({track_energy_desc})
- 발랄함: {track_valence:.2f} ({track_valence_desc})
- 어쿠스틱: {acousticness:.2f}

위 정보를 바탕으로 이 곡을 추천한 이유를 친근한 말투로 2문장으로 설명해주세요.
반드시 한국어로 작성하고, 구체적인 숫자는 언급하지 마세요.
중요: 위에 제공된 정보만 사용하고, 추측하거나 새로운 정보를 만들어내지 마세요."""


async def explain_recommendation(
    track: dict,
    user_preferences: dict,
) -> dict:
    """
    추천된 곡에 대한 설명 생성 (Gemini 사용)

    Args:
        track: 추천된 곡 정보
            - artist: str
            - title: str
            - audio_features: dict (energy, valence, acousticness 등)
        user_preferences: 사용자 취향 정보
            - avg_energy: float
            - avg_valence: float
            - top_genres: list[str]
            - recent_artists: list[str] (선택)

    Returns:
        dict:
            - explanation: str (자연어 설명)
            - match_score: float (매칭 점수, 선택)
    """
    if not model:
        return {
            "explanation": "API 키가 설정되지 않아 설명을 생성할 수 없습니다.",
            "match_score": None
        }

    # 오디오 피처 추출
    audio = track.get("audio_features", {})
    track_energy = audio.get("energy", 0.5)
    track_valence = audio.get("valence", 0.5)
    acousticness = audio.get("acousticness", 0.5)

    # 사용자 취향 추출
    user_energy = user_preferences.get("avg_energy", 0.5)
    user_valence = user_preferences.get("avg_valence", 0.5)
    top_genres = user_preferences.get("top_genres", ["pop"])
    recent_artists = user_preferences.get("recent_artists", [])

    # 아티스트 태그 추출 (없으면 빈 문자열)
    artist_tags = track.get("artist_tags", "") or track.get("lfm_artist_tags", "")
    if not artist_tags:
        artist_tags = "정보 없음"
    elif isinstance(artist_tags, list):
        artist_tags = ", ".join(artist_tags[:5])
    else:
        # 문자열인 경우 앞 5개 태그만 사용
        tags_list = [t.strip() for t in str(artist_tags).split(",")][:5]
        artist_tags = ", ".join(tags_list) if tags_list else "정보 없음"

    # 프롬프트 생성
    prompt = EXPLANATION_PROMPT.format(
        user_energy=user_energy,
        user_energy_desc=_interpret_energy(user_energy),
        user_valence=user_valence,
        user_valence_desc=_interpret_valence(user_valence),
        top_genres=", ".join(top_genres[:5]),
        recent_artists=", ".join(recent_artists[:3]) if recent_artists else "없음",
        artist=track.get("artist", "Unknown"),
        title=track.get("title", "Unknown"),
        artist_tags=artist_tags,
        track_energy=track_energy,
        track_energy_desc=_interpret_energy(track_energy),
        track_valence=track_valence,
        track_valence_desc=_interpret_valence(track_valence),
        acousticness=acousticness
    )

    try:
        # Gemini API 호출
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=config.MAX_TOKENS,
                temperature=config.TEMPERATURE
            )
        )

        explanation = response.text.strip()

        # 매칭 점수 계산 (에너지와 발랄함 유사도)
        energy_diff = abs(track_energy - user_energy)
        valence_diff = abs(track_valence - user_valence)
        match_score = 1 - (energy_diff + valence_diff) / 2

        return {
            "explanation": explanation,
            "match_score": round(match_score, 2)
        }

    except Exception as e:
        return {
            "explanation": f"설명 생성 중 오류가 발생했습니다: {str(e)}",
            "match_score": None
        }


async def explain_recommendation_batch(
    tracks: list[dict],
    user_preferences: dict,
    max_tracks: int = 5
) -> list[dict]:
    """
    여러 곡에 대한 설명 일괄 생성

    Args:
        tracks: 추천된 곡 목록
        user_preferences: 사용자 취향 정보
        max_tracks: 최대 처리 곡 수

    Returns:
        list[dict]: 각 곡에 대한 설명 목록
    """
    results = []

    for track in tracks[:max_tracks]:
        result = await explain_recommendation(track, user_preferences)
        results.append({
            "track": track,
            **result
        })

    return results


# ============================================
# 검색 맥락 기반 설명 생성 (자연어 검색용)
# ============================================

CONTEXTUAL_PROMPT = """사용자 검색어: "{search_context}"

추천된 곡 목록:
{tracks_info}

위 검색어를 바탕으로, 각 곡이 왜 이 검색에 적합한지 설명해주세요.

규칙:
1. 각 곡마다 1-2문장으로 짧게 설명
2. 검색어의 분위기/상황과 연결해서 설명
3. 친근한 말투 사용 (예: ~해요, ~입니다)
4. 제공된 태그와 장르 정보만 활용
5. 숫자나 점수 언급하지 않기

응답 형식 (JSON 배열):
["첫 번째 곡 설명", "두 번째 곡 설명", ...]"""


async def generate_contextual_explanation(
    tracks: list[dict],
    search_context: str
) -> list[str]:
    """
    검색 맥락을 반영한 추천 이유 생성 (배치)

    Args:
        tracks: 검색 결과 곡 목록
        search_context: 사용자 검색어 (예: "비오는 날 카페 음악")

    Returns:
        list[str]: 각 곡에 대한 맥락 맞춤 설명
    """
    if not model:
        return ["API 키가 설정되지 않았습니다."] * len(tracks)

    if not tracks:
        return []

    # 곡 정보 포맷팅
    tracks_info = ""
    for i, track in enumerate(tracks, 1):
        artist = track.get("artist", "Unknown")
        title = track.get("title", "Unknown")
        genre = track.get("genre", "")
        tags = track.get("tags", "")

        tracks_info += f"{i}. {artist} - {title}\n"
        tracks_info += f"   장르: {genre}\n"
        tracks_info += f"   태그: {tags[:100]}\n\n"

    # 프롬프트 생성
    prompt = CONTEXTUAL_PROMPT.format(
        search_context=search_context,
        tracks_info=tracks_info
    )

    try:
        # Gemini API 호출 (배치)
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=2000,  # 20곡 기준
                temperature=0.7
            )
        )

        response_text = response.text.strip()

        # JSON 파싱 시도
        import json
        try:
            # JSON 배열 추출
            if "[" in response_text and "]" in response_text:
                start = response_text.index("[")
                end = response_text.rindex("]") + 1
                json_str = response_text[start:end]
                explanations = json.loads(json_str)

                # 곡 수에 맞게 조정
                while len(explanations) < len(tracks):
                    explanations.append("이 곡을 추천드려요.")
                return explanations[:len(tracks)]

        except (json.JSONDecodeError, ValueError):
            pass

        # JSON 파싱 실패 시 줄 단위로 분리
        lines = [line.strip() for line in response_text.split("\n") if line.strip()]
        explanations = []
        for line in lines:
            # 번호 제거 (1. 2. 등)
            if line and line[0].isdigit() and "." in line[:3]:
                line = line.split(".", 1)[1].strip()
            if line and not line.startswith("["):
                explanations.append(line)

        while len(explanations) < len(tracks):
            explanations.append("이 곡을 추천드려요.")

        return explanations[:len(tracks)]

    except Exception as e:
        # 오류 시 기본 메시지
        return [f"'{search_context}'에 어울리는 곡이에요."] * len(tracks)
