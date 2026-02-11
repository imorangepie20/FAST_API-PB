"""
LLM Service Configuration - Google Gemini
"""
import os
from dotenv import load_dotenv

load_dotenv()


class LLMConfig:
    """LLM 서비스 설정 (Gemini)"""

    # Google Gemini API 설정
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")

    # 모델 설정
    DEFAULT_MODEL: str = "gemini-2.0-flash"  # 무료, 빠른 모델

    # 생성 파라미터
    MAX_TOKENS: int = 200
    TEMPERATURE: float = 0.7  # 창의성 (0=결정적, 1=창의적)

    # 프롬프트 설정
    SYSTEM_PROMPT: str = """당신은 음악 추천 시스템의 AI 어시스턴트입니다.
사용자의 음악 취향을 분석하고, 추천된 곡이 왜 사용자에게 맞는지 친근하게 설명합니다.
설명은 간결하고 따뜻한 말투로 작성하세요."""


def get_llm_config() -> LLMConfig:
    """LLM 설정 인스턴스 반환"""
    return LLMConfig()
