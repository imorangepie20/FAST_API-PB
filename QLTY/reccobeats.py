"""
ReccoBeats API Client — 1순위 오디오 피처 소스

Spotify Audio Features의 무료 대체제.
ISRC(국제표준음반코드)로 실제 오디오 분석 기반 피처를 조회한다.

엔드포인트: GET https://api.reccobeats.com/v1/track/audio-features?isrc={isrc}
Rate Limit: 과도한 요청 시 차단 가능 → batch 5건, 0.5초 간격
"""
import httpx
import asyncio
import logging
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)

BASE_URL = "https://api.reccobeats.com/v1/track/audio-features"

# ReccoBeats가 반환하는 피처 키 → 우리 DB 컬럼명 매핑
# ReccoBeats는 Spotify와 동일한 키명을 사용하므로 대부분 1:1
FEATURE_KEYS = [
    "danceability", "energy", "valence", "tempo",
    "acousticness", "instrumentalness", "liveness",
    "speechiness", "loudness",
]


async def fetch_by_isrc(isrc: str, timeout: float = 10.0) -> Optional[Dict]:
    """
    단일 ISRC로 ReccoBeats API를 호출하여 오디오 피처를 가져온다.

    Args:
        isrc: 국제표준음반코드 (예: "USRC11700356")
        timeout: HTTP 요청 타임아웃(초)

    Returns:
        성공 시: {"danceability": 0.65, "energy": 0.8, ...} 형태의 dict
        실패 시: None
    """
    if not isrc or not isrc.strip():
        return None

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(BASE_URL, params={"isrc": isrc.strip()})

            if resp.status_code == 200:
                data = resp.json()
                # 응답에서 우리가 필요한 피처만 추출
                features = {}
                for key in FEATURE_KEYS:
                    val = data.get(key)
                    if val is not None:
                        features[key] = float(val)

                if features:
                    logger.info(f"[ReccoBeats] ISRC={isrc} → {len(features)}개 피처 획득")
                    return features

            elif resp.status_code == 404:
                # DB에 해당 ISRC가 없음 — 정상적인 miss
                logger.debug(f"[ReccoBeats] ISRC={isrc} → not found (404)")
            else:
                logger.warning(
                    f"[ReccoBeats] ISRC={isrc} → HTTP {resp.status_code}"
                )

    except httpx.TimeoutException:
        logger.warning(f"[ReccoBeats] ISRC={isrc} → timeout")
    except Exception as e:
        logger.error(f"[ReccoBeats] ISRC={isrc} → error: {e}")

    return None


async def fetch_batch(
    isrc_list: List[str],
    batch_size: int = 5,
    delay: float = 0.5,
) -> Dict[str, Dict]:
    """
    여러 ISRC를 배치로 조회한다.
    Rate limit 보호를 위해 batch_size건마다 delay초 대기.

    Args:
        isrc_list: ISRC 목록
        batch_size: 한 번에 동시 요청할 수 (기본 5)
        delay: 배치 간 대기 시간(초) (기본 0.5)

    Returns:
        {isrc: features_dict} — 성공한 것만 포함
    """
    results = {}

    for i in range(0, len(isrc_list), batch_size):
        batch = isrc_list[i : i + batch_size]

        # 배치 내 동시 요청
        tasks = [fetch_by_isrc(isrc) for isrc in batch]
        responses = await asyncio.gather(*tasks)

        for isrc, features in zip(batch, responses):
            if features:
                results[isrc] = features

        # 다음 배치 전 대기 (마지막 배치 제외)
        if i + batch_size < len(isrc_list):
            await asyncio.sleep(delay)

    logger.info(
        f"[ReccoBeats] 배치 완료: {len(results)}/{len(isrc_list)} 성공 "
        f"({len(results)/len(isrc_list)*100:.1f}%)"
    )
    return results
