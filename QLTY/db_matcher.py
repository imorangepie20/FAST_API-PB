"""
DB Matcher — 2순위 오디오 피처 소스

spotify_reference 테이블(10만곡 CSV 데이터)에서
title+artist 조합으로 매칭하여 오디오 피처를 가져온다.

API 호출 없이 로컬 DB만 사용하므로 즉시 응답.
"""
import logging
from typing import Optional, Dict
from sqlalchemy.orm import Session
from sqlalchemy import text

logger = logging.getLogger(__name__)

# spotify_reference에서 가져올 오디오 피처 컬럼들
FEATURE_COLUMNS = [
    "danceability", "energy", "valence", "tempo",
    "acousticness", "instrumentalness", "liveness",
    "speechiness", "loudness", "music_key", "mode",
    "time_signature",
]

# 매칭 쿼리: title + artist 정확 매칭 (대소문자 무시, 앞뒤 공백 제거)
# audio features가 있는 레코드만 대상 (danceability IS NOT NULL)
# 여러 건 매칭 시 popularity 높은 것 우선
MATCH_QUERY = text("""
    SELECT danceability, energy, valence, tempo,
           acousticness, instrumentalness, liveness,
           speechiness, loudness, music_key, mode,
           time_signature
    FROM spotify_reference
    WHERE LOWER(TRIM(track_name)) = LOWER(TRIM(:title))
      AND LOWER(TRIM(artist_name)) = LOWER(TRIM(:artist))
      AND danceability IS NOT NULL
    ORDER BY popularity DESC
    LIMIT 1
""")


def match_track(
    title: str,
    artist: str,
    db: Session,
) -> Optional[Dict]:
    """
    title+artist로 spotify_reference에서 오디오 피처를 매칭한다.

    Args:
        title: 트랙 제목
        artist: 아티스트명
        db: SQLAlchemy Session

    Returns:
        성공 시: {"danceability": 0.65, ...} dict
        실패 시: None
    """
    if not title or not artist:
        return None

    try:
        row = db.execute(
            MATCH_QUERY,
            {"title": title.strip(), "artist": artist.strip()},
        ).fetchone()

        if row is None:
            logger.debug(f"[DB Matcher] '{artist} - {title}' → no match")
            return None

        # Row → dict 변환, NULL이 아닌 값만 포함
        features = {}
        for i, col in enumerate(FEATURE_COLUMNS):
            val = row[i]
            if val is not None:
                features[col] = float(val)

        if features:
            logger.info(
                f"[DB Matcher] '{artist} - {title}' → {len(features)}개 피처 매칭"
            )
            return features

    except Exception as e:
        logger.error(f"[DB Matcher] error: {e}")

    return None
