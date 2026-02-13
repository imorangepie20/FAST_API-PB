"""
QLTY Pipeline — 오디오 피처 보강 오케스트레이터

불완전한 트랙 데이터를 받아 다단계 우선순위로
오디오 피처를 채워 넣는 핵심 파이프라인.

우선순위:
1순위: ReccoBeats API (ISRC → 실제 오디오 분석)
2순위: spotify_reference DB 매칭 (title+artist)
3순위: Gemini + Google Search → Divergence-Based Routing (M1과 비교)
4순위: DL 모델 예측 (학습된 LightGBM, 최후 수단)

Divergence-Based Routing (3순위 이후 적용):
  QLTY(Gemini)와 M1(Ridge)을 동시에 돌린 뒤,
  두 예측의 차이(divergence)를 기준으로 최적 값을 선택한다.
  - div > 0.16 → M1 (QLTY 신뢰 불가)
  - QLTY 강점 장르/피처 → QLTY
  - div < 0.05 → 평균 (두 모델 합의)
  - default → M1 (안전)
"""
import logging
from typing import Optional, Dict, List
from dataclasses import dataclass, field
from sqlalchemy.orm import Session

from . import reccobeats, db_matcher, llm_estimator, m1_predictor
from .model import AudioFeatureModel, ALL_FEATURES

logger = logging.getLogger(__name__)

# 전역 DL 모델 인스턴스 (서버 수명 동안 1회 로드)
_dl_model: Optional[AudioFeatureModel] = None


def _get_dl_model() -> Optional[AudioFeatureModel]:
    """학습된 DL 모델을 lazy-load한다."""
    global _dl_model
    if _dl_model is not None:
        return _dl_model

    model = AudioFeatureModel()
    if model.load():
        _dl_model = model
        logger.info("[QLTY Pipeline] DL 모델 로드 완료")
        return _dl_model
    else:
        logger.warning("[QLTY Pipeline] DL 모델 파일 없음 (학습 필요)")
        return None


@dataclass
class TrackInput:
    """
    파이프라인에 넣을 트랙 데이터.

    필수: title, artist
    선택: 나머지 (있으면 더 정확한 결과)
    """
    title: str
    artist: str
    album: str = ""
    genre: str = ""
    duration_ms: float = 0
    popularity: float = 0
    isrc: str = ""


@dataclass
class EnrichResult:
    """
    파이프라인 실행 결과.

    features: 최종 오디오 피처 dict
    sources: 각 피처가 어디서 왔는지 기록
    attempted: 시도한 소스 목록
    """
    features: Dict = field(default_factory=dict)
    sources: Dict = field(default_factory=dict)
    attempted: List[str] = field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        """12개 피처 전부 채워졌는지 확인"""
        return len(self.features) >= len(ALL_FEATURES)

    @property
    def coverage(self) -> float:
        """피처 채움 비율 (0.0 ~ 1.0)"""
        return len(self.features) / len(ALL_FEATURES) if ALL_FEATURES else 0.0


# ==================== Divergence-Based Routing ====================

# QLTY(Gemini)가 강점인 장르 (승률 60%+)
_QLTY_GENRES = {"pop", "rock", "afrobeats", "gaming", "reggae", "brazilian", "folk", "classical"}

# QLTY가 M1보다 항상 나은 피처
_QLTY_FEATURES = {"tempo", "acousticness"}

# Divergence routing 대상 피처 (0~1 스케일 + tempo + loudness)
_ROUTING_FEATURES = {
    "danceability", "energy", "valence", "tempo",
    "acousticness", "instrumentalness", "speechiness", "loudness",
}


def _norm_diff(feature: str, diff: float) -> float:
    """피처별 normalized difference (0~1 스케일로 통일)."""
    if feature == "tempo":
        return diff / 250.0
    if feature == "loudness":
        return diff / 60.0
    return diff


def _apply_divergence_routing(
    qlty_features: Dict,
    m1_features: Dict,
    genre: str,
    popularity: float,
) -> Dict:
    """
    QLTY와 M1 예측을 비교하여 피처별로 최적 값을 선택한다.

    Returns:
        라우팅된 최종 피처 dict (routing 대상 외 피처는 qlty_features 유지)
    """
    routed = dict(qlty_features)  # QLTY 값을 기본으로 복사

    for feat in _ROUTING_FEATURES:
        qlty_val = qlty_features.get(feat)
        m1_val = m1_features.get(feat)

        if qlty_val is None or m1_val is None:
            continue

        div = _norm_diff(feat, abs(m1_val - qlty_val))

        # 1. High divergence → M1 (QLTY 17% 승률)
        if div > 0.16:
            routed[feat] = m1_val
            continue

        # 2. QLTY 강점 장르 → QLTY 유지
        if genre in _QLTY_GENRES:
            continue

        # 3. QLTY 강점 피처 → QLTY 유지
        if feat in _QLTY_FEATURES:
            continue

        # 4. Low divergence → 평균 (두 모델 합의)
        if div < 0.05:
            routed[feat] = round((m1_val + qlty_val) / 2, 3)
            continue

        # 5. Mid divergence → 가중 평균 (M1:60 QLTY:40)
        if div < 0.10:
            routed[feat] = round(m1_val * 0.6 + qlty_val * 0.4, 3)
            continue

        # 6. High popularity → 가중 평균
        if popularity >= 80:
            routed[feat] = round(m1_val * 0.6 + qlty_val * 0.4, 3)
            continue

        # 7. Default → M1
        routed[feat] = m1_val

    return routed


def _merge_features(
    result: EnrichResult,
    new_features: Dict,
    source_name: str,
) -> int:
    """
    새로운 피처를 결과에 병합한다.
    이미 존재하는 피처는 덮어쓰지 않는다 (상위 소스 우선).

    Args:
        result: 현재까지의 결과
        new_features: 새로 얻은 피처 dict
        source_name: 소스 이름 (예: "reccobeats", "db_match")

    Returns:
        새로 추가된 피처 수
    """
    added = 0
    for key, val in new_features.items():
        if key not in result.features:
            result.features[key] = val
            result.sources[key] = source_name
            added += 1
    return added


async def enrich_track(
    track: TrackInput,
    db: Optional[Session] = None,
    skip_api: bool = False,
    skip_llm: bool = False,
) -> EnrichResult:
    """
    단일 트랙의 오디오 피처를 보강한다.

    4단계 우선순위로 시도하며, 모든 피처가 채워지면 즉시 중단.
    각 단계에서 이미 채워진 피처는 건드리지 않는다.

    Args:
        track: TrackInput 데이터
        db: SQLAlchemy Session (2순위 DB매칭 + 3순위 DL모델에 필요)
        skip_api: True면 ReccoBeats API 호출 스킵 (테스트용)
        skip_llm: True면 LLM 호출 스킵 (비용/속도 절감)

    Returns:
        EnrichResult (features, sources, attempted)
    """
    result = EnrichResult()

    # ========== 1순위: ReccoBeats API (ISRC 필요) ==========
    if not skip_api and track.isrc:
        result.attempted.append("reccobeats")
        try:
            features = await reccobeats.fetch_by_isrc(track.isrc)
            if features:
                added = _merge_features(result, features, "reccobeats")
                logger.info(
                    f"[QLTY] 1순위 ReccoBeats: +{added} 피처 "
                    f"(ISRC={track.isrc})"
                )
                if result.is_complete:
                    return result
        except Exception as e:
            logger.warning(f"[QLTY] ReccoBeats 실패: {e}")

    # ========== 2순위: spotify_reference DB 매칭 ==========
    if db is not None:
        result.attempted.append("db_match")
        try:
            features = db_matcher.match_track(track.title, track.artist, db)
            if features:
                added = _merge_features(result, features, "db_match")
                logger.info(
                    f"[QLTY] 2순위 DB매칭: +{added} 피처 "
                    f"('{track.artist} - {track.title}')"
                )
                if result.is_complete:
                    return result
        except Exception as e:
            logger.warning(f"[QLTY] DB매칭 실패: {e}")

    # ========== 3순위: Gemini + Google Search + Divergence Routing ==========
    if not skip_llm and not result.is_complete:
        result.attempted.append("llm_search")
        try:
            qlty_features = await llm_estimator.estimate(
                title=track.title,
                artist=track.artist,
                album=track.album,
                genre=track.genre,
                duration_ms=track.duration_ms,
            )
            if qlty_features:
                # M1 예측을 받아 Divergence-Based Routing 적용
                m1_features = m1_predictor.predict(
                    title=track.title,
                    artist=track.artist,
                    album=track.album,
                    genre=track.genre,
                    duration_ms=track.duration_ms,
                    popularity=track.popularity,
                )
                if m1_features:
                    result.attempted.append("divergence_routing")
                    routed = _apply_divergence_routing(
                        qlty_features, m1_features,
                        genre=track.genre.lower(),
                        popularity=track.popularity,
                    )
                    routed_count = sum(
                        1 for f in _ROUTING_FEATURES
                        if routed.get(f) != qlty_features.get(f)
                    )
                    added = _merge_features(result, routed, "divergence_routed")
                    logger.info(
                        f"[QLTY] 3순위 LLM+Routing: +{added} 피처 "
                        f"({routed_count}개 M1 라우팅) "
                        f"('{track.artist} - {track.title}')"
                    )
                else:
                    # M1 모델 없음 → QLTY 단독
                    added = _merge_features(result, qlty_features, "llm_search")
                    logger.info(
                        f"[QLTY] 3순위 LLM+Search: +{added} 피처 "
                        f"(M1 없음, QLTY 단독) "
                        f"('{track.artist} - {track.title}')"
                    )
                if result.is_complete:
                    return result
        except Exception as e:
            logger.warning(f"[QLTY] LLM+Search 실패: {e}")

    # ========== 4순위: DL 모델 예측 (최후 수단) ==========
    if not result.is_complete:
        dl_model = _get_dl_model()
        if dl_model is not None:
            result.attempted.append("dl_model")
            try:
                features = dl_model.predict(
                    title=track.title,
                    artist=track.artist,
                    album=track.album,
                    genre=track.genre,
                    duration_ms=track.duration_ms,
                    popularity=track.popularity,
                )
                if features:
                    added = _merge_features(result, features, "dl_model")
                    logger.info(
                        f"[QLTY] 4순위 DL모델: +{added} 피처 "
                        f"('{track.artist} - {track.title}')"
                    )
            except Exception as e:
                logger.warning(f"[QLTY] DL모델 실패: {e}")

    # 최종 로그
    logger.info(
        f"[QLTY] 완료: '{track.artist} - {track.title}' → "
        f"{len(result.features)}/{len(ALL_FEATURES)} 피처 "
        f"({', '.join(result.attempted)})"
    )

    return result


async def enrich_batch(
    tracks: List[TrackInput],
    db: Optional[Session] = None,
    skip_api: bool = False,
    skip_llm: bool = False,
) -> List[EnrichResult]:
    """
    여러 트랙을 순차적으로 보강한다.

    배치 처리이지만 각 트랙은 순차 실행한다.
    (ReccoBeats rate limit 보호 + DB 세션 안전성)

    Args:
        tracks: TrackInput 리스트
        db: SQLAlchemy Session
        skip_api: API 호출 스킵 여부
        skip_llm: LLM 호출 스킵 여부

    Returns:
        EnrichResult 리스트 (입력과 같은 순서)
    """
    results = []

    for i, track in enumerate(tracks):
        logger.info(
            f"[QLTY Batch] [{i+1}/{len(tracks)}] "
            f"'{track.artist} - {track.title}'"
        )
        result = await enrich_track(
            track, db=db, skip_api=skip_api, skip_llm=skip_llm
        )
        results.append(result)

    # 배치 요약
    total_features = sum(len(r.features) for r in results)
    complete = sum(1 for r in results if r.is_complete)
    logger.info(
        f"[QLTY Batch] 완료: {len(tracks)}곡 처리, "
        f"{complete}곡 완전, "
        f"평균 {total_features/len(tracks):.1f} 피처/곡"
    )

    return results


def reload_dl_model() -> bool:
    """
    DL 모델을 강제 재로드한다.
    학습 후 새 모델을 적용할 때 호출.

    Returns:
        성공 여부
    """
    global _dl_model
    _dl_model = None  # 캐시 무효화
    model = _get_dl_model()
    return model is not None
