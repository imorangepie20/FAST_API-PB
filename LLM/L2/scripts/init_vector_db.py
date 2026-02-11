"""
벡터 DB 초기화 스크립트

114k Spotify 데이터셋을 ChromaDB에 로드하여 자연어 검색을 가능하게 합니다.
아티스트 태그(lfm_artist_tags)를 벡터화하여 의미 기반 검색을 수행합니다.

사용법:
    python scripts/init_vector_db.py
    python scripts/init_vector_db.py --batch-size 500
    python scripts/init_vector_db.py --clear  # 기존 데이터 삭제 후 재초기화
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from time import time

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 데이터 파일 경로
DATA_DIR = project_root / "data"
SPOTIFY_CSV = DATA_DIR / "spotify_114k_with_tags.csv"


def load_spotify_data() -> pd.DataFrame:
    """Spotify 114k 데이터셋 로드"""
    if not SPOTIFY_CSV.exists():
        logger.error(f"데이터 파일을 찾을 수 없습니다: {SPOTIFY_CSV}")
        sys.exit(1)

    logger.info(f"데이터 로드 중: {SPOTIFY_CSV}")
    df = pd.read_csv(SPOTIFY_CSV)

    # 필수 컬럼 확인
    required_cols = ["track_id", "artists", "track_name", "lfm_artist_tags", "track_genre"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"필수 컬럼 누락: {missing_cols}")
        sys.exit(1)

    logger.info(f"총 {len(df):,}곡 로드 완료")

    # 중복 track_id 제거 (첫 번째 것만 유지)
    original_count = len(df)
    df = df.drop_duplicates(subset=["track_id"], keep="first")
    duplicate_count = original_count - len(df)
    if duplicate_count > 0:
        logger.warning(f"중복 track_id {duplicate_count:,}개 제거됨 → {len(df):,}곡")

    return df


def init_vector_db(batch_size: int = 500, clear_existing: bool = False):
    """
    벡터 DB 초기화

    Args:
        batch_size: 배치 처리 크기 (ChromaDB 성능 최적화)
        clear_existing: True면 기존 데이터 삭제 후 재초기화
    """
    from app.services.llm.vector_search import (
        get_vector_search_service,
        VectorSearchService
    )

    start_time = time()

    # 데이터 로드
    df = load_spotify_data()

    # 벡터 서비스 초기화
    logger.info("VectorSearchService 초기화 중...")
    service = get_vector_search_service()

    if not service.is_ready:
        logger.error("VectorSearchService 초기화 실패")
        sys.exit(1)

    logger.info(f"현재 컬렉션 크기: {service.collection_count:,}")

    # 기존 데이터 삭제 옵션
    if clear_existing and service.collection_count > 0:
        logger.warning("기존 컬렉션 삭제 중...")
        service.clear_collection()
        logger.info("컬렉션 초기화 완료")

    # 이미 데이터가 있으면 스킵 옵션 제공
    if service.collection_count > 0 and not clear_existing:
        logger.info(f"이미 {service.collection_count:,}개의 문서가 존재합니다.")
        response = input("계속 추가하시겠습니까? (y/N): ").strip().lower()
        if response != 'y':
            logger.info("초기화 취소됨")
            return

    # 태그가 있는 곡만 필터링
    df_with_tags = df[df["lfm_artist_tags"].notna() & (df["lfm_artist_tags"] != "")]
    logger.info(f"태그가 있는 곡: {len(df_with_tags):,}곡 ({len(df_with_tags)/len(df)*100:.1f}%)")

    # 배치 처리
    total = len(df_with_tags)
    success_count = 0
    fail_count = 0

    logger.info(f"벡터 DB에 {total:,}곡 추가 시작 (배치 크기: {batch_size})")

    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch_df = df_with_tags.iloc[batch_start:batch_end]

        # 배치 데이터 준비
        ids = []
        documents = []
        metadatas = []

        for _, row in batch_df.iterrows():
            track_id = str(row["track_id"])
            artist = str(row["artists"]) if pd.notna(row["artists"]) else "Unknown"
            title = str(row["track_name"]) if pd.notna(row["track_name"]) else "Unknown"
            tags = str(row["lfm_artist_tags"]) if pd.notna(row["lfm_artist_tags"]) else ""
            genre = str(row["track_genre"]) if pd.notna(row["track_genre"]) else ""

            # 검색용 텍스트 생성
            search_text = f"{genre}, {tags}" if genre else tags
            if not search_text.strip():
                continue

            # 메타데이터
            metadata = {
                "artist": artist[:500],  # ChromaDB 메타데이터 크기 제한
                "title": title[:500],
                "tags": tags[:1000],
                "genre": genre[:100],
            }

            # 오디오 피처 추가 (있으면)
            for feature in ["energy", "valence", "acousticness", "danceability", "tempo"]:
                if feature in row and pd.notna(row[feature]):
                    metadata[feature] = float(row[feature])

            ids.append(track_id)
            documents.append(search_text)
            metadatas.append(metadata)

        # 배치 upsert
        if ids:
            try:
                service.collection.upsert(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas
                )
                success_count += len(ids)
            except Exception as e:
                logger.error(f"배치 {batch_start//batch_size + 1} 실패: {e}")
                fail_count += len(ids)

        # 진행 상황 출력
        progress = (batch_end / total) * 100
        elapsed = time() - start_time
        eta = (elapsed / batch_end) * (total - batch_end) if batch_end > 0 else 0

        logger.info(
            f"진행: {batch_end:,}/{total:,} ({progress:.1f}%) | "
            f"성공: {success_count:,} | 실패: {fail_count:,} | "
            f"경과: {elapsed:.1f}s | ETA: {eta:.1f}s"
        )

    # 완료 보고
    total_time = time() - start_time
    logger.info("=" * 60)
    logger.info("벡터 DB 초기화 완료!")
    logger.info(f"총 처리: {success_count + fail_count:,}곡")
    logger.info(f"성공: {success_count:,}곡")
    logger.info(f"실패: {fail_count:,}곡")
    logger.info(f"최종 컬렉션 크기: {service.collection_count:,}")
    logger.info(f"총 소요 시간: {total_time:.1f}초 ({total_time/60:.1f}분)")
    logger.info("=" * 60)


def test_search():
    """초기화 후 검색 테스트"""
    from app.services.llm.vector_search import get_vector_search_service

    service = get_vector_search_service()

    if not service.is_ready:
        logger.error("서비스 준비 안됨")
        return

    test_queries = [
        "비오는 날 카페에서 들을 잔잔한 재즈",
        "신나는 댄스 파티 음악",
        "우울할 때 듣는 감성적인 발라드",
        "운동할 때 듣는 힙합",
        "집중력 높이는 클래식 피아노"
    ]

    logger.info("\n" + "=" * 60)
    logger.info("검색 테스트")
    logger.info("=" * 60)

    for query in test_queries:
        logger.info(f"\n쿼리: {query}")
        results = service.search(query, n_results=3)

        for i, track in enumerate(results, 1):
            logger.info(
                f"  {i}. {track['artist']} - {track['title']} "
                f"(유사도: {track['similarity_score']:.3f}, 장르: {track['genre']})"
            )


def main():
    parser = argparse.ArgumentParser(description="벡터 DB 초기화 스크립트")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="배치 처리 크기 (기본: 500)"
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="기존 데이터 삭제 후 재초기화"
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="검색 테스트만 실행"
    )

    args = parser.parse_args()

    if args.test_only:
        test_search()
    else:
        init_vector_db(
            batch_size=args.batch_size,
            clear_existing=args.clear
        )
        # 초기화 후 테스트
        test_search()


if __name__ == "__main__":
    main()
