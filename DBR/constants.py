"""
DBR Constants — Divergence-Based Routing 임계값 및 피처 집합

벤치마크 기반 튜닝 결과:
- M1 단독: MNE 0.1116
- DBR 적용: MNE 0.1030 (+7.7%)
"""

# QLTY(Gemini+Search)가 M1보다 강한 장르 (승률 60%+)
QLTY_GENRES = {
    "pop", "rock", "afrobeats", "gaming",
    "reggae", "brazilian", "folk", "classical",
}

# QLTY가 M1보다 항상 나은 피처
QLTY_FEATURES = {"tempo", "acousticness"}

# Divergence routing 대상 피처 (0~1 스케일 + tempo + loudness)
ROUTING_FEATURES = {
    "danceability", "energy", "valence", "tempo",
    "acousticness", "instrumentalness", "speechiness", "loudness",
}

# Divergence 임계값
DIV_CRITICAL = 0.16   # 초과 시 M1 (QLTY 17% 승률)
DIV_MID = 0.10        # 0.05~0.10 → 가중 평균 (M1:60 QLTY:40)
DIV_LOW = 0.05        # 미만 → 단순 평균 (두 모델 합의)

# 인기도 임계값
POP_HIGH = 80         # 이상 → 가중 평균 적용
