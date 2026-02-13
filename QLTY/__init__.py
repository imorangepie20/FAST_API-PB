"""
QLTY - Audio Feature Enrichment Pipeline

불완전한 트랙 데이터의 오디오 특성을 채워넣는 다층 파이프라인.

우선순위:
1. ReccoBeats API (ISRC 기반, 실제 오디오 분석)
2. spotify_reference DB 매칭 (title+artist)
3. Gemini + Google Search (웹 검색 기반 조회/추정)
4. DL 모델 예측 (LightGBM, 최후 수단)
"""
