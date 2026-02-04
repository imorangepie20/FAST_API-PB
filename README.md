# 🤖 AI Music Analysis API (FastAPI + Docker)

EMS (Explore Music Space) 데이터 기반 AI 음악 분석 및 추천 서버

## 📁 프로젝트 구조

```
FAST_API/
├── main.py              # FastAPI 메인 앱
├── requirements.txt     # Python 의존성
├── Dockerfile           # Docker 이미지 설정
├── docker-compose.yml   # Docker Compose 설정
├── .env.example         # 환경 변수 템플릿
├── .dockerignore        # Docker 빌드 제외 파일
│
├── M1/                  # 트랙 분석 모델
│   ├── analysis.py
│   ├── spotify_recommender.py
│   └── audio_predictor.pkl
│
├── M2/                  # 콘텐츠 기반 추천 모델
│   ├── m2.py
│   └── tfidf_gbr_models.pkl
│
└── M3/                  # 협업 필터링 모델
    ├── m3.py
    └── recommender_*.cbm
```

## 🚀 Docker로 실행

### 1. 환경 변수 설정

```bash
copy .env.example .env
# .env 파일 편집하여 실제 값 입력
```

### 2. Docker 빌드 및 실행

```bash
# 빌드 및 실행
docker-compose up --build

# 백그라운드 실행
docker-compose up -d --build

# 로그 확인
docker-compose logs -f ai-api

# 중지
docker-compose down
```

### 3. API 접속

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

## 📡 API 엔드포인트

| Method | Endpoint | 설명 |
|--------|----------|------|
| GET | `/` | API 상태 확인 |
| GET | `/health` | 헬스 체크 |
| POST | `/api/analyze` | 트랙 분석 (M1) |
| POST | `/api/recommend` | 콘텐츠 기반 추천 (M2) |
| POST | `/api/recommend/collaborative` | 협업 필터링 추천 (M3) |
| GET | `/api/score/{track_id}` | 트랙 AI 점수 조회 |
| GET | `/api/ems/analysis` | EMS 데이터 종합 분석 |

## 🔗 EMS API 연동

이 서버는 Node.js 백엔드의 EMS API와 통신합니다.

- **로컬**: `http://host.docker.internal:3001/api/ems`
- **외부**: `https://homological-ashlyn-supercrowned.ngrok-free.dev/api/ems`

## 🤖 ML 모델

| 모델 | 용도 | 파일 |
|------|------|------|
| M1 | 트랙 분석 & Spotify 연동 | `audio_predictor.pkl` |
| M2 | TF-IDF 콘텐츠 기반 추천 | `tfidf_gbr_models.pkl` |
| M3 | CatBoost 협업 필터링 | `recommender_*.cbm` |

## 🛠 개발 모드

코드 변경 시 자동 반영 (volumes 마운트됨):

```bash
docker-compose up
# main.py 수정하면 자동 리로드
```

## 📋 TODO

- [ ] M1/M2/M3 모델 main.py 연동
- [ ] EMS API 클라이언트 구현
- [ ] 실시간 점수 캐싱 (Redis)
- [ ] 배치 분석 스케줄러
