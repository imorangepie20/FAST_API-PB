"""
데이터베이스 연결 모듈
MariaDB/MySQL 연결 설정
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base
import os

# 환경 변수에서 DB 설정 읽기
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_NAME = os.getenv("DB_NAME", "music_space")
DB_USER = os.getenv("DB_USER", "musicadmin")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")

# Docker 환경에서는 host.docker.internal 사용
if os.getenv("ENVIRONMENT") == "production" or os.getenv("DOCKER_ENV"):
    DB_HOST = os.getenv("DB_HOST", "host.docker.internal")

DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# SQLAlchemy 엔진 생성
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=3600,
    echo=False
)

# 세션 팩토리
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base 클래스
Base = declarative_base()


def get_db():
    """FastAPI Dependency: DB 세션 제공"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def test_connection():
    """DB 연결 테스트"""
    try:
        with engine.connect() as conn:
            conn.execute("SELECT 1")
        return True
    except Exception as e:
        print(f"DB 연결 실패: {e}")
        return False
