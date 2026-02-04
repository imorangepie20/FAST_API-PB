import os
import glob
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Any

from fastapi import FastAPI, APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy import text
from scipy.spatial.distance import cdist
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split

# [주의] 이 부분은 실제 프로젝트 환경에 맞게 DB 엔진 설정을 가져와야 합니다.
# 만약 별도의 database.py가 있다면 아래 주석을 풀고 사용하세요.
# from app.database import engine 
# 여기서는 예시를 위해 engine 객체가 이미 정의되어 있다고 가정하거나 로컬 설정을 참조합니다.

app = FastAPI(title="Music Recommendation API")

# --- [1. Pydantic Models] ---
class AnalysisRequest(BaseModel):
    userid: str

class AnalysisResponse(BaseModel):
    status: str
    message: str

# --- [2. Configuration & Globals] ---
# 현재 파일 위치 기준으로 경로 설정
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 기존 경로 로직을 유지하되, 통합 파일 위치에 따라 BASE_DIR 조정 필요
BASE_DIR = os.path.dirname(CURRENT_DIR) 

MODEL_DIR = os.path.join(BASE_DIR, 'model')
DATASET_PATH = os.path.join(BASE_DIR, 'dataset', 'dataset.csv')

# 전역 변수 초기화
model = CatBoostRegressor()
df = None
features = ['artists', 'album_name', 'track_genre']
target_columns = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness']

# --- [3. Internal Functions] ---

def get_latest_model_path(user_id):
    search_pattern = os.path.join(MODEL_DIR, f"recommender_U{user_id}_*.cbm")
    model_files = glob.glob(search_pattern)
    if not model_files:
        return None
    latest_model = max(model_files, key=os.path.getmtime)
    return latest_model

def generate_new_model_path(user_id, playlist_title=""):
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_title = "".join(x for x in playlist_title if x.isalnum())[:10]
    return os.path.join(MODEL_DIR, f"recommender_U{user_id}_{safe_title}_{date_str}.cbm")

def train_user_model(user_id, playlist_title, temp_df):
    global model
    print(f"🚀 [Training] 유저 {user_id} 전용 모델 학습 시작...")
    
    train_df, eval_df = train_test_split(temp_df, test_size=0.3, random_state=42)
    train_pool = Pool(data=train_df[features], label=train_df[target_columns], cat_features=features)
    eval_pool = Pool(data=eval_df[features], label=eval_df[target_columns], cat_features=features)
    
    new_model = CatBoostRegressor(iterations=500, learning_rate=0.05, depth=6, loss_function='MultiRMSE', random_seed=42, verbose=False)
    new_model.fit(train_pool, eval_set=eval_pool, early_stopping_rounds=50)
    
    new_path = generate_new_model_path(user_id, playlist_title)
    os.makedirs(MODEL_DIR, exist_ok=True)
    new_model.save_model(new_path)
    return new_model

def save_gms_to_db(user_id, recommended_tracks, engine):
    print(f"💾 유저 {user_id}의 GMS 플레이리스트 생성 중...")
    try:
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO playlists (user_id, title, description, space_type, status_flag, source_type)
                VALUES (:uid, 'AI 추천 리스트', '최신 모델 기반 추천', 'GMS', 'PTP', 'System')
            """), {"uid": user_id})
            gms_id = conn.execute(text("SELECT LAST_INSERT_ID()")).scalar()
            
            for _, row in recommended_tracks.iterrows():
                metadata = json.dumps({"kaggle_id": row.get('track_id', ''), "genre": row.get('track_genre', 'unknown')})
                conn.execute(text("""
                    INSERT INTO tracks (title, artist, album, external_metadata)
                    VALUES (:title, :artist, :album, :meta)
                """), {
                    "title": row['track_name'], "artist": row['artists'], 
                    "album": row['album_name'], "meta": metadata
                })
                tid = conn.execute(text("SELECT LAST_INSERT_ID()")).scalar()
                conn.execute(text("""
                    INSERT INTO playlist_tracks (playlist_id, track_id)
                    VALUES (:pid, :tid)
                """), {"pid": gms_id, "tid": tid})
        return True
    except Exception as e:
        print(f"🔥 GMS 저장 오류: {e}")
        return False

# --- [4. Core Service Logic] ---

def process_analysis(userid: str):
    global model, df
    # 외부 app.database에서 engine을 가져온다고 가정
    from app.database import engine 
    
    uid = int(userid)
    try:
        if df is None:
            if not os.path.exists(DATASET_PATH):
                return f"데이터셋 파일을 찾을 수 없습니다: {DATASET_PATH}"
            df = pd.read_csv(DATASET_PATH)
            for col in features: df[col] = df[col].fillna('unknown').astype(str)
            df[target_columns] = df[target_columns].fillna(0)

        latest_model_path = get_latest_model_path(uid)
        
        with engine.connect() as conn:
            query = text("""
                SELECT p.title, t.title as track_title, t.artist, t.album, t.external_metadata
                FROM tracks t
                JOIN playlist_tracks pt ON t.track_id = pt.track_id
                JOIN playlists p ON pt.playlist_id = p.playlist_id
                WHERE p.user_id = :uid AND p.space_type = 'PMS'
            """)
            result = conn.execute(query, {"uid": uid}).fetchall()
            
            if not result:
                return "분석할 PMS 데이터가 없습니다."
            
            pms_tracks = pd.DataFrame(result)
            playlist_title = pms_tracks['title'].iloc[0]

        if latest_model_path:
            model.load_model(latest_model_path)
        else:
            model = train_user_model(uid, playlist_title, df)

        def safe_get_genre(meta):
            try: return json.loads(meta).get('genre', 'unknown')
            except: return 'unknown'

        pms_tracks['track_genre'] = pms_tracks['external_metadata'].apply(safe_get_genre)
        pms_input = pms_tracks[['artist', 'album', 'track_genre']].fillna('unknown').astype(str)
        pms_input.columns = features 

        pms_predictions = model.predict(pms_input)
        user_taste_vector = pms_predictions.mean(axis=0)

        ems_features_input = df[features].fillna('unknown').astype(str)
        ems_predictions = model.predict(ems_features_input)

        distances = cdist([user_taste_vector], ems_predictions, metric='euclidean')[0]
        top_indices = np.argsort(distances)[:50]
        recommended_tracks = df.iloc[top_indices].copy()

        if save_gms_to_db(uid, recommended_tracks, engine):
            model_info = os.path.basename(latest_model_path) if latest_model_path else "신규 생성"
            return f"유저 {uid} 분석 및 GMS 생성 완료! (사용 모델: {model_info})"
        else:
            return "GMS 생성 중 오류가 발생했습니다."

    except Exception as e:
        return f"에러 발생: {str(e)}"

# --- [5. API Router] ---
router = APIRouter(tags=["Analysis"])

@router.post("/analyze", response_model=AnalysisResponse, summary="분석 요청 처리")
async def analyze_data(request: AnalysisRequest):
    result_message = process_analysis(request.userid)
    
    if "에러" in result_message or "없습니다" in result_message:
        return {"status": "error", "message": result_message}
        
    return {
        "status": "success",
        "message": result_message
    }

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)