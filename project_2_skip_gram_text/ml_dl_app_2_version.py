import os
import json
import pandas as pd
from sqlalchemy import create_engine
from pydantic import BaseModel
from fastapi import FastAPI
from typing import List
from datetime import datetime
from catboost import CatBoostClassifier


def get_model_path(path: str) -> str:
    if os.environ.get("IS_LMS") == "1":
        MODEL_PATH = '/workdir/user_input/model'
    else:
        MODEL_PATH = path
    return MODEL_PATH


def load_models():
    model_path = get_model_path("catboost_neuro.cbm")
    model = CatBoostClassifier()
    model.load_model(model_path)
    return model


def load_feature_columns():
    with open("feature_columns.json") as f:
        return json.load(f)


def batch_load_sql(query: str) -> pd.DataFrame:
    CHUNKSIZE = 200000
    engine = create_engine(
        "postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
        "postgres.lab.karpov.courses:6432/startml"
    )
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=CHUNKSIZE):
        chunks.append(chunk_dataframe)
    conn.close()
    return pd.concat(chunks, ignore_index=True)


# загружаем пользователей и посты
def load_features():
    df_users = batch_load_sql('select * from final_user_sadyhov')
    df_posts = batch_load_sql('select * from final_post_sadyhov_neuro')
    return df_users, df_posts


df_users, df_posts = load_features()
model = load_models()
feature_columns = load_feature_columns()


def posts_recommendation(user_id, n=5):
    # 1. пользователь
    df = df_users[df_users['user_id'] == user_id]

    if df.empty:
        return []

    # 2. берем часть постов (ускорение)
    posts_sample = df_posts.sample(5000)

    # 3. cross join
    df = df.merge(posts_sample.drop(columns='text'), how='cross')

    # 4. удаляем user_id
    df = df.drop(columns='user_id')

    # 5. индекс
    df = df.set_index('post_id')

    # 6. строго те же признаки что при обучении
    df = df[feature_columns]

    # 7. предсказание
    df['predict_proba'] = model.predict_proba(df)[:, 1]

    # 8. топ n
    df = df.sort_values(by='predict_proba', ascending=False).head(n)

    # 9. возвращаем post_id
    df = df.reset_index()

    # 10. подтягиваем текст и правильный topic
    df = pd.merge(
        df,
        df_posts[['post_id', 'text', 'topic']],
        how='inner',
        on='post_id',
        suffixes=('', '_post')
    )

    # 11. фикс topic
    if 'topic_post' in df.columns:
        df = df.rename(columns={'topic_post': 'topic'})

    # 12. финальный формат
    df = df.rename(columns={'post_id': 'id'})
    df = df[['id', 'text', 'topic']]

    return df.to_dict('records')


app = FastAPI()


class PostGet(BaseModel):
    id: int
    text: str
    topic: str

    class Config:
        from_attributes = True


@app.get("/post/recommendations/", response_model=List[PostGet])
def recommended_posts(
        id: int,
        time: datetime,
        limit: int = 5) -> List[PostGet]:
    return posts_recommendation(user_id=id, n=limit)