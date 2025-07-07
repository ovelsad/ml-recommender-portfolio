import hashlib
import os
from datetime import datetime
from typing import List
import pandas as pd
from catboost import CatBoostClassifier
from fastapi import FastAPI
from pydantic import BaseModel
from sqlalchemy import create_engine



def get_model_path(path: str, name: str) -> str:
    if os.environ.get("IS_LMS") == "1":
        MODEL_PATH = '/workdir/user_input/' + name
    else:
        MODEL_PATH = path
    return MODEL_PATH

def load_models(model_name: str):
    if model_name == 'model_control':
        model_path = get_model_path("model_control.cbm", model_name)
    else:
        model_path = get_model_path("model_test.cbm", model_name)
    model = CatBoostClassifier(iterations=500, learning_rate=0.1, depth=5, random_seed=42)

    return model.load_model(model_path)



model_control = load_models("model_control")
model_test = load_models("model_test")


def batch_load_sql(query: str) -> pd.DataFrame:
    CHUNKSIZE = 200000  # Размер чанков для загрузки
    engine = create_engine(
        "postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
        "postgres.lab.karpov.courses:6432/startml"
    )
    conn = engine.connect().execution_options(stream_results=True)  # Потоковая загрузка
    chunks = []
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=CHUNKSIZE):
        chunks.append(chunk_dataframe)
    conn.close()
    return pd.concat(chunks, ignore_index=True)


### ЗАГРУЖАЕМ ТАБЛИЦЫ ЮЗЕРОВ И ПОСТОВ(С TF-IDF)
def load_features() -> pd.DataFrame:
    df_users = batch_load_sql('select * from final_user_sadyhov')
    df_posts = batch_load_sql('select * from final_post_sadyhov_neuro')
    return df_users, df_posts


df_users, df_posts = load_features()
# print(df_users.columns)
# print(df_posts.columns)


salt = 'final_salt'  # соль для хэширования
split_const = 50  # 50/50


def get_exp_group(user_id: int) -> str:
    user = str(user_id)
    value_str = user + salt
    value_num = int(hashlib.md5(value_str.encode()).hexdigest(), 16)
    if value_num % 100 < split_const:
        return "control"
    else:
        return "test"




# функция для обработки датафрейма в предикты и ответы
def posts_recommendation(user_id, n=5):
    '''
    :param user_id:-> int принимает id пользователя
    n:-> int, количество возвращаемых постов
    :return: список с рекомендациями id постов
    '''

    # Выбираем модель в зависимости от группы
    exp_group = get_exp_group(user_id)

    if exp_group == "control":
        model = model_control
    else:
        model = model_test

    ### 1. Фильтруем DataFrame пользователей по user_id
    df = df_users[df_users['user_id'] == user_id]

    ### 2. Выполняем кросс-слияние с постами без текста
    df = df.merge(df_posts.drop(columns='text'), how='cross')

    ### 3.Удаляем user_id
    df = df.drop(columns='user_id')

    ### 4. Устанавливаем post_id в качестве индекса
    df = df.set_index('post_id')

    ### 5. Выбираем для предсказания столбцы
    # print(df.columns[0:20])
    df = df.drop(columns='vector')

    ### 6. Предсказываем вероятности для каждого поста
    df['predict_proba'] = model.predict_proba(df)[:, 1]

    ### 7. Сортируем по вероятностям и выбираем топ-n постов
    df = df.sort_values(by='predict_proba', ascending=False).head(n)

    ### 8. возвращаем post_id в столбец
    df = df.reset_index()

    ### 9. Объединяем с полными данными постов для получения текста и темы и удаляем дублирующиеся столбцы
    df = pd.merge(df, df_posts.drop(['topic', 'text_length', '!_count', '?_count', 'tf_1', 'tf_2',
                                     'tf_3', 'tf_4', 'tf_5'], axis=1), 'inner', 'post_id')

    ### 10. Оставляем только нужные столбцы для результата
    df = df.rename(columns={'post_id': 'id'})  # Переименовываем 'post_id' в 'id'
    df = df[['id', 'text', 'topic']]

    ### Преобразуем DataFrame в список словарей
    list_of_dicts = df.to_dict('records')

    return list_of_dicts


app = FastAPI()


class PostGet(BaseModel):
    id: int
    text: str
    topic: str

    class Config:
        orm_mode = True


class Response(BaseModel):
    exp_group: str
    recommendations: List[PostGet]


@app.get("/post/recommendations/", response_model=Response)
def recommended_posts(
        id: int,
        time: datetime,
        limit: int = 5) -> List[PostGet]:
    exp_group = get_exp_group(id)

    recommendations = posts_recommendation(id, limit)

    return Response(exp_group=exp_group, recommendations=[
        PostGet(**post) for post in recommendations
    ])

