import pandas as pd

from sqlalchemy import create_engine
engine = create_engine(
    "postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
    "postgres.lab.karpov.courses:6432/startml")


user_data = pd.read_sql(
    """ SELECT * FROM user_data""",
    con = engine
)

post_text_df = pd.read_sql(
    """ SELECT * FROM post_text_df""",
    con = engine
)

feed_data = pd.read_sql(
    'SELECT * FROM public.feed_data limit 5000000; --используем лимит тк данных очень много',
    con=engine
)

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# Добавление текстовых признаков
post_text_df['text_length'] = post_text_df['text'].str.len()  # Длина текста
post_text_df['!_count'] = post_text_df['text'].str.count('!')  # Количество восклицательных знаков
post_text_df['?_count'] = post_text_df['text'].str.count('\\?')  # Количество вопросительных знаков

# Преобразование текста в матрицу TF-IDF
max_features = 32  # Максимальное количество признаков
vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))  # Установка векторизатора
tfidf_matrix = vectorizer.fit_transform(post_text_df['text'])  # Преобразование текста

# Уменьшение размерности матрицы TF-IDF
n_components = 5  # Количество компонентов для SVD
svd = TruncatedSVD(n_components=n_components, random_state=41)  # Инициализация SVD
reduced_tfidf = svd.fit_transform(tfidf_matrix)  # Уменьшение размерности

# Создание нового DataFrame из SVD
svd_df = pd.DataFrame(reduced_tfidf, columns=[f'tf_{i+1}' for i in range(n_components)])
# Объединение с оригинальным DataFrame
final_post_text_df = pd.concat([post_text_df, svd_df], axis=1)


final_post_text_df


import pandas as pd
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk

nltk.download("punkt")

# Токенизируем тексты
tokenized_texts = final_post_text_df["text"].apply(lambda x: word_tokenize(x.lower()))

# Обучаем модель Skip-gram
model = Word2Vec(sentences=tokenized_texts, vector_size=100, window=5, min_count=1, sg=1)  # sg=1 — это Skip-gram

# Сохраняем модель
model.save("skipgram_model.w2v")

# Проверяем вектор слова
print(model.wv["uk"])


import numpy as np

# Функция усреднения векторов слов в тексте
def text_vector(text, model):
    words = word_tokenize(text.lower())
    vectors = [model.wv[word] for word in words if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

# Применяем к каждому тексту
final_post_text_df["vector"] = final_post_text_df["text"].apply(lambda x: text_vector(x, model))

print(final_post_text_df.head())


# Объединение таблиц
final_post_sadyhov = final_post_text_df

import json
vector_df = pd.DataFrame(final_post_text_df["vector"].to_list(),
                         columns=[f'w2v_{i}' for i in range(100)])

final_post_sadyhov_neuro = pd.concat(
    [final_post_text_df.drop(columns=["vector"]), vector_df],
    axis=1
)

from sqlalchemy import create_engine


# Убедитесь, что все строки имеют корректные значения
assert not final_post_sadyhov_neuro.isnull().any().any(), "Есть пропущенные значения в данных!"

# Настройки подключения к PostgreSQL
DATABASE_URL = "postgresql://robot-startml-ro:pheiph0hahj1Vaif@postgres.lab.karpov.courses:6432/startml"
engine = create_engine(DATABASE_URL)

# Загрузка данных в PostgreSQL
table_name_1 = "final_post_sadyhov_neuro"
final_post_sadyhov_neuro.to_sql(
    table_name_1,
    engine,
    if_exists="replace",  # Заменяет таблицу, если она существует
    index=False,          # Индекс из DataFrame не сохраняется
)
print(f"Таблица {table_name_1} успешно загружена!")


final_user_sadyhov=user_data
print(final_user_sadyhov.head())
print(final_user_sadyhov.info())
table_name_2 = "final_user_sadyhov"
user_data.to_sql(
    table_name_2,
    engine,
    if_exists="replace",  # Заменяет таблицу, если она уже существует
    index=False,          # Не сохраняет индекс из DataFrame как отдельный столбец
)