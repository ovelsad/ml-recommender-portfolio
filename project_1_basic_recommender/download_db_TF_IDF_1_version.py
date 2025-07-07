import os
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
svd_df = pd.DataFrame(reduced_tfidf, columns=[f'tf_{i+1}' for i in range(n_components)])  # Новый DataFrame с компонентами

# Объединение с оригинальным DataFrame, исключая текст
final_post_text_df = pd.concat([post_text_df.drop(columns='text'), svd_df], axis=1)  # Объединение DataFrame


final_post_sadyhov = pd.concat([post_text_df, svd_df], axis=1)

# Проверка структуры
print(final_post_sadyhov.head())
print(final_post_sadyhov.info())

# Убедитесь, что все строки имеют корректные значения
assert not final_post_sadyhov.isnull().any().any(), "Есть пропущенные значения в данных!"

# Настройки подключения к PostgreSQL
DATABASE_URL = "postgresql://robot-startml-ro:pheiph0hahj1Vaif@postgres.lab.karpov.courses:6432/startml"
engine = create_engine(DATABASE_URL)

# Загрузка данных в PostgreSQL
table_name = "final_post_sadyhov"
final_post_sadyhov.to_sql(
    table_name,
    engine,
    if_exists="replace",  # Заменяет таблицу, если она существует
    index=False,          # Индекс из DataFrame не сохраняется
)
print(f"Таблица {table_name} успешно загружена!")


table_name_2 = "final_user_sadyhov"
user_data.to_sql(
    table_name_2,
    engine,
    if_exists="replace",  # Заменяет таблицу, если она уже существует
    index=False,          # Не сохраняет индекс из DataFrame как отдельный столбец
)