## 🔎 `project_4_dl_transformer_recommender/README.md`

# Project 4 — Рекомендательная система на Two-Tower + Transformer embeddings

## Задача
Построение современной рекомендательной системы с использованием deep learning подхода:
- кодирование текстов постов через трансформер
- обучение two-tower модели (user tower + post tower)
- быстрый inference через векторное сходство

## Методы
- SentenceTransformer (`all-MiniLM-L6-v2`) для эмбеддингов текста
- Two-Tower Neural Network (PyTorch)
- Negative Sampling
- Cosine Similarity (через нормализацию эмбеддингов)

## Метрика
- Average Precision (AP) на тесте

## Файлы
- `load_data_dl.py` — загрузка данных из БД и сохранение в parquet
- `build_embeddings.py` — генерация эмбеддингов постов через transformer
- `train_two_tower.py` — обучение two-tower модели
- `app_dl.py` — FastAPI сервис для рекомендаций

## 🚀 Запуск проекта

### 1. Клонировать репозиторий

```
git clone https://github.com/ovelsad/ml-recommender-portfolio.git
cd your_repo/project_4_dl_transformer_recommender
```

### 2. Установить зависимости
```
pip install -r requirements.txt
```

### 3. Загрузка данных из БД
```
python load_data_dl.py
```
Сохранит:
* user_data.parquet
* post_text_df.parquet
* feed_data.parquet

### 4. Построение эмбеддингов постов
```
python build_embeddings.py
```
Сохранит:
* post_embeddings.parquet

### 5. Обучение модели
```
python train_two_tower.py
```
После обучения появятся:
* two_tower_model.pt
* user_encoders.json

### 6. Запуск API
```
uvicorn app_dl:app --reload
```

### 7. Пример запроса
```
http://127.0.0.1:8000/post/recommendations/?id=200&time=2024-01-01T00:00:00&limit=5
```

Параметры:
* id — id пользователя
* time — datetime (формат ISO)
* limit — количество рекомендаций
