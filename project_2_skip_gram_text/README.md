## 🔎 `project_2_deep_recommender/README.md`


# 🤖 Project 2 — Улучшенная рекомендательная система с нейросетевым методом.

## Задача
Улучшение модели рекомендаций с использованием векторизации текста через Word2Vec и дальнейшее обучение модели.

## Методы
- TF-IDF + SVD
- Word2Vec (Skip-gram) со средними векторами
- CatBoost с дополнительными признаками

## Метрика
- Hitrate@5 > 0.57

## Файлы
- `load_db_tf_idf__word2vec.py` — генерация TF-IDF и word2vec векторов
- `ml_dl_app_2_version.py` — FastAPI сервис с новой моделью `catboost_neuro.cbm`

##  Запуск проекта

### 1. Клонировать репозиторий


```git clone https://github.com/ovelsad/ml-recommender-portfolio.git```

```cd your_repo/project_2_skip_gram_text bash```

### 2. установить зависимости

``` pip install -r requirements.txt ```

### 3. Подготовка данных

Загрузка и обработка данных:

``` python load_db_tf_idf__word2vec.py ```

### 4. Обучение модели
```python learning_model.py```

После этого появятся:

* ```catboost_neuro.cbm```
* ```feature_columns.json```

### 5. Запуск API

```uvicorn ml_dl_app_2_version:app --reload```


### Пример запроса:

```http://127.0.0.1:8000/post/recommendations/?id=200&time=2024-01-01T00:00:00&limit=5```

Параметры:
* id — id пользователя
* time — datetime (формат ISO)
* limit — количество рекомендаций