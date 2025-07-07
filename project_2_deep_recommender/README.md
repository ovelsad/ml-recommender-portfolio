## 🔎 `project_2_deep_recommender/README.md`


# 🤖 Project 2 — Улучшенная рекомендательная система с нейросетями

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

