import pandas as pd
from sentence_transformers import SentenceTransformer

# загружаем данные
post_text_df = pd.read_parquet("post_text_df.parquet")

# модель трансформера
model = SentenceTransformer("all-MiniLM-L6-v2")

# кодируем тексты
embeddings = model.encode(
    post_text_df["text"].tolist(),
    batch_size=64,
    show_progress_bar=True
)

# превращаем в DataFrame
emb_df = pd.DataFrame(
    embeddings,
    columns=[f"emb_{i}" for i in range(embeddings.shape[1])]
)

# финальная таблица
post_embeddings = pd.concat(
    [post_text_df[["post_id"]], emb_df],
    axis=1
)

# сохраняем
post_embeddings.to_parquet("post_embeddings.parquet", index=False)

print("Эмбеддинги сохранены")