import pandas as pd
from sqlalchemy import create_engine

# подключение к БД
engine = create_engine(
    "postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
    "postgres.lab.karpov.courses:6432/startml"
)

def batch_load_sql(query: str) -> pd.DataFrame:
    CHUNKSIZE = 200000
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []

    for chunk_dataframe in pd.read_sql(query, conn, chunksize=CHUNKSIZE):
        chunks.append(chunk_dataframe)

    conn.close()
    return pd.concat(chunks, ignore_index=True)


if __name__ == "__main__":
    # загружаем данные
    user_data = batch_load_sql("SELECT * FROM user_data")
    post_text_df = batch_load_sql("SELECT * FROM post_text_df")
    feed_data = batch_load_sql("SELECT * FROM feed_data LIMIT 1000000")

    # сохраняем локально (чтобы не дергать БД каждый раз)
    user_data.to_parquet("user_data.parquet", index=False)
    post_text_df.to_parquet("post_text_df.parquet", index=False)
    feed_data.to_parquet("feed_data.parquet", index=False)

    print("Данные сохранены локально")