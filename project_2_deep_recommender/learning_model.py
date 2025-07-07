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




import pandas as pd
from sqlalchemy import create_engine


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

def load_features() -> pd.DataFrame:
    df_users = batch_load_sql('select * from final_user_sadyhov')
    df_posts = batch_load_sql('select * from final_post_sadyhov_neuro')
    return df_users, df_posts


df_users, df_posts = load_features()

df_0 = pd.merge(feed_data, df_users, 'left', 'user_id')
features = pd.merge(df_0, df_posts, 'left', 'post_id')


features = features.drop(columns=['vector','timestamp','action','text','user_id'])
features = features.set_index('post_id')

train = features.iloc[:-1000000].copy()
test = features.iloc[-1000000:].copy()

X_train = train.drop('target', axis=1)
X_test = test.drop('target', axis=1)

y_train = train['target']
y_test = test['target']

from catboost import CatBoostClassifier, Pool

catboost_neuro = CatBoostClassifier(iterations=500, learning_rate=0.1, depth=5, random_seed=42)

cat_features = ['country', 'city', 'os', 'source', 'topic']

catboost_neuro.fit(X_train,
             y_train,
             cat_features=cat_features
             )
from sklearn.metrics import mean_squared_error

y_pred = catboost_neuro.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

catboost_neuro.save_model("catboost_neuro.cbm")