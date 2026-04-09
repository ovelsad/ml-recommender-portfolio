import torch
import pandas as pd
import json
from fastapi import FastAPI
from typing import List
from datetime import datetime
from pydantic import BaseModel
import torch.nn as nn


user_data = pd.read_parquet("user_data.parquet")
post_embeddings = pd.read_parquet("post_embeddings.parquet")
post_text_df = pd.read_parquet("post_text_df.parquet")

with open("user_encoders.json") as f:
    encoders = json.load(f)


# ENCODING

for col, classes in encoders.items():
    mapping = {v: i for i, v in enumerate(classes)}
    user_data[col] = user_data[col].astype(str).map(mapping).fillna(0)

# MODEL
class TwoTowerModel(nn.Module):
    def __init__(self, user_dim, post_dim, emb_dim=64):
        super().__init__()

        self.user_tower = nn.Sequential(
            nn.Linear(user_dim, 128),
            nn.ReLU(),
            nn.Linear(128, emb_dim)
        )

        self.post_tower = nn.Sequential(
            nn.Linear(post_dim, 128),
            nn.ReLU(),
            nn.Linear(128, emb_dim)
        )

    def get_user_embedding(self, user_x):
        user_vec = self.user_tower(user_x)
        return nn.functional.normalize(user_vec, dim=1)

    def get_post_embedding(self, post_x):
        post_vec = self.post_tower(post_x)
        return nn.functional.normalize(post_vec, dim=1)


# признаки
user_cols = [c for c in user_data.columns if c != "user_id"]
post_cols = [c for c in post_embeddings.columns if c != "post_id"]

# модель
model = TwoTowerModel(len(user_cols), len(post_cols))
model.load_state_dict(torch.load("two_tower_model.pt", map_location="cpu"))
model.eval()

post_tensor = torch.tensor(post_embeddings[post_cols].values, dtype=torch.float32)

with torch.no_grad():
    post_vectors = model.get_post_embedding(post_tensor)

post_ids = post_embeddings["post_id"].values


app = FastAPI()

class PostGet(BaseModel):
    id: int
    text: str
    topic: str



# RECOMMENDATION FUNCTION
def posts_recommendation(user_id, n=5):
    df_user = user_data[user_data["user_id"] == user_id]

    if df_user.empty:
        return []

    user_tensor = torch.tensor(df_user[user_cols].values, dtype=torch.float32)

    with torch.no_grad():
        user_vector = model.get_user_embedding(user_tensor)

        # dot product со всеми постами
        scores = torch.matmul(post_vectors, user_vector.T).squeeze()

    top_idx = torch.topk(scores, k=n).indices.numpy()
    top_post_ids = post_ids[top_idx]

    result = post_text_df[post_text_df["post_id"].isin(top_post_ids)].copy()
    result = result.rename(columns={"post_id": "id"})

    return result[["id", "text", "topic"]].to_dict("records")

### запрос
@app.get("/post/recommendations/", response_model=List[PostGet])
def recommended_posts(
        id: int,
        time: datetime,
        limit: int = 5) -> List[PostGet]:

    return posts_recommendation(user_id=id, n=limit)