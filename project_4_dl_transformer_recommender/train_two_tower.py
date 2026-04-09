import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import json
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from torch.utils.data import Dataset, DataLoader


# ЗАГРУЗКА
user_data = pd.read_parquet("user_data.parquet")
post_embeddings = pd.read_parquet("post_embeddings.parquet")
feed_data = pd.read_parquet("feed_data.parquet")

# позитивные
pos_df = feed_data[feed_data["target"] == 1].copy()

# NEGATIVE SAMPLING
print("Генерация negative sampling...")

neg_samples = []
all_posts = post_embeddings["post_id"].values

for user_id in pos_df["user_id"].unique():
    user_pos = pos_df[pos_df["user_id"] == user_id]["post_id"].values

    neg_posts = np.random.choice(all_posts, size=len(user_pos), replace=False)

    for p in neg_posts:
        if p not in user_pos:
            neg_samples.append((user_id, p, 0))

neg_df = pd.DataFrame(neg_samples, columns=["user_id", "post_id", "target"])

data = pd.concat([pos_df[["user_id", "post_id", "target"]], neg_df])

data = data.merge(user_data, on="user_id")
data = data.merge(post_embeddings, on="post_id")

user_cols = [c for c in user_data.columns if c != "user_id"]
post_cols = [c for c in post_embeddings.columns if c != "post_id"]

X_user = data[user_cols].copy()
X_post = data[post_cols].copy()
y = data["target"]


# ENCODING
encoders = {}

for col in user_cols:
    if X_user[col].dtype == "object":
        le = LabelEncoder()
        X_user[col] = le.fit_transform(X_user[col].astype(str))
        encoders[col] = le.classes_.tolist()

with open("user_encoders.json", "w") as f:
    json.dump(encoders, f)


# SPLIT
X_user_train, X_user_test, X_post_train, X_post_test, y_train, y_test = train_test_split(
    X_user, X_post, y, test_size=0.2, random_state=42
)


# датасет
class RecDataset(Dataset):
    def __init__(self, X_user, X_post, y):
        self.X_user = torch.tensor(X_user.values, dtype=torch.float32)
        self.X_post = torch.tensor(X_post.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_user[idx], self.X_post[idx], self.y[idx]


train_loader = DataLoader(RecDataset(X_user_train, X_post_train, y_train),
                          batch_size=1024, shuffle=True)

test_loader = DataLoader(RecDataset(X_user_test, X_post_test, y_test),
                         batch_size=1024, shuffle=False)

# -----------------------------
# MODEL
# -----------------------------
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

    def forward(self, user_x, post_x):
        user_vec = self.user_tower(user_x)
        post_vec = self.post_tower(post_x)

        # cosine similarity
        user_vec = nn.functional.normalize(user_vec, dim=1)
        post_vec = nn.functional.normalize(post_vec, dim=1)

        score = torch.sum(user_vec * post_vec, dim=1)
        return torch.sigmoid(score)


model = TwoTowerModel(len(user_cols), len(post_cols))

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# -----------------------------
# TRAIN LOOP
# -----------------------------
EPOCHS = 100

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for user_x, post_x, target in train_loader:
        optimizer.zero_grad()

        preds = model(user_x, post_x)
        loss = criterion(preds, target)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # -----------------------------
    # EVAL
    # -----------------------------
    model.eval()
    preds_list = []
    targets_list = []

    with torch.no_grad():
        for user_x, post_x, target in test_loader:
            preds = model(user_x, post_x)
            preds_list.extend(preds.numpy())
            targets_list.extend(target.numpy())

    ap = average_precision_score(targets_list, preds_list)

    print(f"Epoch {epoch}: loss={total_loss:.4f}, AP={ap:.4f}")

# сохраняем
torch.save(model.state_dict(), "two_tower_model.pt")

print("Модель обучена СТАБИЛЬНО")