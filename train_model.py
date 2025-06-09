# train_model.py

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import joblib
import os
import re

# ------------------------------
# Step 1: Load and Clean Nutrition Data
# ------------------------------

def clean_nutrition_data(df):
    df = df.copy()
    df = df.drop(columns=["name", "serving_size"], errors="ignore")
    for col in df.columns:
        df[col] = df[col].astype(str).str.extract(r'([\d.]+)').astype(float)
    df = df.dropna(axis=1, how="all")
    df = df.dropna()
    return df

df_raw = pd.read_csv("dataset/nutrition.csv")
df_clean = clean_nutrition_data(df_raw)
df_raw_filtered = df_raw.loc[df_clean.index]  # Keep original names aligned

# Normalize food features
food_scaler = StandardScaler()
food_features = food_scaler.fit_transform(df_clean.values)

# ------------------------------
# Step 2: Generate User Profiles
# ------------------------------

def generate_user_profiles(n):
    profiles = []
    for _ in range(n):
        age = np.random.randint(18, 60)
        weight = np.random.randint(50, 100)
        height = np.random.randint(150, 200)
        gender = np.random.choice([0, 1])
        activity = np.random.choice([0, 1, 2])
        goal = np.random.choice([0, 1, 2])
        profiles.append([age, weight, height, gender, activity, goal])
    return np.array(profiles)

user_profiles = generate_user_profiles(1000)
user_scaler = StandardScaler()
user_profiles_scaled = user_scaler.fit_transform(user_profiles)

# ------------------------------
# Step 3: Create Training Pairs
# ------------------------------

positive_pairs = []
negative_pairs = []

for user in user_profiles_scaled:
    for _ in range(5):
        food_idx = np.random.randint(len(food_features))
        calories = food_features[food_idx][0]
        goal = user[-1]

        is_positive = (
            (goal == 0 and calories < 0) or
            (goal == 2 and calories > 0) or
            (goal == 1 and abs(calories) < 0.5)
        )

        label = 1 if is_positive else 0
        (positive_pairs if label else negative_pairs).append((user, food_features[food_idx], label))

pairs = positive_pairs + negative_pairs
np.random.shuffle(pairs)

# ------------------------------
# Step 4: Dataset and Dataloader
# ------------------------------

class FoodDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        user, food, label = self.pairs[idx]
        return torch.tensor(user, dtype=torch.float32), torch.tensor(food, dtype=torch.float32), torch.tensor([label], dtype=torch.float32)

dataset = FoodDataset(pairs)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# ------------------------------
# Step 5: Model Definition
# ------------------------------

class Recommender(nn.Module):
    def __init__(self, user_dim, food_dim):
        super().__init__()
        self.user_net = nn.Sequential(
            nn.Linear(user_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )
        self.food_net = nn.Sequential(
            nn.Linear(food_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )
        self.output = nn.CosineSimilarity(dim=1)

    def forward(self, user, food):
        u = self.user_net(user)
        f = self.food_net(food)
        return self.output(u, f).unsqueeze(1)

model = Recommender(user_dim=6, food_dim=food_features.shape[1])
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ------------------------------
# Step 6: Train the Model
# ------------------------------

for epoch in range(10):
    total_loss = 0
    for user, food, label in dataloader:
        optimizer.zero_grad()
        output = model(user, food)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# ------------------------------
# Step 7: Save Model and Scalers
# ------------------------------

os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/food_recommender.pt")
joblib.dump(food_scaler, "models/food_scaler.pkl")
joblib.dump(user_scaler, "models/user_scaler.pkl")
df_raw_filtered.to_csv("models/cleaned_food.csv", index=False)
