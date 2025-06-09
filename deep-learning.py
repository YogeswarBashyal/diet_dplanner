# Deep Learning Based Food Recommender System
# Recommends breakfast, lunch, and dinner items based on user input

import pandas as pd
import numpy as np
import torch
import csv
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Step 1: Load and preprocess data
food_data = pd.read_csv("dataset/nutrition.csv")  # Replace with actual file path
bmi_data = pd.read_csv("dataset/bmi.csv")

user_profiles = []
required_columns = ['Gender', 'Height', 'Weight', 'Index']
for idx, row in bmi_data.items():
    # encoding -> Male - 0, Female - 1
    # pre-processing -> age = np.random.randint(18, 160)
    # user = [['Male', 174, 96, 4], ['Female', 180, 95, 2]]
    # user = [[0, 174, 96, 4], [1, 180, 95, 2]]
    age = np.random.randint(18, 60)
    height = row['Gender']
    profile = [age, height]
    user_profiles.append(profile)
print(user_profiles)
# bmi_data -> gender(0,1), age(random(18,60))

# # Drop non-numeric or unnecessary columns
# food_data = food_data.drop(columns=["name", "serving_size"], errors='ignore')
# food_data = food_data.dropna()
#
# # Scale nutrient features
# scaler = StandardScaler()
# food_features = scaler.fit_transform(food_data.values)
#
# # Step 2: Generate synthetic user profiles
# # TODO: return np.arrary(profiles) -> take this data from bmi.csv
# def generate_user_profiles(n):
#     profiles = []
#     for _ in bmi_data:
#         age = np.random.randint(18, 60)
#         weight = np.random.randint(50, 100)
#         height = np.random.randint(150, 190)
#         gender = np.random.choice([0, 1])  # 0 = Male, 1 = Female
#         activity = np.random.choice([0, 1, 2])  # 0 = Sedentary, 1 = Moderate, 2 = Active
#         goal = np.random.choice([0, 1, 2])  # 0 = Weight Loss, 1 = Maintenance, 2 = Gain
#         profiles.append([age, weight, height, gender, activity, goal])
#     return np.array(profiles)
#
# user_profiles = generate_user_profiles(1000)
# user_profiles = StandardScaler().fit_transform(user_profiles)
#
# # Step 3: Create positive and negative pairs
# positive_pairs = []
# negative_pairs = []
#
# for user in user_profiles:
#     for _ in range(5):
#         food_idx = np.random.randint(0, len(food_features))
#         score = 1 if user[-1] == 0 and food_features[food_idx][0] < 200 else \
#                 1 if user[-1] == 2 and food_features[food_idx][0] > 300 else \
#                 1 if user[-1] == 1 and 200 <= food_features[food_idx][0] <= 300 else 0
#         if score == 1:
#             positive_pairs.append((user, food_features[food_idx], 1))
#         else:
#             negative_pairs.append((user, food_features[food_idx], 0))
#
# pairs = positive_pairs + negative_pairs
# np.random.shuffle(pairs)
#
# # Step 4: Dataset and Dataloader
# class FoodDataset(Dataset):
#     def __init__(self, pairs):
#         self.pairs = pairs
#
#     def __len__(self):
#         return len(self.pairs)
#
#     def __getitem__(self, idx):
#         user, food, label = self.pairs[idx]
#         return torch.tensor(user, dtype=torch.float32), torch.tensor(food, dtype=torch.float32), torch.tensor([label], dtype=torch.float32)
#
# dataset = FoodDataset(pairs)
# dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
#
# # Step 5: Model
# class Recommender(nn.Module):
#     def __init__(self, user_dim, food_dim):
#         super().__init__()
#         self.user_net = nn.Sequential(
#             nn.Linear(user_dim, 32),
#             nn.ReLU(),
#             nn.Linear(32, 16)
#         )
#         self.food_net = nn.Sequential(
#             nn.Linear(food_dim, 32),
#             nn.ReLU(),
#             nn.Linear(32, 16)
#         )
#         self.output = nn.CosineSimilarity(dim=1)
#
#     def forward(self, user, food):
#         u = self.user_net(user)
#         f = self.food_net(food)
#         return self.output(u, f).unsqueeze(1)
#
# model = Recommender(user_dim=6, food_dim=food_features.shape[1])
# criterion = nn.BCEWithLogitsLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
#
# # Step 6: Training
# for epoch in range(5):
#     total_loss = 0
#     for user, food, label in dataloader:
#         optimizer.zero_grad()
#         output = model(user, food)
#         loss = criterion(output, label)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#     print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
#
# # Step 7: Recommend Foods
#
# def recommend(user_input, top_k=3):
#     user = StandardScaler().fit_transform([user_input])[0]
#     user_tensor = torch.tensor(user, dtype=torch.float32).unsqueeze(0).repeat(len(food_features), 1)
#     food_tensor = torch.tensor(food_features, dtype=torch.float32)
#     with torch.no_grad():
#         scores = model(user_tensor, food_tensor).squeeze().numpy()
#     top_idx = np.argsort(scores)[-top_k:][::-1]
#     return top_idx
#
# # Example:
# # TODO: user_input -> user_dictionary = [{ 'age': 27, 'height': 170 }]
# user_input = [27, 80, 170, 0, 0, 0]  # Age, weight, height, male, sedentary, weight loss
# top_indexes = recommend(user_input, top_k=3)
# print("Recommended food indexes:", top_indexes)
