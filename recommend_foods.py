# recommend_foods.py

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib

# ------------------------------
# Load Scalers, Model, and Data
# ------------------------------

food_scaler = joblib.load("models/food_scaler.pkl")
user_scaler = joblib.load("models/user_scaler.pkl")
food_df = pd.read_csv("models/cleaned_food.csv")

def clean_nutrition_data(df):
    df = df.copy()
    df = df.drop(columns=["name", "serving_size"], errors="ignore")
    for col in df.columns:
        df[col] = df[col].astype(str).str.extract(r'([\d.]+)').astype(float)
    df = df.dropna(axis=1, how="all")
    df = df.dropna()
    return df

food_features = clean_nutrition_data(food_df)
food_features_scaled = food_scaler.transform(food_features.values)
food_names = food_df.loc[food_features.index]["name"].values
food_calories = food_features["calories"].values

# ------------------------------
# Recommender Model Definition
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

model = Recommender(user_dim=6, food_dim=food_features_scaled.shape[1])
model.load_state_dict(torch.load("models/food_recommender.pt"))
model.eval()

# ------------------------------
# Calorie Estimator using BMI
# ------------------------------

def calculate_daily_calories(age, weight, height, gender, activity, goal):
    if gender == 0:  # Male
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    else:  # Female
        bmr = 10 * weight + 6.25 * height - 5 * age - 161

    activity_factors = [1.2, 1.5, 1.75]  # sedentary, moderate, active
    bmr *= activity_factors[activity]

    if goal == 0:  # weight loss
        bmr -= 500
    elif goal == 2:  # weight gain
        bmr += 300

    return max(1200, round(bmr))  # Ensure a healthy minimum

# ------------------------------
# Recommender Function
# ------------------------------

def recommend(user_input, top_k=100):
    user_scaled = user_scaler.transform([user_input])[0]
    user_tensor = torch.tensor(user_scaled, dtype=torch.float32).unsqueeze(0).repeat(len(food_features_scaled), 1)
    food_tensor = torch.tensor(food_features_scaled, dtype=torch.float32)

    with torch.no_grad():
        scores = model(user_tensor, food_tensor).squeeze().numpy()

    top_idx = np.argsort(scores)[-top_k:][::-1]
    return top_idx, scores[top_idx]

# ------------------------------
# Meal Planning Logic
# ------------------------------

def create_meal_plan(user_input):
    daily_calories = calculate_daily_calories(*user_input)
    print(f"\nEstimated Daily Calorie Need: {daily_calories} kcal")

    meal_targets = {
        "breakfast": 0.3 * daily_calories,
        "lunch": 0.4 * daily_calories,
        "dinner": 0.3 * daily_calories,
    }

    top_idx, _ = recommend(user_input, top_k=100)
    used = set()
    meal_plan = []

    def select_meal_items(target_kcal):
        items = []
        total = 0
        for i in top_idx:
            if i in used:
                continue
            cal = food_calories[i]
            if total + cal <= target_kcal + 100:
                used.add(i)
                items.append((food_names[i], round(cal)))
                total += cal
            if total >= target_kcal - 100:
                break
        return items, round(total)

    for day in range(1, 8):
        daily_meals = {}
        for meal, kcal_target in meal_targets.items():
            items, total_kcal = select_meal_items(kcal_target)
            daily_meals[meal] = (items, total_kcal)
        meal_plan.append(daily_meals)

    return meal_plan

# ------------------------------
# CLI Main
# ------------------------------

if __name__ == "__main__":
    # Example user input: age, weight (kg), height (cm), gender (0: M, 1: F), activity (0–2), goal (0–2)
    user_profile = [50, 55, 175, 0, 1, 2]  # weight loss example
    plan = create_meal_plan(user_profile)
    print("\n--- 7-Day Personalized Meal Plan ---")
    for day, meals in enumerate(plan, 1):
        print(f"\nDay {day}")
        for meal in ["breakfast", "lunch", "dinner"]:
            items, kcal = meals[meal]
            if not items:
                print(f"  {meal.title()}: No Match (0 kcal)")
            else:
                item_names = ', '.join(f"{name} ({cal} kcal)" for name, cal in items)
                print(f"  {meal.title()}: {item_names} = {kcal} kcal")
