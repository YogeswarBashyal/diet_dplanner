import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the datasets
bmi_df = pd.read_csv('dataset/bmi.csv')
meals_df = pd.read_csv('dataset/mealplans.csv')
nutrition_df = pd.read_csv('dataset/nutrition.csv')

# Step 2: Clean and preprocess the BMI data
bmi_df.dropna(inplace=True)
bmi_df['Bmi'] = bmi_df['Weight'] / (bmi_df['Height'] ** 2)
bmi_df['BmiClass'] = pd.cut(bmi_df['Bmi'], bins=[0, 18.5, 24.9, 29.9, 34.9, 39.9, np.inf],
                            labels=['Underweight', 'Normal weight', 'Overweight', 'Obese Class 1', 'Obese Class 2', 'Obese Class 3'])

# Step 3: Explore and clean the meal data
print(meals_df.columns)

# Step 4: Explore the nutrition data
print(nutrition_df.describe())
print(nutrition_df.columns)

# Step 5: Normalize the nutritional data
def extract_numeric(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float)):
        return value
    return float(''.join(filter(str.isdigit, str(value))))

columns_to_normalize = ['calories', 'total_fat', 'cholesterol', 'sodium', 'fiber', 'protein']

for col in columns_to_normalize:
    if col not in nutrition_df.columns:
        print(f"Column {col} not found in the dataset")
    else:
        nutrition_df[col] = nutrition_df[col].apply(extract_numeric)
        nutrition_df[col] = pd.to_numeric(nutrition_df[col], errors='coerce')

# Remove rows with NaN values after preprocessing
nutrition_df.dropna(subset=columns_to_normalize, inplace=True)

scaler = StandardScaler()
nutrition_normalized = scaler.fit_transform(nutrition_df[columns_to_normalize])
nutrition_normalized_df = pd.DataFrame(nutrition_normalized, columns=columns_to_normalize)

# Step 6: Perform clustering on the nutrition data
kmeans = KMeans(n_clusters=2, random_state=42)
new_cluster = kmeans.fit_predict(nutrition_normalized_df)
nutrition_df['cluster'] = new_cluster

print("-------------------------- cluster information ----------------")
print(nutrition_df['cluster'])
print("-------------------------- cluster information ----------------")

# Step 7: Visualize clusters
plt.figure(figsize=(12, 8))
sns.scatterplot(x='calories', y='sodium', hue='cluster', data=nutrition_df, palette='viridis')
plt.title('Clusters of Meals Based on Nutritional Information')
plt.show()

# Step 8: Generate meal plans based on BMI
def generate_meal_plan(bmi_class, num_days):
    if bmi_class == 'Underweight':
        cluster = 0
    elif bmi_class == 'Normal weight':
        cluster = 1
    elif bmi_class == 'Overweight':
        cluster = 2
    elif bmi_class == 'Obese Class 1':
        cluster = 3
    else:
        cluster = 4
    meal_plan = nutrition_df[nutrition_df['cluster'] == cluster].sample(num_days)
    return meal_plan[['name', 'calories', 'total_fat', 'protein', 'sodium', 'fiber']]

# Ask the user for their BMI
user_bmi = float(input("Please enter your BMI: "))

# Determine the user's BMI class
# 25.5 -> Overweight
if user_bmi < 18.5:
    bmi_class = 'Underweight'
elif user_bmi < 24.9:
    bmi_class = 'Normal weight'
elif user_bmi < 29.9:
    bmi_class = 'Overweight'
elif user_bmi < 34.9:
    bmi_class = 'Obese Class 1'
else:
    bmi_class = 'Obese Class 2'

# Generate a 30-day meal plan
num_days = 30
meal_plan = generate_meal_plan(bmi_class, num_days)

# Print the meal plan
print(f"Meal plan for {bmi_class}:\n", meal_plan)

# Save the meal plan to a CSV file
meal_plan.to_csv('./trained/output.csv', index=False)
