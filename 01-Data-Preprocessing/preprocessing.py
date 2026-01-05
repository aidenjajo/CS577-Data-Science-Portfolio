"""
Author: Aiden B Jajo
Institution: San Diego State University
Course: CS577 - Principles & Techniques of Data Science
Project: Data Preprocessing Pipeline
"""

# Step 1: Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Step 2: Load the dataset
df = pd.read_csv("Data.csv")

# Step 3: Handle missing values by replacing them with column means
df.fillna(df.mean(numeric_only=True), inplace=True)

# Step 4: Encode categorical variables using Label Encoding
label_encoders = {}
for column in df.columns:
    if df[column].dtype == "object":
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

# Step 5: Split features and target variable
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Step 6: Split dataset into training and test sets (80/20 ratio)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Apply feature scaling using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Display results
print("First 5 rows of scaled training data:")
print(X_train_scaled[:5])
print()

print("Training data size:")
print(X_train_scaled.shape)
print()

print("Testing data size:")
print(X_test_scaled.shape)
print()

print("Encoded columns:")
print(list(label_encoders.keys()))
