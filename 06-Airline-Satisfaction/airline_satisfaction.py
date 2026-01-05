"""
Author: Aiden B Jajo
Institution: San Diego State University
Course: CS577 - Principles & Techniques of Data Science
Project: Airline Passenger Satisfaction Prediction
"""

# Step 1: Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
from fpdf import FPDF

print("=== Part 1: Data Preprocessing ===")

# Step 2: Load dataset and inspect
df = pd.read_csv("dataset-1.csv")
print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nDataset info:")
print(df.info())

# Step 3: Drop unrelated columns
# Unnamed: 0, id, and Gate location columns are unnecessary or have too many missing values
df.drop(columns=['Unnamed: 0', 'id', 'Gate location'], errors='ignore', inplace=True)

# Step 4a: Remove extreme outliers in delay columns
# Drop outliers in 'Departure Delay in Minutes' and 'Arrival Delay in Minutes'
df = df[(df['Departure Delay in Minutes'] < df['Departure Delay in Minutes'].quantile(0.99)) & 
        (df['Arrival Delay in Minutes'] < df['Arrival Delay in Minutes'].quantile(0.99))]

# Step 4b: Handle missing values appropriately
# Fill numeric missing values with the median to maintain distribution integrity
df.loc[:, 'Arrival Delay in Minutes'] = df['Arrival Delay in Minutes'].fillna(df['Arrival Delay in Minutes'].median())
df.fillna(df.select_dtypes(include=['number']).median(), inplace=True)

# Fill categorical missing values with the most frequent category (mode)
categorical_columns = ['Gender', 'Customer Type', 'Type of Travel', 'Class', 'satisfaction']
for col in categorical_columns:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].mode()[0])

# Step 4c: Encode categorical data using Label Encoding
label_encoders = {}
for col in categorical_columns:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le  # Store encoders for future reference

# Step 5: Define features and target variable
X = df.drop(columns=['satisfaction'])
y = df['satisfaction']

# Step 6: Train-test split (ensuring balanced data distribution)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 7: Standardize numerical features for better model performance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Data preprocessing completed.\n")

print("=== Part 2: Model Training ===")

# Step 8: Define hyperparameter grids for GridSearchCV
rf_params = {'n_estimators': [50, 100, 200], 'max_depth': [10, 20, None], 'min_samples_split': [2, 5, 10]}
xgb_params = {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 6, 9]}

# Step 9: Optimize Random Forest model using GridSearchCV
rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=3, n_jobs=-1, verbose=1)
rf_grid.fit(X_train, y_train)
rf_best = rf_grid.best_estimator_
print(f"\nBest Random Forest Parameters: {rf_grid.best_params_}")

# Step 10: Optimize XGBoost model using GridSearchCV
xgb_grid = GridSearchCV(XGBClassifier(eval_metric='logloss'), xgb_params, cv=3, n_jobs=-1, verbose=1)
xgb_grid.fit(X_train, y_train)
xgb_best = xgb_grid.best_estimator_
print(f"Best XGBoost Parameters: {xgb_grid.best_params_}")

print("\nModel training completed.\n")

print("=== Part 3: Model Evaluation ===")

# Step 11: Make predictions with the best models
y_pred_rf = rf_best.predict(X_test)
y_pred_xgb = xgb_best.predict(X_test)

# Step 12: Evaluate model performance
rf_report = classification_report(y_test, y_pred_rf)
xgb_report = classification_report(y_test, y_pred_xgb)

print("Optimized Random Forest Classification Report:")
print(rf_report)
print("Optimized Random Forest Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))
print()

print("Optimized XGBoost Classification Report:")
print(xgb_report)
print("Optimized XGBoost Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_xgb))

# Step 13: Generate confusion matrix visualizations
fig, axes = plt.subplots(1, 2, figsize=(8, 3))
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', ax=axes[0], cbar=False)
axes[0].set_title("Optimized Random Forest Confusion Matrix")
sns.heatmap(confusion_matrix(y_test, y_pred_xgb), annot=True, fmt='d', ax=axes[1], cbar=False)
axes[1].set_title("Optimized XGBoost Confusion Matrix")

# Save confusion matrix image
confusion_matrix_image = "confusion_matrix.png"
plt.savefig(confusion_matrix_image, dpi=100, bbox_inches='tight')
plt.close()

# Step 14: Generate PDF summary report
# Automatically create a one-page PDF report with classification metrics and confusion matrices
pdf = FPDF()
pdf.add_page()
pdf.set_font("Times", size=10)

pdf.cell(200, 8, "Optimized Model Evaluation Report", ln=True, align='C')

pdf.ln(3)
pdf.multi_cell(0, 5, "Random Forest Classification Report:\n" + rf_report)
pdf.ln(2)
pdf.multi_cell(0, 5, "XGBoost Classification Report:\n" + xgb_report)

# Add optimized confusion matrix image to the PDF
pdf.ln(2)
pdf.image(confusion_matrix_image, x=10, w=160)

pdf.output("model_evaluation_report.pdf")
print("\nPDF Report Generated: model_evaluation_report.pdf with Optimized Models and Confusion Matrix Image")
