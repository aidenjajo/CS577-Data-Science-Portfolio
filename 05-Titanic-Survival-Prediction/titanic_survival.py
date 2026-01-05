"""
Author: Aiden B Jajo
Institution: San Diego State University
Course: CS577 - Principles & Techniques of Data Science
Project: Titanic Survival Prediction
"""

# Step 1: Import required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix

print("=== Data Preprocessing ===")

# Step 2: Load the datasets
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# Step 3: Drop irrelevant columns
train_df.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
test_df.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# Step 4: Handle missing values
train_df = train_df.assign(Age=train_df['Age'].fillna(train_df['Age'].median()))
test_df = test_df.assign(Age=test_df['Age'].fillna(test_df['Age'].median()))
test_df = test_df.assign(Fare=test_df['Fare'].fillna(test_df['Fare'].median()))
train_df = train_df.assign(Embarked=train_df['Embarked'].fillna('S'))
test_df = test_df.assign(Embarked=test_df['Embarked'].fillna('S'))

# Step 5: Encode categorical variables
label_encoder = LabelEncoder()
train_df['Sex'] = label_encoder.fit_transform(train_df['Sex'])
test_df['Sex'] = label_encoder.transform(test_df['Sex'])
train_df['Embarked'] = label_encoder.fit_transform(train_df['Embarked'])
test_df['Embarked'] = label_encoder.transform(test_df['Embarked'])

# Step 6: Prepare features and target variable
X = train_df.drop(['Survived', 'PassengerId'], axis=1)
y = train_df['Survived']

# Step 7: Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 8: Apply feature scaling using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Data preprocessing completed.\n")

print("=== Model Training and Evaluation ===")

# Define parameter grids for GridSearchCV
param_grids = {
    "SVM": {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto'], 'kernel': ['rbf']},
    "Logistic Regression": {'C': [0.1, 1, 10]},
    "KNN": {'n_neighbors': [3, 5, 7]},
    "Decision Tree": {'max_depth': [3, 5, 10]},
    "Random Forest": {'n_estimators': [50, 100], 'max_depth': [3, 5, 10]},
    "Naive Bayes": {}
}

def train_and_evaluate(model, model_name, params):
    """
    Train model with GridSearchCV (if parameters exist) and evaluate performance
    """
    if params:
        # Use GridSearchCV to find best parameters
        grid = GridSearchCV(model, params, cv=5, scoring='accuracy')
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        print(f"\n=== {model_name} ===")
        print(f"Best Parameters: {grid.best_params_}")
    else:
        # Train directly if no tunable parameters
        best_model = model
        best_model.fit(X_train, y_train)
        print(f"\n=== {model_name} ===")
    
    # Predict and evaluate
    y_pred = best_model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

# Define models
models = {
    "SVM": SVC(),
    "Logistic Regression": LogisticRegression(),
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Naive Bayes": GaussianNB()
}

# Train and evaluate each model
for name, model in models.items():
    train_and_evaluate(model, name, param_grids[name])
