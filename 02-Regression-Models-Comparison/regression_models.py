"""
Author: Aiden B Jajo
Institution: San Diego State University
Course: CS577 - Principles & Techniques of Data Science
Project: Regression Models Comparison
"""

# Step 1: Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

# Step 2: Load the dataset
dataset = pd.read_csv("combined_cycle_power_plant.csv")
print("Dataset Head:\n", dataset.head())

# Step 3: Split the dataset into training and testing sets
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4a: Train Random Forest model
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

# Step 4b: Train Decision Tree model
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)
dt_predictions = dt_model.predict(X_test)

# Step 4c: Train Multiple Linear Regression model
multi_model = LinearRegression()
multi_model.fit(X_train, y_train)
multi_predictions = multi_model.predict(X_test)

# Step 4d: Train Polynomial Regression model
poly = PolynomialFeatures(degree=4)
X_poly = poly.fit_transform(X_train)
poly_model = LinearRegression()
poly_model.fit(X_poly, y_train)
poly_predictions = poly_model.predict(poly.transform(X_test))

# Step 5: Evaluate model performance for Part A
print("\n=== Part A: Model Performance ===")
models = {"Random Forest": rf_predictions,
          "Decision Tree": dt_predictions,
          "Linear Regression": multi_predictions,
          "Polynomial Regression": poly_predictions}

for name, predictions in models.items():
    print(f"\n{name} Model")
    print("MSE:", mean_squared_error(y_test, predictions))
    print("R2 Score:", r2_score(y_test, predictions))

# Step 6: Apply feature scaling for SVR
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()

# Step 7: Train SVR model
svr_model = SVR(kernel='rbf')
svr_model.fit(X_train_scaled, y_train_scaled)

# Step 8: Predict test results for SVR
svr_predictions_scaled = svr_model.predict(X_test_scaled)
svr_predictions = scaler_y.inverse_transform(svr_predictions_scaled.reshape(-1, 1)).flatten()

# Step 9: Evaluate SVR model performance
print("\n=== Part B: SVR Model Performance ===")
print("MSE:", mean_squared_error(y_test, svr_predictions))
print("R2 Score:", r2_score(y_test, svr_predictions))
