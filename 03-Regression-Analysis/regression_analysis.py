"""
Author: Aiden B Jajo
Institution: San Diego State University
Course: CS577 - Principles & Techniques of Data Science
Project: Simple, Multiple, and Polynomial Regression Analysis
"""

# Step 1: Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures

# Part A: Simple Linear Regression
print("=== Part A: Simple Linear Regression ===")

# Step 2: Load dataset for simple linear regression
simple_linear_regression_data = pd.read_csv('data_linear_regression.csv')
X_simple = simple_linear_regression_data[['YearsExperience']].values
y_simple = simple_linear_regression_data['Salary'].values

# Step 3: Split the dataset into training and testing sets
X_train_simple, X_test_simple, y_train_simple, y_test_simple = train_test_split(X_simple, y_simple, test_size=0.2, random_state=42)

# Step 4: Train the Simple Linear Regression model on the training set
simple_linear_regressor = LinearRegression()
simple_linear_regressor.fit(X_train_simple, y_train_simple)

# Step 5: Visualize training set results
plt.scatter(X_train_simple, y_train_simple, color='red')
plt.plot(X_train_simple, simple_linear_regressor.predict(X_train_simple), color='blue')
plt.title('Simple Linear Regression (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Step 6: Visualize test set results
plt.scatter(X_test_simple, y_test_simple, color='green')
plt.plot(X_train_simple, simple_linear_regressor.predict(X_train_simple), color='blue')
plt.title('Simple Linear Regression (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

print("Part A completed.\n")

# Part B: Multiple Regression
print("=== Part B: Multiple Regression ===")

# Step 1: Load dataset for multiple regression
multiple_regression_data = pd.read_csv('data_multiple_regression.csv')
X_multiple = multiple_regression_data.iloc[:, :-1].values
y_multiple = multiple_regression_data.iloc[:, -1].values

# Step 2: Encode the categorical data
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X_multiple = ct.fit_transform(X_multiple)

# Step 3: Split the dataset into training and testing sets
X_train_multiple, X_test_multiple, y_train_multiple, y_test_multiple = train_test_split(X_multiple, y_multiple, test_size=0.2, random_state=42)

# Step 4: Train the Multiple Linear Regression model on the training set
multiple_regressor = LinearRegression()
multiple_regressor.fit(X_train_multiple, y_train_multiple)

# Step 5: Predict the test set results
y_pred_multiple = multiple_regressor.predict(X_test_multiple)
print("Multiple Regression Results:")
for actual, predicted in zip(y_test_multiple, y_pred_multiple):
    print(f"Actual: {actual:.2f}, Predicted: {predicted:.2f}")

print("Part B completed.\n")

# Part C: Polynomial Regression
print("=== Part C: Polynomial Regression ===")

# Step 1: Load dataset for polynomial regression
polynomial_regression_data = pd.read_csv('data_polynomial_regression.csv')
X_poly_data = polynomial_regression_data[['Level']].values
y_poly_data = polynomial_regression_data['Salary'].values

# Step 2: Train Linear Regression model on the whole dataset
linear_regressor_poly = LinearRegression()
linear_regressor_poly.fit(X_poly_data, y_poly_data)

# Step 3: Train Polynomial Regression model on the whole dataset
poly_features = PolynomialFeatures(degree=4)
X_poly = poly_features.fit_transform(X_poly_data)
poly_regressor = LinearRegression()
poly_regressor.fit(X_poly, y_poly_data)

# Step 4: Visualize Linear Regression fit
plt.scatter(X_poly_data, y_poly_data, color='red')
plt.plot(X_poly_data, linear_regressor_poly.predict(X_poly_data), color='blue')
plt.title('Polynomial Regression (Linear Fit)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Step 5: Visualize Polynomial Regression (basic resolution)
plt.scatter(X_poly_data, y_poly_data, color='red')
plt.plot(X_poly_data, poly_regressor.predict(X_poly), color='blue')
plt.title('Polynomial Regression (Basic Resolution)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Step 6: Visualize Polynomial Regression (high resolution - smooth curve)
X_grid = np.arange(min(X_poly_data), max(X_poly_data), 0.1).reshape(-1, 1)
plt.scatter(X_poly_data, y_poly_data, color='red')
plt.plot(X_grid, poly_regressor.predict(poly_features.transform(X_grid)), color='blue')
plt.title('Polynomial Regression (Smooth Fit)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Step 7: Predict results for Level 6.5
linear_prediction = linear_regressor_poly.predict([[6.5]])
polynomial_prediction = poly_regressor.predict(poly_features.transform([[6.5]]))
print(f"Prediction using Linear Regression for Level 6.5: {linear_prediction[0]:.2f}")
print(f"Prediction using Polynomial Regression for Level 6.5: {polynomial_prediction[0]:.2f}")

print("Part C completed.")
