# Regression Models Comparison

## Overview
This project implements and compares five different regression models on a combined cycle power plant dataset. The models are evaluated based on Mean Squared Error (MSE) and R² Score to determine prediction accuracy.

## Technologies Used
- Python 3
- pandas
- NumPy
- scikit-learn
- matplotlib

## Models Implemented
- **Random Forest**: Ensemble learning method using multiple decision trees
- **Decision Tree**: Tree-based model for non-linear regression
- **Multiple Linear Regression**: Linear approach using multiple features
- **Polynomial Regression**: Non-linear regression using polynomial features (degree 4)
- **Support Vector Regression (SVR)**: Kernel-based regression with feature scaling

## Features
- Train-test split with 80/20 ratio
- Feature scaling for SVR model using StandardScaler
- Performance evaluation using MSE and R² metrics
- Comparison across multiple regression approaches

## Usage
Ensure `combined_cycle_power_plant.csv` is in the same directory, then run:
```bash
python regression_models.py
```

## Output
The script displays:
- Dataset preview (first 5 rows)
- Part A: Performance metrics for Random Forest, Decision Tree, Linear, and Polynomial models
- Part B: Performance metrics for SVR model

## Course Information
**Course**: CS577 - Principles & Techniques of Data Science  
**Institution**: San Diego State University  
**Author**: Aiden B Jajo
