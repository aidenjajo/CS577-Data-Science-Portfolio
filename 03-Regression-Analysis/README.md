# Simple, Multiple, and Polynomial Regression Analysis

## Overview
This project demonstrates three fundamental regression techniques applied to different datasets. Each regression type is implemented, visualized, and evaluated to showcase their distinct approaches to modeling relationships between variables.

## Technologies Used
- Python 3
- pandas
- NumPy
- scikit-learn
- matplotlib

## Project Components

### Part A: Simple Linear Regression
Predicts salary based on years of experience using a single feature linear model.
- Dataset: `data_linear_regression.csv`
- Visualizations: Training set and test set scatter plots with regression line

### Part B: Multiple Regression
Predicts outcomes using multiple features with categorical encoding.
- Dataset: `data_multiple_regression.csv`
- Features: OneHotEncoding for categorical variables
- Output: Actual vs predicted values comparison

### Part C: Polynomial Regression
Models non-linear relationships between position level and salary.
- Dataset: `data_polynomial_regression.csv`
- Polynomial degree: 4
- Visualizations: Linear fit, polynomial fit (basic and high-resolution)
- Predictions: Comparison of linear vs polynomial predictions for Level 6.5

## Features
- Train-test split (80/20 ratio) for simple linear regression
- Categorical data encoding using OneHotEncoder
- Multiple visualization techniques for model comparison
- Polynomial feature transformation for non-linear modeling

## Usage
Ensure all three CSV files are in the same directory, then run:
```bash
python regression_analysis.py
```

## Output
The script displays:
- Five matplotlib visualizations showing different regression fits
- Multiple regression predictions with actual vs predicted values
- Final predictions comparing linear and polynomial approaches

## Course Information
**Course**: CS577 - Principles & Techniques of Data Science  
**Institution**: San Diego State University  
**Author**: Aiden B Jajo
