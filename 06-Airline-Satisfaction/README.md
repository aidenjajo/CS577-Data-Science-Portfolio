# Airline Passenger Satisfaction Prediction

## Overview
This project analyzes airline passenger satisfaction survey data to identify factors correlated with passenger satisfaction and build predictive models. Two ensemble learning algorithms are implemented and optimized to predict whether a passenger will be satisfied or dissatisfied with their flight experience.

## Technologies Used
- Python 3
- pandas
- NumPy
- scikit-learn
- XGBoost
- matplotlib
- seaborn
- fpdf

## Models Implemented
1. **Random Forest Classifier**: Ensemble learning method using multiple decision trees
2. **XGBoost Classifier**: Gradient boosting framework for efficient and accurate predictions

## Data Preprocessing Steps
- **Outlier Removal**: Removed extreme outliers (99th percentile) from departure and arrival delay columns
- **Missing Value Imputation**: 
  - Numeric features: Filled with median values to maintain distribution integrity
  - Categorical features: Filled with mode (most frequent value)
- **Feature Selection**: Dropped irrelevant columns (Unnamed: 0, id, Gate location)
- **Categorical Encoding**: Applied Label Encoding to all categorical variables
- **Feature Scaling**: Standardized all features using StandardScaler
- **Stratified Split**: 80/20 train-test split ensuring balanced class distribution

## Hyperparameter Tuning
GridSearchCV with 3-fold cross-validation is used to optimize:

**Random Forest:**
- n_estimators: [50, 100, 200]
- max_depth: [10, 20, None]
- min_samples_split: [2, 5, 10]

**XGBoost:**
- n_estimators: [50, 100, 200]
- learning_rate: [0.01, 0.1, 0.2]
- max_depth: [3, 6, 9]

## Features
- Comprehensive data preprocessing pipeline
- Automated hyperparameter optimization
- Performance evaluation with multiple metrics
- Confusion matrix visualization
- Automated PDF report generation

## Usage
Ensure `dataset-1.csv` is in the same directory, then run:
```bash
python airline_satisfaction.py
```

## Output
The script generates:
- Console output with best hyperparameters for each model
- Classification reports (precision, recall, F1-score) for both models
- Confusion matrices in console
- `confusion_matrix.png`: Visual comparison of both models' confusion matrices
- `model_evaluation_report.pdf`: One-page summary report with all metrics and visualizations

## Evaluation Metrics
Models are evaluated using:
- Accuracy score
- Precision, recall, and F1-score for both classes (satisfied/dissatisfied)
- Macro and weighted averages
- Confusion matrix showing prediction distribution

## Dataset Features
The dataset includes passenger demographics, flight details, and satisfaction ratings for various service aspects:
- Demographics: Gender, Age, Customer Type
- Flight details: Travel type, Class, Flight Distance
- Service ratings: WiFi, Food, Seat comfort, Entertainment, Cleanliness, etc.
- Delays: Departure and Arrival delay times
- Target: Overall satisfaction level

## Course Information
**Course**: CS577 - Principles & Techniques of Data Science  
**Institution**: San Diego State University  
**Author**: Aiden B Jajo
