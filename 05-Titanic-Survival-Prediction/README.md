# Titanic Survival Prediction

## Overview
This project uses machine learning to predict passenger survival on the Titanic. Six classification algorithms are implemented and optimized using GridSearchCV to find the best performing model for this binary classification task.

## Technologies Used
- Python 3
- pandas
- NumPy
- scikit-learn

## Models Implemented
1. **Support Vector Machine (SVM)**: Kernel-based classification with RBF kernel
2. **Logistic Regression**: Linear probabilistic classifier
3. **K-Nearest Neighbors (KNN)**: Instance-based learning algorithm
4. **Decision Tree**: Tree-based classification model
5. **Random Forest**: Ensemble of decision trees
6. **Naive Bayes**: Probabilistic classifier based on Bayes' theorem

## Data Preprocessing Steps
- **Feature Selection**: Removed irrelevant columns (Name, Ticket, Cabin)
- **Missing Value Imputation**: Filled Age and Fare with median values, Embarked with mode
- **Categorical Encoding**: Converted Sex and Embarked to numerical values using LabelEncoder
- **Feature Scaling**: Standardized features using StandardScaler
- **Train-Test Split**: 80/20 ratio with random_state=42

## Hyperparameter Tuning
GridSearchCV with 5-fold cross-validation is used to find optimal parameters for:
- **SVM**: C, gamma, kernel
- **Logistic Regression**: C
- **KNN**: n_neighbors
- **Decision Tree**: max_depth
- **Random Forest**: n_estimators, max_depth
- **Naive Bayes**: No tunable parameters (trained directly)

## Usage
Ensure `train.csv` and `test.csv` are in the same directory, then run:
```bash
python titanic_survival.py
```

## Output
For each model, the script displays:
- Best hyperparameters found by GridSearchCV
- Classification report (precision, recall, F1-score)
- Confusion matrix

## Evaluation Metrics
Models are evaluated using:
- Accuracy score
- Precision, recall, and F1-score for both classes
- Confusion matrix showing true positives, false positives, true negatives, and false negatives

## Course Information
**Course**: CS577 - Principles & Techniques of Data Science  
**Institution**: San Diego State University  
**Author**: Aiden B Jajo
