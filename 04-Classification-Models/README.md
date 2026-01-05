# Classification Models Comparison

## Overview
This project implements and compares seven classification algorithms on a social network advertising dataset. Each model is optimized using GridSearchCV for hyperparameter tuning and evaluated through multiple visualization techniques and performance metrics.

## Technologies Used
- Python 3
- pandas
- NumPy
- scikit-learn
- matplotlib
- seaborn
- mlxtend

## Models Implemented
1. **Logistic Regression**: Linear classification model
2. **K-Nearest Neighbors (KNN)**: Instance-based learning algorithm
3. **Support Vector Machine (Linear)**: Linear kernel SVM
4. **Support Vector Machine (RBF)**: Radial basis function kernel SVM
5. **Decision Tree**: Tree-based classification model
6. **Random Forest**: Ensemble of decision trees
7. **Naive Bayes**: Probabilistic classifier based on Bayes' theorem

## Features
- Hyperparameter tuning using GridSearchCV with 5-fold cross-validation
- Feature scaling using StandardScaler
- Comprehensive evaluation metrics: confusion matrix, classification report, accuracy score
- Multiple visualization types for result analysis

## Visualizations Generated
The script produces the following visualizations for each model:
- **Training Set Scatter Plots** (7 plots): Shows correct (green) vs incorrect (red) predictions on training data
- **Test Set Scatter Plots** (7 plots): Shows correct (green) vs incorrect (red) predictions on test data
- **Decision Boundary Plots** (7 plots): Visualizes how each model partitions the feature space
- **Confusion Matrix Heatmaps** (1 grid with 7 matrices): Displays true positives, false positives, true negatives, and false negatives

Total: 28 visualizations

## Usage
Ensure `Social_Network_Ads_Classification.csv` is in the same directory, then run:
```bash
python classification_models.py
```

## Output
The script displays:
- Classification reports with precision, recall, and F1-scores for each model
- Model performance comparison table sorted by accuracy
- All visualizations as matplotlib windows

## Performance Metrics
Models are evaluated using:
- Accuracy score
- Confusion matrix
- Precision, recall, and F1-score for both classes
- Macro and weighted averages

## Course Information
**Course**: CS577 - Principles & Techniques of Data Science  
**Institution**: San Diego State University  
**Author**: Aiden B Jajo
