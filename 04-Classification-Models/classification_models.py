"""
Author: Aiden B Jajo
Institution: San Diego State University
Course: CS577 - Principles & Techniques of Data Science
Project: Classification Models Comparison
"""

# Part 1: Data Preparation

# Step 1a: Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from mlxtend.plotting import plot_decision_regions

print("=== Part 1: Data Preparation ===")

# Step 1b: Load the dataset
data = pd.read_csv("Social_Network_Ads_Classification.csv")
features = data[['Age', 'EstimatedSalary']]
target = data['Purchased']

# Step 1c: Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=0)

# Step 1d: Apply feature scaling using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Data preparation completed.\n")

# Part 2: Model Training

print("=== Part 2: Model Training ===")

# Step 2a: Define models and hyperparameters for GridSearchCV
models = {
    "LogReg": (LogisticRegression(), {'C': [0.1, 1, 10]}),
    "KNN": (KNeighborsClassifier(), {'n_neighbors': [3, 5, 7]}),
    "SVM Linear": (SVC(kernel='linear'), {'C': [0.1, 1, 10]}),
    "SVM RBF": (SVC(), {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}),
    "Decision Tree": (DecisionTreeClassifier(), {'max_depth': [3, 5, 10]}),
    "Random Forest": (RandomForestClassifier(), {'n_estimators': [50, 100], 'max_depth': [3, 5, 10]}),
    "Naive Bayes": (GaussianNB(), {}),
}

# Initialize dictionaries to store results
best_models = {}
performance = {}

# Train models using GridSearchCV and evaluate performance
def train_and_evaluate(models, X_train, y_train, X_test, y_test):
    """
    Train each model with GridSearchCV to find best parameters,
    then evaluate performance on test set
    """
    for name, (model, params) in models.items():
        grid = GridSearchCV(model, params, cv=5, scoring='accuracy')
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        best_models[name] = best_model
        
        # Step 2b: Predict test set results
        y_pred = best_model.predict(X_test)
        
        # Store evaluation metrics
        performance[name] = {
            "conf_matrix": confusion_matrix(y_test, y_pred),
            "report": classification_report(y_test, y_pred, output_dict=True),
            "accuracy": accuracy_score(y_test, y_pred),
            "y_pred": y_pred
        }

# Execute training function
train_and_evaluate(models, X_train_scaled, y_train, X_test_scaled, y_test)

print("Model training completed.\n")

# Part 3: Model Evaluation and Visualization

print("=== Part 3: Model Evaluation and Visualization ===")

# Plot confusion matrices for all models
def plot_confusion_matrices(results):
    """
    Visualize confusion matrices for all models in a grid layout
    """
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    for i, (name, data) in enumerate(results.items()):
        sns.heatmap(data['conf_matrix'], annot=True, fmt='d', ax=axes[i], cmap='coolwarm')
        axes[i].set_title(name)
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')
    plt.tight_layout()
    plt.show()

# Visualize decision boundaries for all models
def visualize_decision_boundaries(models, X_train, y_train):
    """
    Plot decision boundaries showing how each model classifies the feature space
    """
    for name, model in models.items():
        if hasattr(model, "predict"):
            plt.figure(figsize=(6, 5))
            plot_decision_regions(X_train, y_train.to_numpy(), clf=model, legend=2)
            plt.title(f"Decision Boundary - {name}")
            plt.xlabel("Age")
            plt.ylabel("Estimated Salary")
            plt.show()

# Print classification reports for all models
def print_classification_reports(results):
    """
    Display detailed classification metrics for each model
    """
    for name, data in results.items():
        print(f"\n{name} Classification Report:")
        print(pd.DataFrame(data['report']))

# Compare model performance by accuracy
def compare_models(results):
    """
    Create a sorted comparison table of model accuracies
    """
    perf_df = pd.DataFrame({
        "Model": list(results.keys()),
        "Accuracy": [data["accuracy"] for data in results.values()]
    })
    print("\nModel Performance Comparison:")
    print(perf_df.sort_values(by="Accuracy", ascending=False))

# Scatter plot function for visualizing TP, FP, FN, TN
def plot_results_scatter(X, y_true, y_pred, title):
    """
    Scatter plot showing correct (green) and incorrect (red) predictions
    """
    plt.figure(figsize=(8, 6))
    colors = ['green' if yt == yp else 'red' for yt, yp in zip(y_true, y_pred)]
    plt.scatter(X[:, 0], X[:, 1], c=colors, alpha=0.6)
    plt.xlabel("Age")
    plt.ylabel("Estimated Salary")
    plt.title(title)
    plt.show()

# Generate scatter plots for training and test results
for name, model in best_models.items():
    y_train_pred = model.predict(X_train_scaled)
    plot_results_scatter(X_train_scaled, y_train, y_train_pred, f"Train Results - {name}")
    plot_results_scatter(X_test_scaled, y_test, performance[name]['y_pred'], f"Test Results - {name}")

# Execute visualization and performance analysis
plot_confusion_matrices(performance)
visualize_decision_boundaries(best_models, X_train_scaled, y_train)
print_classification_reports(performance)
compare_models(performance)
