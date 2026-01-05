# Artificial Neural Network for Customer Churn Prediction

## Overview
This project implements a deep learning solution using Artificial Neural Networks (ANN) to predict customer churn in the banking industry. The model analyzes customer demographics, account information, and activity patterns to predict whether a customer is likely to leave the bank.

## Technologies Used
- Python 3
- TensorFlow/Keras
- pandas
- NumPy
- scikit-learn
- matplotlib

## Problem Statement
Customer churn is a critical business metric for banks. Identifying customers at risk of leaving allows banks to take proactive retention measures. This project builds a neural network to predict churn based on customer features.

## Model Architecture

### Neural Network Structure
- **Input Layer**: 11 features (after encoding and preprocessing)
- **Hidden Layer 1**: 6 neurons with ReLU activation
- **Hidden Layer 2**: 6 neurons with ReLU activation
- **Hidden Layer 3**: 6 neurons with ReLU activation
- **Output Layer**: 1 neuron with Sigmoid activation (binary classification)

### Training Configuration
- **Optimizer**: Adam
- **Loss Function**: Binary Crossentropy
- **Metrics**: Accuracy
- **Epochs**: 100 (with early stopping)
- **Batch Size**: 32
- **Validation Split**: 20%
- **Early Stopping**: Monitors validation loss with patience of 10 epochs

## Data Preprocessing Steps

### Feature Engineering
1. **Feature Selection**: Used columns [3:-1] from the dataset
2. **Categorical Encoding**:
   - Gender: Label Encoding (Male/Female → 0/1)
   - Geography: One-Hot Encoding with dummy variable trap avoidance
3. **Feature Scaling**: StandardScaler for all features
4. **Train-Test Split**: 80/20 ratio

### Dataset Features
- Credit Score
- Geography (France, Spain, Germany)
- Gender (Male, Female)
- Age
- Tenure (years with bank)
- Account Balance
- Number of Products
- Has Credit Card (Yes/No)
- Is Active Member (Yes/No)
- Estimated Salary

## Key Features
- Deep neural network with multiple hidden layers
- Early stopping to prevent overfitting
- Training history visualization (loss and accuracy curves)
- Single customer prediction capability
- Comprehensive evaluation metrics

## Usage
Ensure `Churn_Modelling.xlsx` is in the same directory, then run:
```bash
python ann_churn_prediction.py
```

## Output
The script generates:
- Console output with dataset information and preprocessing steps
- Model architecture summary
- Training progress with epoch-by-epoch metrics
- `training_history.png`: Visualization of training and validation loss/accuracy
- Single customer prediction example
- Confusion matrix and accuracy score on test set
- Comprehensive model summary

## Evaluation Metrics
- **Accuracy Score**: Overall prediction accuracy
- **Confusion Matrix**: Breakdown of predictions
  - True Negatives: Correctly predicted customers who stayed
  - False Positives: Incorrectly predicted as churned
  - False Negatives: Missed churn cases
  - True Positives: Correctly predicted churn

## Model Performance
The model achieves approximately 86% accuracy on the test set, demonstrating strong performance in predicting customer churn.

## Deep Learning Highlights
- **Activation Functions**: ReLU for hidden layers, Sigmoid for binary output
- **Regularization**: Early stopping prevents overfitting
- **Optimization**: Adam optimizer for efficient gradient descent
- **Validation**: 20% validation split during training for model monitoring

## Course Information
**Course**: CS577 - Principles & Techniques of Data Science  
**Institution**: San Diego State University  
**Author**: Aiden B Jajo
