# Imbalanced Data Handling with SMOTE

## Overview
This project demonstrates techniques for handling imbalanced datasets in machine learning, specifically for credit card fraud detection. Two models are trained and compared: a baseline Random Forest classifier and an enhanced model using SMOTE (Synthetic Minority Over-sampling Technique) to address class imbalance.

## Technologies Used
- Python 3
- pandas
- NumPy
- scikit-learn
- imbalanced-learn (SMOTE)
- matplotlib
- seaborn

## Problem Statement
Credit card fraud detection presents a significant class imbalance problem, with fraudulent transactions representing only ~0.26% of all transactions. This imbalance can cause models to be biased toward the majority class, resulting in poor detection of fraudulent cases.

## Approach

### Part 1: Baseline Model
- Train a Random Forest Classifier on imbalanced data
- Evaluate performance on test set
- Analyze confusion matrix and classification metrics

### Part 2: SMOTE-Enhanced Model
- Apply SMOTE to synthetically balance the training dataset
- Train a Random Forest Classifier on balanced data
- Evaluate performance and compare with baseline

### Part 3: Comparison and Analysis
- Generate comprehensive PDF report with visualizations
- Compare precision, recall, and F1-scores
- Analyze trade-offs between models

## Key Findings

### Dataset Characteristics
- Total transactions: 39,999
- Fraudulent cases: 104 (0.26%)
- Legitimate cases: 39,895 (99.74%)
- Severe class imbalance ratio: ~1:384

### Model Performance
**Baseline Model:**
- High precision but lower recall for fraud detection
- May miss fraudulent transactions (higher false negatives)

**SMOTE-Enhanced Model:**
- Improved recall by ~19%
- Improved F1-score by ~11%
- Better at catching fraudulent transactions
- Slight increase in false positives

## Usage
Ensure `creditcard.csv` is in the same directory, then run:
```bash
python imbalanced_data.py
```

## Output
The script generates:
- Console output with dataset information and class distributions
- Confusion matrices and classification reports for both models
- `model_report.pdf`: Comprehensive 3-page report containing:
  - Side-by-side confusion matrix visualizations
  - Detailed metric comparison table
  - Summary with improvement percentages and observations

## Evaluation Metrics
Both models are evaluated using:
- Confusion matrix
- Precision: Accuracy of positive predictions
- Recall: Ability to find all positive cases
- F1-score: Harmonic mean of precision and recall
- Class-wise metrics for both fraud (Class 1) and legitimate (Class 0) transactions

## Key Insights
1. **Class Imbalance Impact**: Without balancing techniques, models tend to favor the majority class
2. **SMOTE Benefits**: Synthetic oversampling improves minority class detection
3. **Business Trade-offs**: Higher recall reduces missed fraud but may increase false alarms
4. **Context Matters**: The optimal model depends on business priorities (catching fraud vs. minimizing false positives)

## Course Information
**Course**: CS577 - Principles & Techniques of Data Science  
**Institution**: San Diego State University  
**Author**: Aiden B Jajo
