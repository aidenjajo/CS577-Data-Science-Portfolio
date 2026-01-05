# Data Preprocessing Pipeline

## Overview
This project implements a complete data preprocessing pipeline for machine learning applications. The script handles missing data, encodes categorical variables, splits the dataset, and applies feature scaling.

## Technologies Used
- Python 3
- pandas
- scikit-learn

## Features
- **Missing Value Imputation**: Replaces missing values with column means
- **Label Encoding**: Converts categorical variables to numerical format
- **Train-Test Split**: Divides data into 80% training and 20% testing sets
- **Feature Scaling**: Standardizes features using StandardScaler

## Usage
Ensure `Data.csv` is in the same directory as the script, then run:
```bash
python preprocessing.py
```

## Output
The script displays:
- First 5 rows of scaled training data
- Training set dimensions
- Testing set dimensions
- List of encoded categorical columns

## Course Information
**Course**: CS577 - Principles & Techniques of Data Science  
**Institution**: San Diego State University  
**Author**: Aiden B Jajo
