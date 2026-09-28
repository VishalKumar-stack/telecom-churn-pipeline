# Customer Churn Prediction Pipeline

## Project Overview

This project demonstrates an end-to-end machine learning pipeline for predicting customer churn using a simulated telecom customer dataset.

The project covers data generation, missing-value handling, categorical feature encoding, train-test splitting, feature scaling, Random Forest classification, hyperparameter tuning, and model evaluation.

## Objective

The objective is to predict whether a customer is likely to churn based on customer and service-related features.

## Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn

## Features

The dataset contains features such as:

- Customer tenure
- Monthly charges
- Total charges
- Internet service
- Contract type
- Technical support calls

## Machine Learning Workflow

1. Generate a simulated customer dataset
2. Create the churn target
3. Introduce missing values
4. Handle missing values using median imputation
5. Encode categorical variables using one-hot encoding
6. Split the data into training and testing sets
7. Scale the features
8. Train a Random Forest classifier
9. Tune model hyperparameters using GridSearchCV
10. Evaluate the model

## Model

Random Forest Classifier was used for the classification task.

GridSearchCV was used to tune:

- `n_estimators`
- `max_depth`
- `min_samples_split`

## Results

The best model selected during hyperparameter tuning used:

- Number of estimators: 200
- Maximum depth: None
- Minimum samples split: 5

Evaluation results on the test set:

- ROC-AUC: 0.840
- Accuracy: 0.77

The model was also evaluated using:

- Precision
- Recall
- F1-score
- Confusion Matrix

## Dataset

The dataset is synthetically generated using NumPy and Pandas for educational purposes.

## Key Learning

This project demonstrates the complete machine learning workflow from data preparation and preprocessing to model training, hyperparameter tuning, and evaluation.