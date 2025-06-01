# This project implements a binary classification model using Logistic Regression on the Breast Cancer Wisconsin Dataset. The goal is to predict whether a tumor is malignant or benign based on various medical features.

# Tools Used

* Python
* Pandas for data handling
* Scikit-learn for model training and evaluation
* Matplotlib for plotting the ROC and sigmoid curves
* NumPy for mathematical operations

# Key Features

# Data Loading & Preprocessing
  * Uses the Breast Cancer dataset from scikit-learn.
  * Splits into training and testing sets.
  * Applies standard scaling to features.

# Model Training

Trains a Logistic Regression model on the processed data.

# Model Evaluation

  * Outputs confusion matrix, precision, recall, F1-score.
  * Calculates and plots the ROC curve.
  * Computes ROC-AUC score.

# Threshold Tuning

* Demonstrates how changing the classification threshold affects predictions.

# Sigmoid Function Demonstration

* Shows how the sigmoid function transforms raw decision scores into probabilities.
* Includes a plot of the sigmoid function for visual understanding.

# Outputs

* Terminal outputs for performance metrics and sigmoid calculations.
* Graphical outputs for ROC curve and sigmoid curve.
