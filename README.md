# This project implements a binary classification model using Logistic Regression on the Breast Cancer Wisconsin Dataset. The goal is to predict whether a tumor is malignant or benign based on various medical features.

# Tools Used

* Python
* Pandas for data handling
* Scikit-learn for modeling and evaluation
* Matplotlib for plotting the ROC curve

# Key Steps

* Load and explore the dataset
* Preprocess data with train/test split and feature standardization
* Train a logistic regression model
* Evaluate performance using confusion matrix, precision, recall, F1-score, and ROC-AUC
* Demonstrate the impact of changing the classification threshold
* Provide insight into how the sigmoid function works in logistic regression

# Output 

* Red Curve (ROC Curve): Shows the trade-off between True Positive Rate (TPR) and False Positive Rate (FPR) at different thresholds.
* Green Dashed Line: Represents a random classifier (AUC = 0.5). Any model performing better should lie above this line.
* AUC = 1.00: This indicates perfect classification on your test set — which means the model predicted all instances correctly. While this may happen with clean datasets like the Breast Cancer Wisconsin dataset, it’s rare in real-world data and can sometimes suggest overfitting.
