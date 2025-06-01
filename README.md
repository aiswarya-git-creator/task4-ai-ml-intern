# This project implements a binary classification model using Logistic Regression on the Breast Cancer Wisconsin Dataset. The goal is to predict whether a tumor is malignant or benign based on various medical features.

# Tools Used

1.Python
2.Pandas for data handling
3.Scikit-learn for modeling and evaluation
4.Matplotlib for plotting the ROC curve

# Key Steps

1.Load and explore the dataset
2.Preprocess data with train/test split and feature standardization
3.Train a logistic regression model
4.Evaluate performance using confusion matrix, precision, recall, F1-score, and ROC-AUC
5.Demonstrate the impact of changing the classification threshold
6.Provide insight into how the sigmoid function works in logistic regression

# Output 

* Red Curve (ROC Curve): Shows the trade-off between True Positive Rate (TPR) and False Positive Rate (FPR) at different thresholds.
* Green Dashed Line: Represents a random classifier (AUC = 0.5). Any model performing better should lie above this line.
* AUC = 1.00: This indicates perfect classification on your test set — which means the model predicted all instances correctly. While this may happen with clean datasets like the Breast Cancer Wisconsin dataset, it’s rare in real-world data and can sometimes suggest overfitting.
