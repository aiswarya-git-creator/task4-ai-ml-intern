import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
print("Dataset Preview:")
print(df.head())
print("\nTarget Value Counts:")
print(df['target'].value_counts())
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = LogisticRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
roc_auc = roc_auc_score(y_test, y_prob)
print(f"\nROC-AUC Score: {roc_auc:.2f}")
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='red', label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='green', linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.show()
threshold = 0.3
y_pred_custom = (y_prob >= threshold).astype(int)
print(f"\nConfusion Matrix with Threshold = {threshold}:")
print(confusion_matrix(y_test, y_pred_custom))
print("\nClassification Report with Custom Threshold:")
print(classification_report(y_test, y_pred_custom))
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
z_values = model.decision_function(X_test_scaled)
sigmoid_probs = sigmoid(z_values)
print("\nSigmoid Function Example (first 5 predictions):")
for i in range(5):
    print(f"z = {z_values[i]:.4f} → sigmoid(z) = {sigmoid_probs[i]:.4f}")
z = np.linspace(-10, 10, 100)
sigmoid_vals = sigmoid(z)
plt.figure(figsize=(8, 5))
plt.plot(z, sigmoid_vals, color='green')
plt.title("Sigmoid Function")
plt.xlabel("z")
plt.ylabel("sigmoid(z)")
plt.grid(True)
plt.axhline(0.5, color='gray', linestyle='--', linewidth=0.8)
plt.axvline(0, color='gray', linestyle='--', linewidth=0.8)
plt.show()
print("\nNote:")
print("Logistic Regression uses the Sigmoid function to convert scores to probabilities.")
print("Sigmoid: 1 / (1 + e^-z), where z is a linear combination of inputs.")
print("You can adjust the threshold to control precision vs recall trade-off.")
input("\nPress Enter to exit...")
