import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load Dataset
file_path = '/kaggle/input/datasets/abulichlupon/lab1dataset/data_banknote_authentication.txt'
column_names = ['variance', 'skewness', 'curtosis', 'entropy', 'class']
df = pd.read_csv(file_path, names=column_names)

# 2. Features & Target
X = df.drop('class', axis=1)
y = df['class']

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Scaling (not required for RF, but kept for consistency)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. Train Random Forest Model
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
rf_model.fit(X_train, y_train)

# 6. Predictions
y_pred = rf_model.predict(X_test)

# 7. Accuracy
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}")

# 8. Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 9. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ============================================================
# 10. "Loss Curve" (Error vs Number of Trees)
# ============================================================

train_errors = []
test_errors = []
n_trees_range = range(10, 201, 10)

for n_trees in n_trees_range:
    model = RandomForestClassifier(
        n_estimators=n_trees,
        random_state=42
    )
    model.fit(X_train, y_train)

    train_errors.append(1 - model.score(X_train, y_train))
    test_errors.append(1 - model.score(X_test, y_test))

# Plot Loss Curve
plt.figure(figsize=(8, 5))
plt.plot(n_trees_range, train_errors, label="Train Error")
plt.plot(n_trees_range, test_errors, label="Test Error")
plt.xlabel("Number of Trees (n_estimators)")
plt.ylabel("Error Rate")
plt.title("Random Forest Loss Curve")
plt.legend()
plt.grid()
plt.show()
