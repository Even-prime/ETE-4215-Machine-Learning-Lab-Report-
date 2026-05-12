import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load Dataset
file_path = '/kaggle/input/datasets/abulichlupon/lab1dataset/data_banknote_authentication.txt'
column_names = ['variance', 'skewness', 'curtosis', 'entropy', 'class']
df = pd.read_csv(file_path, names=column_names)

# 2. Features & Target
X = df.drop('class', axis=1)
y = df['class']

# 3. Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Scaling (optional for DT, but kept for consistency)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. Train Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# 6. Predictions
y_pred = dt_model.predict(X_test)

# 7. Accuracy
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}")

# 8. Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 9. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Decision Tree")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ============================================================
# 10. "Loss Curve" using pruning path (Cost Complexity Pruning)
# ============================================================

path = dt_model.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas
impurities = path.impurities

train_scores = []
test_scores = []

for alpha in ccp_alphas:
    model = DecisionTreeClassifier(random_state=42, ccp_alpha=alpha)
    model.fit(X_train, y_train)
    
    train_scores.append(model.score(X_train, y_train))
    test_scores.append(model.score(X_test, y_test))

# Plot "Loss Curve" (Error = 1 - Accuracy)
plt.figure(figsize=(8, 5))
plt.plot(ccp_alphas, 1 - np.array(train_scores), label="Train Error")
plt.plot(ccp_alphas, 1 - np.array(test_scores), label="Test Error")
plt.xlabel("ccp_alpha (Pruning Strength)")
plt.ylabel("Error Rate")
plt.title("Decision Tree Loss Curve (Pruning Path)")
plt.legend()
plt.grid()
plt.show()
