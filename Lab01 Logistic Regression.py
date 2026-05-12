import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# 1. Load the Dataset
file_path = '/kaggle/input/datasets/abulichlupon/lab1dataset/data_banknote_authentication.txt'
column_names = ['variance', 'skewness', 'curtosis', 'entropy', 'class']
df = pd.read_csv(file_path, names=column_names)

# 2. Feature and Target Selection
X = df.drop('class', axis=1)
y = df['class']

# 3. Data Splitting (80% Training, 20% Testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# 4. Feature Scaling 
# Scaling is recommended for Logistic Regression to speed up convergence
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. Implementation of Logistic Regression
log_model = LogisticRegression()
log_model.fit(X_train, y_train)

# 6. Predictions
y_pred = log_model.predict(X_test)

# 7. Evaluation
print("--- Logistic Regression Performance ---")
print(f"Accuracy Score: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 8. Confusion Matrix Visualization
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Authentic', 'Forged'], 
            yticklabels=['Authentic', 'Forged'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix: Logistic Regression')
plt.show()

# 9. Display Model Coefficients (Weight of each feature)
weights = pd.DataFrame({'Feature': column_names[:-1], 'Weight': log_model.coef_[0]})
print("\nFeature Importance (Coefficients):")
print(weights.sort_values(by='Weight', ascending=False))
