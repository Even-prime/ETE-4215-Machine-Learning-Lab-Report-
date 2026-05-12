import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load the Dataset
file_path = '/kaggle/input/datasets/abulichlupon/lab1dataset/data_banknote_authentication.txt'
column_names = ['variance', 'skewness', 'curtosis', 'entropy', 'class']
df = pd.read_csv(file_path, names=column_names)

# 2. Feature and Target Selection
X = df.drop('class', axis=1)
y = df['class']

# 3. Data Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# 4. Feature Scaling 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --- Error Rate Calculation (K-NN's version of a Loss Curve) ---
error_rate = []

# Iterating through different K values to find the best fit
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))

# Plotting the Error Rate Curve (Cleaned)
plt.figure(figsize=(10, 6))
plt.plot(range(1, 40), error_rate, color='blue') # Markers and dashes removed
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.show()

# 5. Final Model Implementation
# (You can adjust n_neighbors based on the plot results)
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

# 6. Predictions
y_pred_knn = knn_model.predict(X_test)

# 7. Evaluation
print("--- K-Nearest Neighbors Performance ---")
print(f"Accuracy Score: {accuracy_score(y_test, y_pred_knn):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_knn))

# 8. Confusion Matrix Visualization
cm_knn = confusion_matrix(y_test, y_pred_knn)
plt.figure(figsize=(6,4))
sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Greens', 
            xticklabels=['Authentic', 'Forged'], 
            yticklabels=['Authentic', 'Forged'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix: K-NN')
plt.show()
