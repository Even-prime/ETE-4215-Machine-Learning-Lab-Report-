import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 1. Load and Clean
df = pd.read_csv('/kaggle/input/datasets/organizations/uciml/adult-census-income/adult.csv')
df.replace('?', np.nan, inplace=True)
df = df.dropna()

# 2. Setup Features and a Multi-class Target
# We use 'relationship' because it has 6 classes, allowing for up to 5 LDA components
numeric_cols = ['age', 'fnlwgt', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']
X = df[numeric_cols]
y_multi = df['relationship'] 

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

le = LabelEncoder()
y_encoded = le.fit_transform(y_multi)

# 3. Apply 2D LDA
# This works now because relationship classes > 2
lda = LDA(n_components=2)
X_lda = lda.fit_transform(X_scaled, y_encoded)

# 4. K-Means Clustering on the LDA result
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_lda)

# 5. Visualization
plt.figure(figsize=(10, 7))
scatter = plt.scatter(X_lda[:, 0], X_lda[:, 1], c=clusters, cmap='viridis', s=50, alpha=0.6)

# Plot Centroids
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centroids')

plt.title('2D LDA + K-Means Clustering (Target: Relationship)', fontsize=14)
plt.xlabel('Linear Discriminant 1')
plt.ylabel('Linear Discriminant 2')
plt.legend()
plt.colorbar(scatter, label='K-Means Cluster')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

print(f"Variance explained by 2 LDs: {np.sum(lda.explained_variance_ratio_)*100:.2f}%")
