import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from mpl_toolkits.mplot3d import Axes3D

# 1. Load and Preprocess
df = pd.read_csv('/kaggle/input/datasets/organizations/uciml/adult-census-income/adult.csv')
df.replace('?', np.nan, inplace=True)
df = df.dropna()

# 2. Select Features and a High-Cardinality Target
# 'occupation' has 14 classes, allowing for up to 13 LDA components
numeric_cols = ['age', 'fnlwgt', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']
X = df[numeric_cols]
y_multi = df['occupation'] 

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

le = LabelEncoder()
y_encoded = le.fit_transform(y_multi)

# 3. Apply 3D LDA
# This will now work because C-1 (14-1) is greater than 3
lda = LDA(n_components=3)
X_lda = lda.fit_transform(X_scaled, y_encoded)

# 4. K-Means Clustering on the 3D LDA Space
n_clusters = 4 
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_lda)

# 5. 3D Visualization
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot of the 3 Linear Discriminants
scatter = ax.scatter(X_lda[:, 0], X_lda[:, 1], X_lda[:, 2], 
                     c=clusters, cmap='viridis', s=30, alpha=0.5)

# Plot Centroids
centroids = kmeans.cluster_centers_
ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], 
           c='red', marker='X', s=200, label='Centroids', depthshade=False)

# Labels
ax.set_title('3D LDA + K-Means Clustering (Target: Occupation)', fontsize=15)
ax.set_xlabel('LD 1')
ax.set_ylabel('LD 2')
ax.set_zlabel('LD 3')
ax.legend()

plt.colorbar(scatter, ax=ax, label='Cluster ID', shrink=0.5)
plt.show()

print(f"Variance explained by 3 LDs: {np.sum(lda.explained_variance_ratio_)*100:.2f}%")
