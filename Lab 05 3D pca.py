import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D

# 1. Scaling the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Apply PCA for 3 Components
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

# 3. Apply K-Means
n_clusters = 3  # You can adjust this based on your silhouette score
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_pca)

# 4. 3D Visualization
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot
# x = PC1, y = PC2, z = PC3
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], 
                     c=clusters, cmap='viridis', s=40, alpha=0.6)

# Plot Centroids
centroids = kmeans.cluster_centers_
ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], 
           c='red', marker='X', s=200, label='Centroids')

# Labeling axes with variance ratios
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)')

plt.title('3D PCA with K-Means Clustering', fontsize=15)
plt.legend()
plt.show()

print(f"Total variance captured by 3 PCA components: {np.sum(pca.explained_variance_ratio_)*100:.2f}%")
