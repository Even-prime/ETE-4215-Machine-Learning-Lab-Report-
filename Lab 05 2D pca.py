import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. Prepare Data (Using the numeric features from your previous step)
# Assuming X is your numeric dataframe
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Apply PCA for 2 Components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 3. Apply K-Means on the PCA results
# Let's assume best_k was determined to be 3
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_pca)

# 4. Visualization
plt.figure(figsize=(10, 7))

# Scatter plot of the two PCA components
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', s=50, alpha=0.7)

# Plotting the Centroids
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centroids')

# Formatting the plot
plt.title('2D PCA with K-Means Clustering', fontsize=14)
plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
plt.legend()
plt.colorbar(scatter, label='Cluster ID')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# Print Variance Summary
print(f"Total variance captured by 2 PCA components: {np.sum(pca.explained_variance_ratio_)*100:.2f}%")
