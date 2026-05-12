import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA

# 1. Load the dataset
# Make sure to verify this path in your Kaggle "Data" sidebar
try:
    df = pd.read_csv('/kaggle/input/datasets/organizations/uciml/adult-census-income/adult.csv')
    print("Dataset loaded successfully!")
except FileNotFoundError:
    # Alternative common path for this dataset on Kaggle
    df = pd.read_csv('/kaggle/input/adult-income-dataset/adult.csv')
    print("Dataset loaded via alternative path!")

# 2. Preprocessing & Cleaning
# Replace '?' with NaN and drop missing values
df = df.replace('?', np.nan).dropna()

# Encode categorical variables using LabelEncoder
le = LabelEncoder()
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# 3. Feature Scaling
# K-Means is distance-based, so scaling is mandatory
scaler = StandardScaler()
# We drop 'income' because clustering is unsupervised
X = df.drop('income', axis=1) 
X_scaled = scaler.fit_transform(X)

# 4. Elbow Method to find optimal K
wcss = []
print("Calculating Elbow Method... please wait.")
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', n_init='auto', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot the Elbow Graph
plt.figure(figsize=(10, 5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--', color='teal')
plt.title('The Elbow Method (Optimal K Discovery)')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS (Inertia)')
plt.grid(True)
plt.show()

# 5. Execute K-Means
# Based on the elbow, 4 clusters is usually a good choice for this dataset
k = 4 
kmeans = KMeans(n_clusters=k, init='k-means++', n_init='auto', random_state=42)
cluster_labels = kmeans.fit_predict(X_scaled)

# Add labels back to the original dataframe for analysis
df['Cluster'] = cluster_labels

# 6. Visualization using PCA
# Reducing 14 dimensions to 2 for plotting
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(12, 8))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df['Cluster'], 
                palette='viridis', s=50, alpha=0.5)
plt.title(f'K-Means Clustering Result (k={k}) Visualized via PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Cluster')
plt.show()

# 7. Cluster Profile Analysis
print("\n--- Mean Values per Cluster ---")
print(df.groupby('Cluster').mean())
