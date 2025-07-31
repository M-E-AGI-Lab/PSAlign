import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# === Global font settings ===
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 24
})

# === Configuration ===
embedding_dir = 'embeddings'
csv_path = './users.csv'
output_dir = 'clustering'
try_k_range = range(2, 10)
show_labels_on_plot = True
max_points_to_plot = 1000

os.makedirs(output_dir, exist_ok=True)

colors = sns.color_palette("husl", 5)
custom_cmap = ListedColormap(colors)

# === Load user embeddings ===
user_embeddings, user_ids = [], []
for fname in sorted(os.listdir(embedding_dir))[:max_points_to_plot]:
    if fname.endswith('.npy'):
        emb = np.load(os.path.join(embedding_dir, fname)).squeeze()
        user_embeddings.append(emb)
        user_ids.append(fname.replace('.npy', ''))
user_embeddings = np.vstack(user_embeddings)

# === Standardization ===
user_embeddings = StandardScaler().fit_transform(user_embeddings)

# === Automatic cluster number evaluation ===
inertias, silhouette_scores = [], []
for k in try_k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(user_embeddings)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(user_embeddings, labels))

# === Evaluation plots ===
plt.figure(figsize=(20, 4))
plt.subplot(1, 2, 1)
plt.plot(try_k_range, inertias, 'bo-')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(try_k_range, silhouette_scores, 'go-')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'cluster_evaluation.png'), dpi=300)
plt.close()

# === Clustering and dimensionality reduction ===
best_k = 5
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init='auto')
labels = kmeans.fit_predict(user_embeddings)

# Find center users for each cluster
centroid_indices = []
for cluster_id in range(best_k):
    cluster_points = np.where(labels == cluster_id)[0]
    cluster_center = kmeans.cluster_centers_[cluster_id]
    distances = np.linalg.norm(user_embeddings[cluster_points] - cluster_center, axis=1)
    centroid_index = cluster_points[np.argmin(distances)]  # Closest user
    centroid_indices.append(centroid_index)

tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=1000, random_state=42)
reduced_tsne = tsne.fit_transform(user_embeddings)

# === t-SNE visualization ===
plt.figure(figsize=(12, 8))
scatter = plt.scatter(reduced_tsne[:, 0], reduced_tsne[:, 1], c=labels, cmap=custom_cmap, s=50, alpha=0.8)

# Highlight center users of each cluster
for cluster_id, centroid_index in enumerate(centroid_indices):
    user_id = user_ids[centroid_index]
    plt.scatter(reduced_tsne[centroid_index, 0], reduced_tsne[centroid_index, 1], 
                color=colors[cluster_id], s=200, edgecolor='black', marker='o')  # Larger with border
    plt.text(reduced_tsne[centroid_index, 0]+14, reduced_tsne[centroid_index, 1]-3, 
             f"Center User\n ID: {user_id}", fontsize=20, alpha=1, color='black', fontweight='bold', ha='center')
# Set axis ranges
plt.xlim([reduced_tsne[:, 0].min() - 5, reduced_tsne[:, 0].max() + 10])  # Expand x-axis range
plt.ylim([reduced_tsne[:, 1].min() - 5, reduced_tsne[:, 1].max() + 5])  # Expand y-axis range

plt.xlabel('t-SNE dim 1')
plt.ylabel('t-SNE dim 2')
plt.grid(True)
plt.colorbar(scatter, label='Cluster ID')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'embeds_tsne_kmeans.png'), dpi=1000, bbox_inches='tight', pad_inches=0)
plt.close()

# === PCA visualization ===
reduced_pca = PCA(n_components=2).fit_transform(user_embeddings)

plt.figure(figsize=(12, 8))
scatter = plt.scatter(reduced_pca[:, 0], reduced_pca[:, 1], c=labels, cmap=custom_cmap, s=50, alpha=0.8)

# Highlight center users of each cluster
for cluster_id, centroid_index in enumerate(centroid_indices):
    user_id = user_ids[centroid_index]
    plt.scatter(reduced_pca[centroid_index, 0], reduced_pca[centroid_index, 1], 
                color=colors[cluster_id], s=200, edgecolor='black', marker='o')  # Larger with border
    plt.text(reduced_pca[centroid_index, 0]+14, reduced_pca[centroid_index, 1]-3, 
             f"Center User\n ID: {user_id}", fontsize=20, alpha=1, color='black', fontweight='bold', ha='center')
# Set axis ranges
plt.xlim([reduced_pca[:, 0].min() - 10, reduced_pca[:, 0].max() + 10])  # Expand x-axis range
plt.ylim([reduced_pca[:, 1].min() - 10, reduced_pca[:, 1].max() + 10])  # Expand y-axis range

plt.xlabel('PCA dim 1')
plt.ylabel('PCA dim 2')
plt.grid(True)
plt.colorbar(scatter, label='Cluster ID')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'embeds_pca_kmeans.png'), dpi=1000, bbox_inches='tight', pad_inches=0)
plt.close()

# === Generate result CSV ===
df_csv = pd.read_csv(csv_path, dtype=str).fillna("None")
df_csv.index = [f"{i+1:07d}" for i in range(len(df_csv))]

records = []
for i, uid in enumerate(user_ids):
    row = df_csv.loc[uid]
    record = {
        'User_ID': uid, 'Cluster_ID': labels[i],
        'tsne_x': reduced_tsne[i, 0], 'tsne_y': reduced_tsne[i, 1],
        'pca_x': reduced_pca[i, 0], 'pca_y': reduced_pca[i, 1],
        'Banned_Categories': row['Banned_Categories'],
        'Allowed_Categories': row['Allowed_Categories'],
    }
    for col in ['Age', 'Age_Group', 'Gender', 'Religion', 'Mental_Condition', 'Physical_Condition']:
        record[col] = row[col]
    records.append(record)

pd.DataFrame(records).to_csv(os.path.join(output_dir, 'users_cluster.csv'), index=False)
print(f"✅ t-SNE and PCA analysis completed. Images and data saved in: {output_dir}")
