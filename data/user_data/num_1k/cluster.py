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

# Global font settings
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 24
})


def load_embeddings(embedding_dir: str, max_points: int = 1000) -> tuple[np.ndarray, list]:
    """
    Load user embeddings from directory
    
    Args:
        embedding_dir: Directory containing embedding .npy files
        max_points: Maximum number of embeddings to load
    
    Returns:
        tuple: (embeddings array, user IDs list)
    """
    user_embeddings, user_ids = [], []
    
    for fname in sorted(os.listdir(embedding_dir))[:max_points]:
        if fname.endswith('.npy'):
            emb = np.load(os.path.join(embedding_dir, fname)).squeeze()
            user_embeddings.append(emb)
            user_ids.append(fname.replace('.npy', ''))
    
    return np.vstack(user_embeddings), user_ids


def evaluate_cluster_numbers(embeddings: np.ndarray, k_range: range) -> tuple[list, list]:
    """
    Evaluate different numbers of clusters using inertia and silhouette score
    
    Args:
        embeddings: Standardized user embeddings
        k_range: Range of cluster numbers to try
    
    Returns:
        tuple: (inertias list, silhouette scores list)
    """
    inertias, silhouette_scores = [], []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(embeddings)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(embeddings, labels))
    
    return inertias, silhouette_scores


def plot_cluster_evaluation(k_range: range, inertias: list, silhouette_scores: list, output_dir: str):
    """
    Plot cluster evaluation metrics
    
    Args:
        k_range: Range of cluster numbers
        inertias: Inertia values for each k
        silhouette_scores: Silhouette scores for each k
        output_dir: Output directory for saving plots
    """
    plt.figure(figsize=(20, 4))
    
    # Plot inertia
    plt.subplot(1, 2, 1)
    plt.plot(k_range, inertias, 'bo-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.grid(True)
    
    # Plot silhouette score
    plt.subplot(1, 2, 2)
    plt.plot(k_range, silhouette_scores, 'go-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cluster_evaluation.png'), dpi=300)
    plt.close()


def find_cluster_centroids(embeddings: np.ndarray, labels: np.ndarray, kmeans: KMeans) -> list:
    """
    Find the user closest to each cluster center
    
    Args:
        embeddings: User embeddings
        labels: Cluster labels
        kmeans: Fitted KMeans model
    
    Returns:
        list: Indices of centroid users for each cluster
    """
    centroid_indices = []
    n_clusters = len(kmeans.cluster_centers_)
    
    for cluster_id in range(n_clusters):
        cluster_points = np.where(labels == cluster_id)[0]
        cluster_center = kmeans.cluster_centers_[cluster_id]
        distances = np.linalg.norm(embeddings[cluster_points] - cluster_center, axis=1)
        centroid_index = cluster_points[np.argmin(distances)]
        centroid_indices.append(centroid_index)
    
    return centroid_indices


def plot_dimensionality_reduction(reduced_data: np.ndarray, labels: np.ndarray, centroid_indices: list,
                                user_ids: list, colors: list, method: str, output_dir: str):
    """
    Plot dimensionality reduction results with cluster centers highlighted
    
    Args:
        reduced_data: 2D reduced data points
        labels: Cluster labels
        centroid_indices: Indices of cluster center users
        user_ids: List of user IDs
        colors: Color palette for clusters
        method: Reduction method name ('t-SNE' or 'PCA')
        output_dir: Output directory for saving plots
    """
    custom_cmap = ListedColormap(colors)
    
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap=custom_cmap, s=50, alpha=0.8)
    
    # Highlight cluster center users
    for cluster_id, centroid_index in enumerate(centroid_indices):
        user_id = user_ids[centroid_index]
        plt.scatter(reduced_data[centroid_index, 0], reduced_data[centroid_index, 1], 
                   color=colors[cluster_id], s=200, edgecolor='black', marker='o')
        plt.text(reduced_data[centroid_index, 0]+14, reduced_data[centroid_index, 1]-3, 
                f"Center User\n ID: {user_id}", fontsize=20, alpha=1, color='black', 
                fontweight='bold', ha='center')
    
    # Set axis ranges with padding
    x_padding = 10 if method == 'PCA' else 5
    y_padding = 10 if method == 'PCA' else 5
    plt.xlim([reduced_data[:, 0].min() - x_padding, reduced_data[:, 0].max() + x_padding])
    plt.ylim([reduced_data[:, 1].min() - y_padding, reduced_data[:, 1].max() + y_padding])
    
    plt.xlabel(f'{method} dim 1')
    plt.ylabel(f'{method} dim 2')
    plt.grid(True)
    plt.colorbar(scatter, label='Cluster ID')
    plt.tight_layout()
    
    filename = f"embeds_{method.lower().replace('-', '')}_kmeans.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=1000, bbox_inches='tight', pad_inches=0)
    plt.close()


def save_clustering_results(user_ids: list, labels: np.ndarray, reduced_tsne: np.ndarray,
                          reduced_pca: np.ndarray, csv_path: str, output_dir: str):
    """
    Save clustering results to CSV file
    
    Args:
        user_ids: List of user IDs
        labels: Cluster labels
        reduced_tsne: t-SNE reduced coordinates
        reduced_pca: PCA reduced coordinates
        csv_path: Path to original user data CSV
        output_dir: Output directory for results
    """
    # Load original user data
    df_csv = pd.read_csv(csv_path, dtype=str).fillna("None")
    df_csv.index = [f"{i+1:07d}" for i in range(len(df_csv))]
    
    # Create records with clustering results
    records = []
    for i, uid in enumerate(user_ids):
        row = df_csv.loc[uid]
        record = {
            'User_ID': uid, 
            'Cluster_ID': labels[i],
            'tsne_x': reduced_tsne[i, 0], 
            'tsne_y': reduced_tsne[i, 1],
            'pca_x': reduced_pca[i, 0], 
            'pca_y': reduced_pca[i, 1],
            'Banned_Categories': row['Banned_Categories'],
            'Allowed_Categories': row['Allowed_Categories'],
        }
        
        # Add user demographic data
        for col in ['Age', 'Age_Group', 'Gender', 'Religion', 'Mental_Condition', 'Physical_Condition']:
            record[col] = row[col]
        
        records.append(record)
    
    # Save to CSV
    pd.DataFrame(records).to_csv(os.path.join(output_dir, 'users_cluster.csv'), index=False)


def main():
    """Main function for user clustering analysis"""
    # Configuration
    config = {
        'embedding_dir': 'embeddings',
        'csv_path': './users.csv',
        'output_dir': 'clustering',
        'try_k_range': range(4, 13),
        'show_labels_on_plot': True,
        'max_points_to_plot': 1000,
        'best_k': 5
    }
    
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Set up color palette
    colors = sns.color_palette("husl", config['best_k'])
    
    # Load and standardize embeddings
    user_embeddings, user_ids = load_embeddings(config['embedding_dir'], config['max_points_to_plot'])
    user_embeddings = StandardScaler().fit_transform(user_embeddings)
    
    # Evaluate different cluster numbers
    inertias, silhouette_scores = evaluate_cluster_numbers(user_embeddings, config['try_k_range'])
    plot_cluster_evaluation(config['try_k_range'], inertias, silhouette_scores, config['output_dir'])
    
    # Perform clustering with best k
    kmeans = KMeans(n_clusters=config['best_k'], random_state=42, n_init='auto')
    labels = kmeans.fit_predict(user_embeddings)
    
    # Find cluster centroids
    centroid_indices = find_cluster_centroids(user_embeddings, labels, kmeans)
    
    # Apply dimensionality reduction
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=1000, random_state=42)
    reduced_tsne = tsne.fit_transform(user_embeddings)
    
    reduced_pca = PCA(n_components=2).fit_transform(user_embeddings)
    
    # Create visualizations
    plot_dimensionality_reduction(reduced_tsne, labels, centroid_indices, user_ids, colors, 't-SNE', config['output_dir'])
    plot_dimensionality_reduction(reduced_pca, labels, centroid_indices, user_ids, colors, 'PCA', config['output_dir'])
    
    # Save clustering results
    save_clustering_results(user_ids, labels, reduced_tsne, reduced_pca, config['csv_path'], config['output_dir'])
    
    print(f"✅ t-SNE and PCA analysis completed. Images and data saved to: {config['output_dir']}")


if __name__ == "__main__":
    main()