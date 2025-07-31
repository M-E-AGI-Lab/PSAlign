import pandas as pd
import numpy as np


def find_cluster_center_users(input_csv: str, output_csv: str) -> pd.DataFrame:
    """
    Find the user closest to the center of each cluster based on t-SNE coordinates
    
    Args:
        input_csv: Path to input CSV file with clustering results
        output_csv: Path to save center users CSV file
    
    Returns:
        DataFrame containing center users for each cluster
    """
    # Load clustering data
    df = pd.read_csv(input_csv)
    
    # Ensure User_ID is string with zero-padded format
    df['User_ID'] = df['User_ID'].astype(str).str.zfill(7)
    
    # Store results
    center_users = []
    
    # Process each cluster
    for cluster_id, group in df.groupby('Cluster_ID'):
        # Calculate t-SNE cluster center
        center_x = group['tsne_x'].mean()
        center_y = group['tsne_y'].mean()
        
        # Calculate Euclidean distance to center
        distances = np.sqrt((group['tsne_x'] - center_x)**2 + (group['tsne_y'] - center_y)**2)
        
        # Find closest user to center
        min_index = distances.idxmin()
        center_user = df.loc[min_index]
        
        center_users.append(center_user)
    
    # Create result DataFrame
    result_df = pd.DataFrame(center_users)
    
    # Save to CSV
    result_df.to_csv(output_csv, index=False)
    
    return result_df


def main():
    """Main function to find cluster center users"""
    # Configuration
    input_file = 'clustering/users_cluster.csv'
    output_file = 'clustering/center_users.csv'
    
    # Find center users
    result_df = find_cluster_center_users(input_file, output_file)
    
    # Display results
    print("Cluster center users:")
    print(result_df)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()