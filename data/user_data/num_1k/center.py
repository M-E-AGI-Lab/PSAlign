import pandas as pd
import numpy as np

# Read data
df = pd.read_csv('clustering/users_cluster.csv')

# Ensure User_ID is string type with leading zeros
df['User_ID'] = df['User_ID'].astype(str).str.zfill(7)

# List to store results
result_rows = []

# Group by Cluster_ID
for cluster_id, group in df.groupby('Cluster_ID'):
    # Calculate t-SNE center
    center_x = group['tsne_x'].mean()
    center_y = group['tsne_y'].mean()
    
    # Calculate Euclidean distance to center
    distances = np.sqrt((group['tsne_x'] - center_x)**2 + (group['tsne_y'] - center_y)** 2)
    
    # Find index of row closest to center
    min_index = distances.idxmin()
    
    # Get corresponding row data
    center_user_row = df.loc[min_index]
    
    # Add to result list
    result_rows.append(center_user_row)

# Convert results to new DataFrame
result_df = pd.DataFrame(result_rows)

# Output results
print(result_df)

# Save to CSV
result_df.to_csv('clustering/center_users.csv', index=False)
    