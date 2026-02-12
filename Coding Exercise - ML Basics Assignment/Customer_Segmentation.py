import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 1. Load the Data
try:
    df = pd.read_csv('customer_spending.csv')
    print(f"Loaded {len(df)} records successfully.")
except FileNotFoundError:
    print("File not found! Please make sure 'customer_spending.csv' is in your files.")
    # Create an empty DataFrame to prevent crash, but code will stop later
    df = pd.DataFrame()

# --- FIX 1: Removed the line 'df = pd.DataFrame(data)' ---
# That line was overwriting your CSV data with a variable 'data' 
# that likely doesn't exist in this specific code block.

# 2. Preprocess data
if not df.empty:
    features = ['annual_spending', 'purchase_frequency', 'age']
    X = df[features]
    
    # Scale the features (Important for K-Means!)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. Determine optimal number of clusters using elbow method
    inertia = []
    K = range(1, 10)  # Expanded range to 10 for better view

    # --- FIX 2: Fixed Indentation for the loop ---
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)

    # 4. Plot elbow curve
    plt.figure(figsize=(8, 5))
    plt.plot(K, inertia, 'bo-')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Inertia (Sum of Squared Distances)')
    plt.title('Elbow Method for Optimal K')
    plt.grid(True)
    plt.show() # Added show() so you see it in Colab immediately

    # 5. Apply K-Means with optimal K (e.g., 3)
    optimal_k = 3
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_scaled)

    # 6. Analyze clusters
    # Group by cluster to see the average stats for each group
    cluster_summary = df.groupby('cluster')[features].mean().round(2)
    print("\nCluster Characteristics:")
    print(cluster_summary)

    # 7. Example of targeted strategies
    print("\nTargeted Strategies:")
    for cluster in range(optimal_k):
        # We look at the average spending of the cluster to decide the strategy
        avg_spend = cluster_summary.loc[cluster, 'annual_spending']
        avg_freq = cluster_summary.loc[cluster, 'purchase_frequency']
        
        print(f"Cluster {cluster}:", end=" ")
        
        if avg_spend > 2000: # Adjusted threshold for the realistic data we made
            print("High Rollers -> Offer exclusive VIP status and luxury bundles.")
        elif avg_freq > 20:
            print("Loyal Frequent Buyers -> Provide subscription discounts.")
        else:
            print("Casual/Low Value -> Send 'We miss you' coupons.")

    # 8. Save cluster assignments to CSV
    df.to_csv('customer_segments.csv', index=False)
    print("\nSegments saved to 'customer_segments.csv'.")
