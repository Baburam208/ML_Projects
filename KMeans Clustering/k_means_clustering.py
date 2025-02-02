import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib import rcParams
from kneed import KneeLocator

# Set font to 'Times New Roman'
rcParams['font.family'] = 'Times New Roman'


def compute_clustering(file_path='annual_return_and_volatility_metrics.csv', k=3):
    # Load the data
    df = pd.read_csv(file_path)

    # Extract relevant data
    if not all(col in df.columns for col in ['annual return', 'annual volatility', 'stock']):
        raise ValueError("The CSV file must contain 'annual return', 'annual volatility', and 'stock' columns.")

    data = df[['annual return', 'annual volatility']].values
    data_label = df['stock'].values

    # Standardize the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # Perform k-means clustering
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(data_scaled)

    # Extract stock names
    try:
        stock_names = [stock.split(' (')[1].split(')')[0].strip() for stock in data_label]
    except IndexError:
        raise ValueError("Ensure stock names are in the format 'Stock Name (TICKER)'.")

    # Create a dictionary for stock cluster labels
    stock_cluster_labels = {stock_name: cluster_label for stock_name, cluster_label in zip(stock_names, cluster_labels)}

    # Group stocks by cluster
    cluster_groups = {}
    for stock_name, cluster_label in zip(stock_names, cluster_labels):
        cluster_groups.setdefault(cluster_label, []).append(stock_name)

    # Print and return grouped stocks
    print("\nStocks in each cluster:")
    for cluster, stocks in cluster_groups.items():
        print(f"Cluster {cluster}: {', '.join(stocks)}")

    # Plot clusters
    plt.figure(figsize=(10, 8))
    for cluster in range(k):
        cluster_data = data_scaled[np.where(cluster_labels == cluster)]
        plt.scatter(
            cluster_data[:, 0],
            cluster_data[:, 1],
            s=100,  # Increase point size
            label=f"Cluster {cluster}"
        )

    # Annotate each stock
    for i, stock in enumerate(stock_names):
        plt.text(
            data_scaled[i, 0],
            data_scaled[i, 1],
            stock,
            fontsize=10,
            alpha=0.7,
            ha='right'  # Align text to the right
        )

    plt.title("Cluster Visualization")
    plt.xlabel("Annual Return (Standardized)")
    plt.ylabel("Annual Volatility (Standardized)")
    plt.legend()
    plt.show()

    return stock_names, cluster_labels, stock_cluster_labels, cluster_groups


def find_optimal_k(file_path='annual_return_and_volatility_metrics.csv'):
    df = pd.read_csv(file_path)
    data = df[['annual return', 'annual volatility']].values

    # Elbow method
    wcss = []
    k_values = range(1, 15)  # Try clustering with 1 to 10 clusters
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)

    # Plot the elbow graph
    plt.figure(figure=(8, 5))
    plt.plot(k_values, wcss, marker='o', linestyle='--')
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
    plt.show()

    # Finding optimal value for K
    # Automatically find the elbow point.
    knee = KneeLocator(k_values, wcss, curve="convex", direction="decreasing")
    print(f"Optimal number of clusters: {knee.knee}")

    # Plot with the elbow highlighted
    knee.plot_knee()

    optimal_k = knee.knee

    return optimal_k


if __name__ == '__main__':
    file_path = 'annual_return_and_volatility_metrics.csv'
    optimal_k = find_optimal_k(file_path=file_path)
    print(f"After computing optimal 'k'")
    compute_clustering(file_path=file_path, k=optimal_k)
    compute_clustering(file_path=file_path, k=4)
