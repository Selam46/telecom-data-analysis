import os
from dotenv import load_dotenv
import psycopg2
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from kneed import KneeLocator


def connect_to_database():
    """Establish connection to PostgreSQL database"""
    try:
        conn = psycopg2.connect(
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT")
        )
        return conn
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None


def create_output_directory():
    """Create directory for output files"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'telecom_analysis_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'data'), exist_ok=True)
    return output_dir

# Previous functions remain the same...

def get_user_engagement_metrics():
    """Get user engagement metrics from database"""
    conn = connect_to_database()
    if conn is None:
        return None
    
    try:
        query = """
        SELECT 
            "MSISDN/Number" as user_id,
            COUNT(*) as session_frequency,
            SUM("Dur. (ms)") as total_duration,
            SUM("Total DL (Bytes)" + "Total UL (Bytes)") as total_traffic,
            SUM("Social Media DL (Bytes)" + "Social Media UL (Bytes)") as social_media_traffic,
            SUM("Google DL (Bytes)" + "Google UL (Bytes)") as google_traffic,
            SUM("Email DL (Bytes)" + "Email UL (Bytes)") as email_traffic,
            SUM("Youtube DL (Bytes)" + "Youtube UL (Bytes)") as youtube_traffic,
            SUM("Netflix DL (Bytes)" + "Netflix UL (Bytes)") as netflix_traffic,
            SUM("Gaming DL (Bytes)" + "Gaming UL (Bytes)") as gaming_traffic,
            SUM("Other DL (Bytes)" + "Other UL (Bytes)") as other_traffic
        FROM xdr_data
        GROUP BY "MSISDN/Number"
        """
        
        df = pd.read_sql_query(query, conn)
        return df
    finally:
        conn.close()

def perform_kmeans_clustering(df, k=3):
    """Perform k-means clustering on normalized engagement metrics"""
    # Select engagement metrics
    engagement_metrics = ['session_frequency', 'total_duration', 'total_traffic']
    
    # Normalize the metrics
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(df[engagement_metrics])
    
    # Perform k-means clustering
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(normalized_data)
    
    # Add cluster labels to dataframe
    df['cluster'] = clusters
    
    return df, normalized_data

def find_optimal_k(normalized_data, max_k=10):
    """Find optimal number of clusters using elbow method"""
    inertias = []
    
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(normalized_data)
        inertias.append(kmeans.inertia_)
    
    # Use KneeLocator to find the elbow point
    kl = KneeLocator(range(1, max_k + 1), inertias, curve='convex', direction='decreasing')
    
    return kl.elbow, inertias

def analyze_clusters(df):
    """Compute statistics for each cluster"""
    metrics = ['session_frequency', 'total_duration', 'total_traffic']
    cluster_stats = df.groupby('cluster')[metrics].agg(['min', 'max', 'mean', 'sum'])
    return cluster_stats

def get_top_users_per_metric(df, metrics, n=10):
    """Get top 10 users per engagement metric"""
    top_users = {}
    for metric in metrics:
        top_users[metric] = df.nlargest(n, metric)[['user_id', metric]]
    return top_users

def get_top_users_per_application(df, n=10):
    """Get top 10 users per application"""
    app_columns = ['social_media_traffic', 'google_traffic', 'email_traffic',
                  'youtube_traffic', 'netflix_traffic', 'gaming_traffic', 'other_traffic']
    
    top_users = {}
    for app in app_columns:
        top_users[app] = df.nlargest(n, app)[['user_id', app]]
    return top_users

def plot_engagement_clusters(normalized_data, clusters, output_dir):
    """Plot clustering results"""
    # PCA for visualization
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(normalized_data)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=clusters, cmap='viridis')
    plt.title('User Engagement Clusters (PCA visualization)')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.colorbar(scatter)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plots', 'engagement_clusters.png'))
    plt.close()

def plot_elbow_curve(k_range, inertias, optimal_k, output_dir):
    """Plot elbow curve for k-means"""
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertias, 'bx-')
    plt.vlines(optimal_k, plt.ylim()[0], plt.ylim()[1], colors='r', linestyles='--', label=f'Optimal k={optimal_k}')
    plt.xlabel('k')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plots', 'elbow_curve.png'))
    plt.close()

def plot_top_applications(df, output_dir):
    """Plot top 3 most used applications"""
    app_columns = ['social_media_traffic', 'google_traffic', 'email_traffic',
                  'youtube_traffic', 'netflix_traffic', 'gaming_traffic', 'other_traffic']
    
    total_usage = df[app_columns].sum().sort_values(ascending=False)
    top_3_apps = total_usage.head(3)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=top_3_apps.index, y=top_3_apps.values)
    plt.title('Top 3 Most Used Applications')
    plt.xticks(rotation=45)
    plt.ylabel('Total Traffic (Bytes)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plots', 'top_applications.png'))
    plt.close()


def export_results(results_dict, output_dir):
    """Export analysis results to CSV files"""
    for name, df in results_dict.items():
        if isinstance(df, pd.DataFrame):
            df.to_csv(os.path.join(output_dir, 'data', f'{name}.csv'))
        elif isinstance(df, np.ndarray):
            pd.DataFrame(df).to_csv(os.path.join(output_dir, 'data', f'{name}.csv'))


def perform_engagement_analysis(output_dir):
    """Perform complete engagement analysis"""
    # Get user engagement metrics
    print("Fetching user engagement metrics...")
    df = get_user_engagement_metrics()
    
    # Get top users per engagement metric
    print("Analyzing top users per metric...")
    engagement_metrics = ['session_frequency', 'total_duration', 'total_traffic']
    top_users = get_top_users_per_metric(df, engagement_metrics)
    
    # Normalize and perform initial clustering
    print("Performing k-means clustering...")
    df_clustered, normalized_data = perform_kmeans_clustering(df)
    
    # Find optimal k
    print("Finding optimal number of clusters...")
    optimal_k, inertias = find_optimal_k(normalized_data)
    plot_elbow_curve(range(1, 11), inertias, optimal_k, output_dir)
    
    # Perform clustering with optimal k
    print(f"Performing clustering with optimal k={optimal_k}...")
    df_clustered, normalized_data = perform_kmeans_clustering(df, optimal_k)
    
    # Analyze clusters
    cluster_stats = analyze_clusters(df_clustered)
    
    # Get top users per application
    print("Analyzing top users per application...")
    top_users_per_app = get_top_users_per_application(df)
    
    # Create visualizations
    print("Creating visualizations...")
    plot_engagement_clusters(normalized_data, df_clustered['cluster'], output_dir)
    plot_top_applications(df, output_dir)
    
    # Export results
    results = {
        'engagement_metrics': df,
        'cluster_statistics': cluster_stats,
        'optimal_k': pd.DataFrame({'optimal_k': [optimal_k]}),
    }
    
    for metric, users in top_users.items():
        results[f'top_users_{metric}'] = users
    
    for app, users in top_users_per_app.items():
        results[f'top_users_{app}'] = users
    
    export_results(results, output_dir)
    
    return results

def main():
    # Create output directory
    output_dir = create_output_directory()
    
    # Perform Task 1 analysis (previous implementation)
    # ...
    
    # Perform Task 2 - User Engagement Analysis
    print("\nPerforming User Engagement Analysis...")
    engagement_results = perform_engagement_analysis(output_dir)
    
    print(f"\nAnalysis complete! Results have been saved to: {output_dir}")
    print("- Plots can be found in the 'plots' subdirectory")
    print("- Data exports can be found in the 'data' subdirectory")

if __name__ == "__main__":
    main()