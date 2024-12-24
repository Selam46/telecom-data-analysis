import os
from dotenv import load_dotenv
import psycopg2
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

load_dotenv()

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

def get_experience_metrics():
    """Get user experience metrics from database"""
    conn = connect_to_database()
    if conn is None:
        return None
    
    try:
        query = """
        SELECT 
            "MSISDN/Number" as user_id,
            "Handset Type",
            AVG("TCP DL Retrans. Vol (Bytes)" + "TCP UL Retrans. Vol (Bytes)") as avg_tcp_retrans,
            AVG("Avg RTT DL (ms)" + "Avg RTT UL (ms)") / 2 as avg_rtt,
            AVG("Avg Bearer TP DL (kbps)" + "Avg Bearer TP UL (kbps)") / 2 as avg_throughput
        FROM xdr_data
        GROUP BY "MSISDN/Number", "Handset Type"
        """
        
        df = pd.read_sql_query(query, conn)
        return df
    finally:
        conn.close()

def get_raw_metrics():
    """Get raw metrics for detailed analysis"""
    conn = connect_to_database()
    if conn is None:
        return None
    
    try:
        query = """
        SELECT 
            "TCP DL Retrans. Vol (Bytes)" + "TCP UL Retrans. Vol (Bytes)" as tcp_retrans,
            "Avg RTT DL (ms)" + "Avg RTT UL (ms)" as rtt,
            "Avg Bearer TP DL (kbps)" + "Avg Bearer TP UL (kbps)" as throughput,
            "Handset Type"
        FROM xdr_data
        """
        
        df = pd.read_sql_query(query, conn)
        return df
    finally:
        conn.close()

def clean_experience_data(df):
    """Clean experience metrics data by handling missing values and outliers"""
    # Handle missing values
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].mean())
    
    df['Handset Type'] = df['Handset Type'].fillna(df['Handset Type'].mode()[0])
    
    # Handle outliers using IQR method
    for col in ['avg_tcp_retrans', 'avg_rtt', 'avg_throughput']:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower_bound, upper_bound)
    
    return df

def analyze_metric_distributions(raw_df):
    """Analyze top, bottom, and most frequent values for network metrics"""
    metrics = {
        'tcp_retrans': 'TCP Retransmission',
        'rtt': 'Round Trip Time',
        'throughput': 'Throughput'
    }
    
    results = {}
    for col, metric_name in metrics.items():
        metric_stats = {
            'top_10': raw_df[col].nlargest(10),
            'bottom_10': raw_df[col].nsmallest(10),
            'most_frequent': raw_df[col].value_counts().head(10)
        }
        results[col] = metric_stats
    
    return results

def plot_throughput_distribution(df, output_dir):
    """Plot throughput distribution per handset type"""
    plt.figure(figsize=(15, 8))
    sns.boxplot(x='Handset Type', y='avg_throughput', data=df)
    plt.xticks(rotation=45)
    plt.title('Distribution of Average Throughput per Handset Type')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plots', 'throughput_distribution.png'))
    plt.close()

def plot_tcp_retrans_distribution(df, output_dir):
    """Plot TCP retransmission distribution per handset type"""
    plt.figure(figsize=(15, 8))
    sns.boxplot(x='Handset Type', y='avg_tcp_retrans', data=df)
    plt.xticks(rotation=45)
    plt.title('Distribution of Average TCP Retransmission per Handset Type')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plots', 'tcp_retrans_distribution.png'))
    plt.close()

def perform_experience_clustering(df):
    """Perform k-means clustering on experience metrics"""
    # Select metrics for clustering
    metrics = ['avg_tcp_retrans', 'avg_rtt', 'avg_throughput']
    
    # Normalize the metrics
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(df[metrics])
    
    # Perform k-means clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(normalized_data)
    
    # Add cluster labels to dataframe
    df['experience_cluster'] = clusters
    
    # Calculate cluster characteristics
    cluster_stats = df.groupby('experience_cluster')[metrics].agg(['mean', 'std'])
    
    return df, cluster_stats

def plot_experience_clusters(df, output_dir):
    """Plot experience clusters"""
    # Create a scatter plot matrix
    metrics = ['avg_tcp_retrans', 'avg_rtt', 'avg_throughput']
    
    fig = plt.figure(figsize=(15, 15))
    for i, metric1 in enumerate(metrics):
        for j, metric2 in enumerate(metrics):
            if i != j:
                plt.subplot(3, 3, i * 3 + j + 1)
                scatter = plt.scatter(df[metric1], df[metric2], 
                                   c=df['experience_cluster'], 
                                   cmap='viridis',
                                   alpha=0.6)
                plt.xlabel(metric1)
                plt.ylabel(metric2)
                
    plt.suptitle('Experience Clusters Visualization')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plots', 'experience_clusters.png'))
    plt.close()

def describe_clusters(cluster_stats):
    """Generate descriptions for each experience cluster"""
    descriptions = {
        0: "High Performance Cluster: ",
        1: "Average Performance Cluster: ",
        2: "Poor Performance Cluster: "
    }
    
    # Sort clusters based on overall performance
    performance_ranking = cluster_stats['avg_throughput']['mean'].sort_values(ascending=False)
    
    for i, cluster in enumerate(performance_ranking.index):
        stats = cluster_stats.loc[cluster]
        description = descriptions[i]
        
        description += f"Average throughput: {stats['avg_throughput']['mean']:.2f} kbps, "
        description += f"RTT: {stats['avg_rtt']['mean']:.2f} ms, "
        description += f"TCP retransmission: {stats['avg_tcp_retrans']['mean']:.2f} bytes"
        
        descriptions[i] = description
    
    return descriptions
def export_results(results_dict, output_dir):
    """Export analysis results to CSV files"""
    for name, df in results_dict.items():
        if isinstance(df, pd.DataFrame):
            df.to_csv(os.path.join(output_dir, 'data', f'{name}.csv'))
        elif isinstance(df, np.ndarray):
            pd.DataFrame(df).to_csv(os.path.join(output_dir, 'data', f'{name}.csv'))
def main():
    # Create output directory
    output_dir = create_output_directory()
    
    # Task 3.1 - Get and clean experience metrics
    print("Fetching and cleaning experience metrics...")
    experience_df = get_experience_metrics()
    experience_df = clean_experience_data(experience_df)
    
    # Task 3.2 - Analyze metric distributions
    print("Analyzing network metrics distributions...")
    raw_df = get_raw_metrics()
    metric_distributions = analyze_metric_distributions(raw_df)
    
    # Task 3.3 - Analyze throughput and TCP retransmission per handset
    print("Analyzing metrics per handset type...")
    plot_throughput_distribution(experience_df, output_dir)
    plot_tcp_retrans_distribution(experience_df, output_dir)
    
    # Task 3.4 - Perform experience clustering
    print("Performing experience clustering...")
    clustered_df, cluster_stats = perform_experience_clustering(experience_df)
    plot_experience_clusters(clustered_df, output_dir)
    
    # Generate cluster descriptions
    cluster_descriptions = describe_clusters(cluster_stats)
    
    # Export results
    results = {
        'experience_metrics': experience_df,
        'cluster_statistics': cluster_stats,
        'cluster_descriptions': pd.DataFrame.from_dict(cluster_descriptions, orient='index', 
                                                     columns=['description'])
    }
    
    for metric, distributions in metric_distributions.items():
        for dist_type, values in distributions.items():
            results[f'{metric}_{dist_type}'] = pd.Series(values)
    
    export_results(results, output_dir)
    
    print("\nCluster Descriptions:")
    for cluster, description in cluster_descriptions.items():
        print(f"\nCluster {cluster}: {description}")
    
    print(f"\nAnalysis complete! Results have been saved to: {output_dir}")
    print("- Plots can be found in the 'plots' subdirectory")
    print("- Data exports can be found in the 'data' subdirectory")

if __name__ == "__main__":
    main()