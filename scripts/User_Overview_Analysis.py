import psycopg2
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate
from dotenv import load_dotenv
import os

# Load environment variables from .env file
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

def get_top_handsets():
    """Get top 10 handsets used by customers"""
    conn = connect_to_database()
    if conn is None:
        return None
    
    try:
        query = """
        SELECT "Handset Type", COUNT(*) as count
        FROM xdr_data
        WHERE "Handset Type" IS NOT NULL
        GROUP BY "Handset Type"
        ORDER BY count DESC
        LIMIT 10
        """
        
        df = pd.read_sql_query(query, conn)
        return df
    finally:
        conn.close()

def get_top_manufacturers():
    """Get top 3 handset manufacturers"""
    conn = connect_to_database()
    if conn is None:
        return None
    
    try:
        query = """
        SELECT "Handset Manufacturer", COUNT(*) as count
        FROM xdr_data
        WHERE "Handset Manufacturer" IS NOT NULL
        GROUP BY "Handset Manufacturer"
        ORDER BY count DESC
        LIMIT 3
        """
        
        df = pd.read_sql_query(query, conn)
        return df
    finally:
        conn.close()

def get_top_handsets_per_manufacturer(manufacturer):
    """Get top 5 handsets for a specific manufacturer"""
    conn = connect_to_database()
    if conn is None:
        return None
    
    try:
        query = """
        SELECT "Handset Type", COUNT(*) as count
        FROM xdr_data
        WHERE "Handset Manufacturer" = %s
        GROUP BY "Handset Type"
        ORDER BY count DESC
        LIMIT 5
        """
        
        df = pd.read_sql_query(query, conn, params=(manufacturer,))
        return df
    finally:
        conn.close()

def get_user_behavior():
    """Aggregate user behavior data"""
    conn = connect_to_database()
    if conn is None:
        return None
    
    try:
        query = """
        SELECT 
            "MSISDN/Number" as user_id,
            COUNT(*) as session_count,
            SUM("Dur. (ms)") as total_duration,
            SUM("Total DL (Bytes)") as total_dl,
            SUM("Total UL (Bytes)") as total_ul,
            SUM("Social Media DL (Bytes)" + "Social Media UL (Bytes)") as social_media_total,
            SUM("Google DL (Bytes)" + "Google UL (Bytes)") as google_total,
            SUM("Email DL (Bytes)" + "Email UL (Bytes)") as email_total,
            SUM("Youtube DL (Bytes)" + "Youtube UL (Bytes)") as youtube_total,
            SUM("Netflix DL (Bytes)" + "Netflix UL (Bytes)") as netflix_total,
            SUM("Gaming DL (Bytes)" + "Gaming UL (Bytes)") as gaming_total,
            SUM("Other DL (Bytes)" + "Other UL (Bytes)") as other_total
        FROM xdr_data
        GROUP BY "MSISDN/Number"
        """
        
        df = pd.read_sql_query(query, conn)
        return df
    finally:
        conn.close()

def perform_eda(df):
    """Perform exploratory data analysis"""
    # Handle missing values
    df = df.fillna(df.mean())
    
    # Create decile classes based on total duration
    df['duration_decile'] = pd.qcut(df['total_duration'], q=5, labels=['D1', 'D2', 'D3', 'D4', 'D5'])
    
    # Basic metrics
    basic_metrics = df.describe()
    
    # Correlation analysis for applications
    app_columns = ['social_media_total', 'google_total', 'email_total', 'youtube_total', 
                  'netflix_total', 'gaming_total', 'other_total']
    correlation_matrix = df[app_columns].corr()
    
    # PCA
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[app_columns])
    pca = PCA()
    pca_result = pca.fit_transform(scaled_data)
    
    return {
        'basic_metrics': basic_metrics,
        'correlation_matrix': correlation_matrix,
        'explained_variance_ratio': pca.explained_variance_ratio_
    }


def plot_visualizations(top_handsets, top_manufacturers, user_behavior, pca_result):
    """Generate visualizations for EDA and other analyses"""
    
    # Define the directory to save plots
    plot_dir = "telecom_analysis_20241222_055426/plots"
    os.makedirs(plot_dir, exist_ok=True)
    
    # Plot Top 10 Handsets
    plt.figure(figsize=(10, 6))
    sns.barplot(x="count", y="Handset Type", data=top_handsets, palette="viridis")
    plt.title("Top 10 Handsets Used by Customers")
    plt.xlabel("Count")
    plt.ylabel("Handset Type")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "top_10_handsets.png"))
    plt.show()
    plt.close()


    # Plot Top 3 Manufacturers
    plt.figure(figsize=(8, 5))
    sns.barplot(x="count", y="Handset Manufacturer", data=top_manufacturers, palette="Blues_d")
    plt.title("Top 3 Handset Manufacturers")
    plt.xlabel("Count")
    plt.ylabel("Manufacturer")
    plt.tight_layout()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "top_3_manufacturers.png"))
    plt.show()
    plt.close()

    # Correlation Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(user_behavior.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Matrix of User Behavior Data")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "correlation_matrix.png"))
    plt.show()
    plt.close()

    # Plot PCA Explained Variance
    plt.figure(figsize=(8, 5))
    plt.plot(np.cumsum(pca_result), marker='o', linestyle='--', color='b')
    plt.title("PCA Explained Variance")
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "pca_explained_variance.png"))
    plt.show()
    plt.close()

def main():
    # Task 1 - User Overview Analysis
    print("Top 10 Handsets:")
    top_handsets = get_top_handsets()
    print(tabulate(top_handsets, headers="keys", tablefmt="pretty"))
    
    print("\nTop 3 Manufacturers:")
    top_manufacturers = get_top_manufacturers()
    print(tabulate(top_manufacturers, headers="keys", tablefmt="pretty"))
    
    print("\nTop 5 Handsets per Manufacturer:")
    for manufacturer in top_manufacturers['Handset Manufacturer']:
        print(f"\n{manufacturer}:")
        top_handsets_per_manufacturer = get_top_handsets_per_manufacturer(manufacturer)
        print(tabulate(top_handsets_per_manufacturer, headers="keys", tablefmt="pretty"))
    
    # Task 1.1 - User Behavior Analysis
    print("\nUser Behavior Analysis:")
    user_behavior = get_user_behavior()
    
    # Task 1.2 - EDA
    print("\nExploratory Data Analysis:")
    eda_results = perform_eda(user_behavior)
    print("\nBasic Metrics:")
    print(tabulate(eda_results['basic_metrics'].transpose(), headers="keys", tablefmt="pretty"))
    print("\nCorrelation Matrix:")
    print(tabulate(eda_results['correlation_matrix'].round(2), headers="keys", tablefmt="pretty"))
    print("\nPCA Explained Variance Ratio:")
    print(eda_results['explained_variance_ratio'])
    
    # Generate visualizations
    plot_visualizations(top_handsets, top_manufacturers, user_behavior, eda_results['explained_variance_ratio'])

if __name__ == "__main__":
    main()
