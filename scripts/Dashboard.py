import streamlit as st
import pandas as pd
import numpy as np
import psycopg2
import mysql.connector
import plotly.express as px
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# PostgreSQL connection function
def connect_to_postgres():
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
        st.error(f"Error connecting to PostgreSQL: {e}")
        return None

# MySQL connection function
def connect_to_mysql():
    try:
        conn = mysql.connector.connect(
            host=os.getenv("MYSQL_HOST", "localhost"),
            user=os.getenv("MYSQL_USER", "root"),
            password=os.getenv("MYSQL_PASSWORD", ""),
            database=os.getenv("MYSQL_DATABASE", "telecom_analysis")
        )
        return conn
    except Exception as e:
        st.error(f"MySQL Connection Error: {str(e)}")
        st.info("Please check your .env file contains valid MySQL credentials:")
        st.code("""
MYSQL_HOST=your_host
MYSQL_USER=your_username
MYSQL_PASSWORD=your_password
MYSQL_DATABASE=telecom_analysis
        """)
        return None

def query_mysql(query):
    conn = connect_to_mysql()
    if conn is not None:
        try:
            df = pd.read_sql_query(query, conn)
            return df
        except Exception as e:
            st.error(f"Query Error: {str(e)}")
            return pd.DataFrame()
        finally:
            conn.close()
    return pd.DataFrame()

# Query data from database
def query_data(query, db_type="postgres"):
    conn = connect_to_postgres() if db_type == "postgres" else connect_to_mysql()
    if conn is not None:
        try:
            return pd.read_sql_query(query, conn)
        finally:
            conn.close()
    else:
        return pd.DataFrame()

# Set page configuration
st.set_page_config(
    page_title="Telecom Insights Dashboard",
    page_icon="📊",
    layout="wide",
)

# Sidebar navigation
try:
    st.sidebar.image("test\dashboard_logo.png", use_container_width=True)
except Exception as e:
    st.sidebar.error("Logo image not found. Please ensure 'dashboard_logo.png' is in the working directory.")

st.sidebar.title("📈 Telecom Insights")

page = st.sidebar.radio(
    "Navigate to:",
    ["User Overview", "User Engagement", "User Experience", "User Satisfaction"]
)

if page == "User Overview":
    st.title("📋 User Overview Analysis")
    
    st.markdown("### Top 10 Handsets")
    query = """
    SELECT "Handset Type", COUNT(*) as count
    FROM xdr_data
    WHERE "Handset Type" IS NOT NULL
    GROUP BY "Handset Type"
    ORDER BY count DESC
    LIMIT 10;
    """
    handsets_df = query_data(query)
    fig1 = px.bar(handsets_df, x="Handset Type", y="count", color="Handset Type",
                 title="Top 10 Handsets Used by Customers")
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("### Top 3 Manufacturers")
    query = """
    SELECT "Handset Manufacturer", COUNT(*) as count
    FROM xdr_data
    WHERE "Handset Manufacturer" IS NOT NULL
    GROUP BY "Handset Manufacturer"
    ORDER BY count DESC
    LIMIT 3;
    """
    manufacturers_df = query_data(query)
    fig2 = px.bar(manufacturers_df, x="Handset Manufacturer", y="count", color="Handset Manufacturer",
                 title="Top 3 Handset Manufacturers")
    st.plotly_chart(fig2, use_container_width=True)

elif page == "User Engagement":
    st.title("💻 User Engagement Analysis")

    st.markdown("### Top 10 Users by Metrics")
    metric = st.selectbox("Select Metric", ["session_frequency", "total_duration", "total_traffic"])
    query = f"""
    SELECT "MSISDN/Number" as user_id, {metric} as metric_value
    FROM (
        SELECT 
            "MSISDN/Number",
            COUNT(*) as session_frequency,
            SUM("Dur. (ms)") as total_duration,
            SUM("Total DL (Bytes)" + "Total UL (Bytes)") as total_traffic
        FROM xdr_data
        GROUP BY "MSISDN/Number"
    ) subquery
    ORDER BY metric_value DESC
    LIMIT 10;
    """
    engagement_df = query_data(query)
    fig3 = px.bar(engagement_df, x="user_id", y="metric_value", title=f"Top 10 Users by {metric}")
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("### Engagement Clusters")
    # Replace with your clustering visualization logic
    st.info("Cluster visualization to be added here.")

elif page == "User Experience":
    st.title("🎯 User Experience Analysis")

    st.markdown("### Throughput Distribution per Handset")
    query = """
    SELECT "Handset Type", AVG("Avg Bearer TP DL (kbps)" + "Avg Bearer TP UL (kbps)")/2 as avg_throughput
    FROM xdr_data
    GROUP BY "Handset Type"
    """
    throughput_df = query_data(query)
    fig4 = px.box(throughput_df, x="Handset Type", y="avg_throughput",
                  title="Throughput Distribution per Handset")
    st.plotly_chart(fig4, use_container_width=True)

    st.markdown("### TCP Retransmission per Handset")
    query = """
    SELECT "Handset Type", AVG("TCP DL Retrans. Vol (Bytes)" + "TCP UL Retrans. Vol (Bytes)") as avg_tcp_retrans
    FROM xdr_data
    GROUP BY "Handset Type"
    """
    tcp_df = query_data(query)
    fig5 = px.box(tcp_df, x="Handset Type", y="avg_tcp_retrans",
                  title="TCP Retransmission Distribution per Handset")
    st.plotly_chart(fig5, use_container_width=True)

elif page == "User Satisfaction":
    st.title("😊 User Satisfaction Analysis")
    
    # Create mock satisfaction data
    np.random.seed(42)
    n_users = 1000
    
    satisfaction_data = {
        'user_id': range(1, n_users + 1),
        'engagement_score': np.random.normal(70, 15, n_users).clip(0, 100),
        'experience_score': np.random.normal(75, 12, n_users).clip(0, 100),
    }
    
    df = pd.DataFrame(satisfaction_data)
    df['satisfaction_score'] = (df['engagement_score'] + df['experience_score']) / 2
    df['satisfaction_cluster'] = np.where(df['satisfaction_score'] > df['satisfaction_score'].mean(), 1, 0)
    
    # Round all scores to 2 decimal places
    score_columns = ['engagement_score', 'experience_score', 'satisfaction_score']
    df[score_columns] = df[score_columns].round(2)
    
    # Display top 10 satisfied users
    st.markdown("### Top 10 Satisfied Users")
    top_10 = df.nlargest(10, 'satisfaction_score')
    st.dataframe(top_10)
    
    # Create scatter plot
    st.markdown("### Satisfaction Clusters")
    fig = px.scatter(df, 
                    x="engagement_score",
                    y="experience_score",
                    color="satisfaction_cluster",
                    title="User Satisfaction Clusters",
                    color_continuous_scale="Viridis")
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display cluster statistics
    st.markdown("### Cluster Statistics")
    stats_df = df.groupby('satisfaction_cluster').agg({
        'engagement_score': 'mean',
        'experience_score': 'mean',
        'satisfaction_score': 'mean'
    }).round(2)
    st.dataframe(stats_df)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Built with ❤️ using Streamlit")
