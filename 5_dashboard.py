import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
import plotly.express as px
import os

# Set page configuration
st.set_page_config(page_title="SDG-16 Information Integrity Dashboard", layout="wide")

def main():
    st.title("SDG-16: Peace, Justice and Strong Institutions")
    st.subheader("Misinformation Propagation and Bridge-Node Analysis")
    
    st.markdown("""
    This application consolidates the end-to-end pipeline of the research project, 
    transforming complex network data into an interactive visual format. It is designed 
    to help institutional observers identify critical information bottlenecks and 
    understand the emotional sentiment behind news propagation.
    """)

    # Check for necessary files
    if not os.path.exists('pheme_enriched_v2.csv') or not os.path.exists('network_metrics_v2.csv'):
        st.error("Missing data files. Please run steps 1 through 4 first.")
        return

    # Load Data
    df = pd.read_csv('pheme_enriched_v2.csv')
    metrics = pd.read_csv('network_metrics_v2.csv')

    # Sidebar Filters
    st.sidebar.header("Analysis Filters")
    event_filter = st.sidebar.multiselect("Select Events", options=df['event'].unique(), default=df['event'].unique())
    filtered_df = df[df['event'].isin(event_filter)]

    # Layout Tabs
    tab1, tab2, tab3 = st.tabs(["Interactive Network", "Content Analytics", "Model Performance"])

    with tab1:
        st.header("Dynamic Propagation Network")
        st.write("Nodes are users, and edges represent replies. You can drag nodes to see the physics-based movement.")
        
        # Create NetworkX graph
        G = nx.DiGraph()
        # Sampling for performance (first 100 interactions in filtered set)
        sample_df = filtered_df.dropna(subset=['parent_user_id']).head(100)
        
        for _, row in sample_df.iterrows():
            G.add_edge(str(row['user_id']), str(int(row['parent_user_id'])))

        # Convert to Pyvis
        net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="black", directed=True)
        net.from_nx(G)
        
        # Enable physics for 'moving' effect
        net.toggle_physics(True)
        
        # Save and render
        try:
            path = "tmp_network.html"
            net.save_graph(path)
            HtmlFile = open(path, 'r', encoding='utf-8')
            source_code = HtmlFile.read()
            components.html(source_code, height=650)
        except:
            st.warning("Network rendering encountered an issue. Check local permissions.")

    with tab2:
        st.header("Social Media Analytics")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Sentiment Distribution")
            fig_sent = px.violin(filtered_df, x="veracity", y="sentiment_score", color="veracity", box=True)
            st.plotly_chart(fig_sent, use_container_width=True)
            
        with col2:
            st.write("Authority Analysis")
            fig_foll = px.bar(filtered_df.groupby('veracity')['followers'].mean().reset_index(), 
                              x='veracity', y='followers', color='veracity')
            st.plotly_chart(fig_foll, use_container_width=True)

    with tab3:
        st.header("Classification Metrics")
        st.write("Top Indicators of Misinformation (Feature Importance)")
        
        # Static representation of your Step 4 results
        # In a production app, you would load the importance values from a saved CSV
        feat_data = pd.DataFrame({
            'Feature': ['Bridge Score', 'Influence Score', 'Sentiment', 'Followers'],
            'Importance': [0.45, 0.25, 0.20, 0.10]
        }).sort_values('Importance', ascending=False)
        
        fig_imp = px.bar(feat_data, x='Importance', y='Feature', orientation='h', color='Importance')
        st.plotly_chart(fig_imp, use_container_width=True)

if __name__ == "__main__":
    main()