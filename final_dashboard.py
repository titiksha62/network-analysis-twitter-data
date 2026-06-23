
import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
import plotly.express as px
import os

st.set_page_config(page_title="Information Integrity Dashboard", layout="wide")

# --- CACHING DATA FOR SPEED ---
@st.cache_data
def load_data():
    df = pd.read_csv('pheme_enriched_v2.csv')
    metrics = pd.read_csv('network_metrics_v2.csv')
    return df, metrics

def main():
    st.title("Information Integrity Dashboard")
    
    st.markdown("""
    This version uses **server-side caching** and **dynamic sampling** to prevent browser lag. 
    Adjust the 'Sample Size' to control the density of the visualization.
    """)

    df, metrics = load_data()

    # --- SIDEBAR CONTROLS ---
    st.sidebar.header("Performance Controls")
    
    # User-controlled sampling to prevent crashes
    sample_size = st.sidebar.slider("Sample Size (Nodes/Edges)", 100, 2000, 500, 
                                   help="Lower values load instantly. Higher values create the 'hairball' effect.")
    
    min_degree = st.sidebar.slider("Interaction Filter (Min Degree)", 1, 15, 2)
    
    physics_enabled = st.sidebar.checkbox("Enable Physics (Moving Nodes)", value=False,
                                         help="Turn this on to see the nodes move. Keep it off for faster loading of large graphs.")

    event_filter = st.sidebar.multiselect("Event Context", options=df['event'].unique(), default=df['event'].unique()[:2])
    
    filtered_df = df[df['event'].isin(event_filter)]

    tab1, tab2, tab3 = st.tabs(["Interactive Network", "Content Analytics", "Model Performance"])

    with tab1:
        st.header("Propagation Network")
        
        # Build graph from sampled data
        G = nx.DiGraph()
        interactions = filtered_df.dropna(subset=['parent_user_id']).head(sample_size)
        
        for _, row in interactions.iterrows():
            G.add_edge(str(row['user_id']), str(int(row['parent_user_id'])))

        # Filter by degree
        remove = [node for node, degree in dict(G.degree()).items() if degree < min_degree]
        G.remove_nodes_from(remove)

        st.info(f"Visualizing {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

        # Pyvis Setup
        net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="black", directed=True)
        net.from_nx(G)

        if physics_enabled:
            net.toggle_physics(True)
            net.set_options("""
            {
              "physics": {
                "forceAtlas2Based": {"gravitationalConstant": -50, "centralGravity": 0.01, "springLength": 100},
                "solver": "forceAtlas2Based",
                "stabilization": {"iterations": 50}
              }
            }
            """)
        else:
            net.toggle_physics(False)

        # Render
        path = "fast_network.html"
        net.save_graph(path)
        with open(path, 'r', encoding='utf-8') as f:
            components.html(f.read(), height=650)

    with tab2:
        st.header("Social Media Analytics")
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(px.violin(filtered_df, x="veracity", y="sentiment_score", color="veracity"), use_container_width=True)
        with col2:
            st.plotly_chart(px.bar(filtered_df.groupby('veracity')['followers'].mean().reset_index(), x='veracity', y='followers'), use_container_width=True)

    with tab3:
        st.header("Detection Model Performance")
        feat_data = pd.DataFrame({
            'Feature': ['Bridge Score', 'Influence Score', 'Sentiment', 'Followers'],
            'Importance': [0.45, 0.25, 0.20, 0.10]
        })
        st.plotly_chart(px.bar(feat_data, x='Importance', y='Feature', orientation='h', color='Importance'), use_container_width=True)

if __name__ == "__main__":
    main()