import streamlit as st
import pandas as pd
import networkx as nx
# pyrefly: ignore [missing-import]
from pyvis.network import Network
import streamlit.components.v1 as components
import os

st.set_page_config(page_title="Information Integrity Dashboard", layout="wide")

def main():
    st.title("Information Integrity Dashboard")
    st.subheader("Interactive News Propagation Mapping")
    
    st.markdown("""
    **Understanding the Network Representation:**
    * **Nodes:** Represent individual social media users involved in the conversation.
    * **Edges:** Represent a reply interaction. The arrow indicates the direction of information flow.
    * **Colors:** Large clusters indicate viral threads where misinformation or verified news is spreading rapidly.
    """)

    if not os.path.exists('pheme_enriched_v2.csv'):
        st.error("Enriched data not found. Please run previous analysis steps.")
        return

    df = pd.read_csv('pheme_enriched_v2.csv')

    st.sidebar.header("Network Controls")
    
    # Hairball Prevention: Degree Filter
    # Only show nodes with at least X interactions
    min_degree = st.sidebar.slider("Minimum Interaction Density", 1, 20, 5)
    
    event_filter = st.sidebar.selectbox("Focus on Specific Event", options=df['event'].unique())
    filtered_df = df[df['event'] == event_filter]

    # Process Graph
    G = nx.DiGraph()
    interactions = filtered_df.dropna(subset=['parent_user_id'])
    
    # Add all potential edges
    for _, row in interactions.iterrows():
        G.add_edge(str(row['user_id']), str(int(row['parent_user_id'])))

    # Apply the Hairball Filter
    # Remove nodes that have fewer than 'min_degree' connections
    remove = [node for node, degree in dict(G.degree()).items() if degree < min_degree]
    G.remove_nodes_from(remove)

    st.write(f"Displaying {G.number_of_nodes()} significant nodes. (Filtered out low-activity users to maintain clarity).")

    # Pyvis Configuration
    net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="black", directed=True)
    net.from_nx(G)

    # --- FIXING THE SLIPPERY ZOOM & INTERACTION ---
    # We use a JSON-like string to set specific Dampening and Interaction options
    options = """
    {
      "interaction": {
        "hover": true,
        "zoomView": true,
        "dragView": true,
        "navigationButtons": true,
        "keyboard": true
      },
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -50,
          "centralGravity": 0.01,
          "springLength": 100,
          "springConstant": 0.08
        },
        "maxVelocity": 50,
        "solver": "forceAtlas2Based",
        "timestep": 0.35,
        "stabilization": { "iterations": 150 }
      }
    }
    """
    net.set_options(options)

    # Generate and Render
    try:
        path = "network_viz.html"
        net.save_graph(path)
        with open(path, 'r', encoding='utf-8') as f:
            source_code = f.read()
        components.html(source_code, height=650)
    except Exception as e:
        st.error(f"Visualization error: {e}")

if __name__ == "__main__":
    main()