import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os

def run_network():
    print("--- Phase 2: Network Analysis & Bridge-Node Detection ---")
    
    input_file = 'pheme_master_v2.csv' 
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found!")
        return

    df = pd.read_csv(input_file)
    
    # 1. Build the Interaction Graph
    # We want a Directed Graph (DiGraph) to show the flow of info
    G = nx.DiGraph()

    # Create edges: User -> Parent User (Reply interaction)
    # We drop NaN because source tweets don't have a 'parent_user_id'
    interactions = df.dropna(subset=['parent_user_id', 'user_id'])
    
    # Ensure IDs are strings to avoid scientific notation issues
    for _, row in interactions.iterrows():
        u_id = str(row['user_id'])
        p_id = str(int(row['parent_user_id'])) # Convert to int then str to clean floats
        G.add_edge(u_id, p_id)

    print(f"Graph Statistics: {G.number_of_nodes()} Users, {G.number_of_edges()} Interactions")

    # 2. Bridge-Node Analysis (Betweenness Centrality)
    # This measures how often a node acts as a bridge between other nodes.
    print("Computing Betweenness Centrality (Identifying Bridge Nodes)...")
    betweenness = nx.betweenness_centrality(G)
    
    # 3. Influencer Analysis (Degree Centrality)
    # This measures total connections (Super-spreaders).
    print("Computing Degree Centrality (Identifying Influencers)...")
    degree = nx.degree_centrality(G)

    # 4. Save Results
    metrics = pd.DataFrame({
        'user_id': list(betweenness.keys()),
        'bridge_score': list(betweenness.values()),
        'influence_score': list(degree.values())
    }).sort_values(by='bridge_score', ascending=False)

    metrics.to_csv('network_metrics_v2.csv', index=False)
    print("Success! Network metrics saved to 'network_metrics_v2.csv'.")

    # 5. Visualization: The Ego-Network of the Top Bridge Node
    # A bridge node often connects two different "clusters" of the rumor.
    plt.figure(figsize=(12, 8))
    top_node = metrics.iloc[0]['user_id']
    
    # Get neighbors (the people this bridge connects)
    subgraph_nodes = [top_node] + list(G.neighbors(top_node))
    sub = G.subgraph(subgraph_nodes)
    
    pos = nx.spring_layout(sub)
    nx.draw(sub, pos, with_labels=False, node_color='teal', edge_color='gray', node_size=500, alpha=0.8)
    # Highlight the bridge node in red
    nx.draw_networkx_nodes(sub, pos, nodelist=[top_node], node_color='red', node_size=800)
    
    plt.title(f"SDG-16 Analysis: Micro-Network of Top Bridge Node (Red)")
    plt.show()

if __name__ == "__main__":
    run_network()