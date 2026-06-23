import pandas as pd
import networkx as nx
import community as community_louvain  # pip install python-louvain
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

def prepare_final_assets():
    print("--- Phase 6: Community Detection & Model Saving ---")
    
    # 1. Load Data
    df = pd.read_csv('pheme_enriched_v2.csv')
    
    # 2. Community Detection
    G = nx.Graph() # Louvain works best on undirected graphs for community structure
    interactions = df.dropna(subset=['parent_user_id', 'user_id'])
    for _, row in interactions.iterrows():
        G.add_edge(str(row['user_id']), str(int(row['parent_user_id'])))
    
    print("Detecting Communities (Echo Chambers)...")
    partition = community_louvain.best_partition(G)
    
    # Map communities back to a dataframe
    comm_df = pd.DataFrame.from_dict(partition, orient='index').reset_index()
    comm_df.columns = ['user_id', 'community_id']
    comm_df.to_csv('community_data.csv', index=False)
    
    # 3. Train and Save the Prediction Model
    print("Training final model for deployment...")
    metrics = pd.read_csv('network_metrics_v2.csv')
    df['user_id'] = df['user_id'].astype(str)
    metrics['user_id'] = metrics['user_id'].astype(str)
    
    final_df = pd.merge(df, metrics, on='user_id', how='left').fillna(0)
    
    features = ['followers', 'sentiment_score', 'bridge_score', 'influence_score']
    X = final_df[features]
    y = final_df['thread_type'].apply(lambda x: 1 if x == 'rumours' else 0)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X, y)
    
    # Save the model and the feature list
    joblib.dump(model, 'veracity_model.joblib')
    print("Model saved as 'veracity_model.joblib'.")

if __name__ == "__main__":
    prepare_final_assets()