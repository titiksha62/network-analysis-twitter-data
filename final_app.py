import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
import plotly.express as px
import joblib
import os
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Set page configuration for a professional wide layout
st.set_page_config(page_title="Information Integrity Dashboard", layout="wide")

@st.cache_resource
def load_assets():
    df = pd.read_csv('pheme_enriched_v2.csv')
    # Load community detection results
    comm = pd.read_csv('community_data.csv')
    # Load the pre-trained classification model
    model = joblib.load('veracity_model.joblib')
    return df, comm, model

def main():
    st.title("Information Integrity Dashboard")
    st.subheader("Automated Misinformation Detection and Social Cluster Analysis")

    df, comm, model = load_assets()
    sia = SentimentIntensityAnalyzer()

    # Layout Tabs for Modular Insight Presentation
    tab1, tab2, tab3 = st.tabs(["Community Visualization", "Social Analytics", "Live Veracity Predictor"])

    with tab1:
        st.header("Community Detection: Echo Chambers")
        st.write("Nodes are colored by their detected community. Dense clusters of the same color represent isolated social groups often responsible for rapid rumor circulation.")
        
        # Merge community IDs and FIX the NaN error by filling missing values with 0
        df_viz = pd.merge(df, comm, on='user_id', how='left')
        df_viz['community_id'] = df_viz['community_id'].fillna(0).astype(int) 

        event = st.selectbox("Select Event", df['event'].unique())
        # Filter for visibility
        subset = df_viz[df_viz['event'] == event].head(200)

        net = Network(height="600px", width="100%", bgcolor="#ffffff", directed=False)
        
        # Add nodes with community-based coloring
        color_map = ["#FF5733", "#33FF57", "#3357FF", "#F333FF", "#FFF333", "#33FFF3"]
        for _, row in subset.iterrows():
            comm_id = int(row['community_id'])
            color = color_map[comm_id % len(color_map)]
            net.add_node(str(row['user_id']), title=f"User: {row['user_screen_name']}", color=color)

        # Add edges representing reply interactions
        for _, row in subset.dropna(subset=['parent_user_id']).iterrows():
            try:
                net.add_edge(str(row['user_id']), str(int(row['parent_user_id'])))
            except: 
                pass

        # Stabilization options for better physics-based movement
        net.toggle_physics(True)
        net.save_graph("comm_net.html")
        
        # MODERN 2026 UPDATE: Replace st.components.v1.html with st.iframe as per log instructions
        st.iframe("comm_net.html", height=650)

    with tab2:
        st.header("Social Sentiment Analysis")
        # MODERN 2026 UPDATE: Replace use_container_width=True with width='stretch'
        fig = px.box(df, x="veracity", y="sentiment_score", color="veracity", points="all")
        st.plotly_chart(fig, width='stretch')

    with tab3:
        st.header("Institutional Verification Tool")
        st.write("Input tweet details below to simulate the institutional automated verification process.")
        
        col1, col2 = st.columns(2)
        with col1:
            input_text = st.text_area("Tweet Content", "Breaking news regarding the latest policy change.")
            followers = st.number_input("User Follower Count", min_value=0, value=1000)
        with col2:
            bridge = st.slider("Calculated Bridge Score", 0.0, 1.0, 0.05)
            influence = st.slider("Calculated Influence Score", 0.0, 1.0, 0.02)

        if st.button("Analyze Veracity"):
            # Real-time sentiment analysis
            sentiment = sia.polarity_scores(input_text)['compound']
            features = pd.DataFrame([[followers, sentiment, bridge, influence]], 
                                    columns=['followers', 'sentiment_score', 'bridge_score', 'influence_score'])
            
            # Predict using the Random Forest model
            prediction = model.predict(features)[0]
            probability = model.predict_proba(features)[0][prediction]
            
            if prediction == 1:
                st.error(f"Alert: High probability of Rumour detected ({probability:.2%})")
            else:
                st.success(f"Verified: Content aligns with Official News patterns ({probability:.2%})")
            
            st.info(f"Analysis parameters: Sentiment: {sentiment:.2f} | Bridge: {bridge:.2f}")

if __name__ == "__main__":
    main()