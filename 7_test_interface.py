import streamlit as st
import pandas as pd
import joblib
# pyrefly: ignore [missing-import]
import nltk
# pyrefly: ignore [missing-import]
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download vader lexicon if not already downloaded
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

# Set simple page configuration
st.set_page_config(page_title="Simple Tweet Verifier", layout="centered")

@st.cache_resource
def load_assets_v2():
    model = joblib.load('veracity_model.joblib')
    
    # Load dataset for realistic examples
    try:
        df = pd.read_csv('pheme_enriched_v2.csv')
        metrics = pd.read_csv('network_metrics_v2.csv')
        df['user_id'] = df['user_id'].astype(str)
        metrics['user_id'] = metrics['user_id'].astype(str)
        final_df = pd.merge(df, metrics, on='user_id', how='left').fillna(0)
    except:
        final_df = None
        
    return model, final_df

def main():
    st.title("Tweet Veracity Checker")
    st.markdown("A simple tool to analyze a tweet and detect whether it's **Verified News** or **Fake/Rumour**.")

    model, df = load_assets_v2()
    sia = SentimentIntensityAnalyzer()

    if df is not None:
        if st.button("🎲 Load Real Example from Dataset"):
            sample = df.sample(1).iloc[0]
            st.session_state['text'] = str(sample['text'])
            st.session_state['followers'] = int(sample['followers'])
            st.session_state['bridge'] = float(sample['bridge_score'])
            st.session_state['influence'] = float(sample['influence_score'])
            st.session_state['actual'] = str(sample['thread_type'])

    # Initialize session state if empty
    if 'text' not in st.session_state:
        st.session_state['text'] = "Breaking news! Something major just happened..."
        st.session_state['followers'] = 1500
        st.session_state['bridge'] = 0.05
        st.session_state['influence'] = 0.02
        st.session_state['actual'] = None

    # --- INPUT SECTION ---
    st.subheader("1. Input Tweet")
    tweet_text = st.text_area("Paste the tweet content here:", value=st.session_state['text'], height=120)
    
    st.subheader("2. Tweet Metadata")
    with st.expander("⚙️ View / Edit Network Metadata", expanded=False):
        st.markdown("*(These metrics are crucial for the AI, but you can leave them as defaults or use the 'Load Real Example' button above.)*")
        col1, col2, col3 = st.columns(3)
        with col1:
            followers = st.number_input("Follower Count", min_value=0, value=st.session_state['followers'], step=100)
        with col2:
            bridge = st.slider("Network Bridge Score", 0.0, 1.0, float(st.session_state['bridge']))
        with col3:
            influence = st.slider("Network Influence", 0.0, 1.0, float(st.session_state['influence']))

    # --- ANALYSIS ACTION ---
    if st.button("Check Veracity", type="primary", use_container_width=True):
        if not tweet_text.strip():
            st.warning("Please enter a tweet to analyze.")
            return
            
        with st.spinner("Analyzing tweet..."):
            # 1. Compute NLP Sentiment automatically
            sentiment_score = sia.polarity_scores(tweet_text)['compound']
            
            # 2. Build feature vector required by the RandomForest Model
            features = pd.DataFrame(
                [[followers, sentiment_score, bridge, influence]], 
                columns=['followers', 'sentiment_score', 'bridge_score', 'influence_score']
            )
            
            # 3. Predict Veracity
            pred_class = model.predict(features)[0]
            pred_proba = model.predict_proba(features)[0]
            confidence = pred_proba[pred_class]
            
            # 4. Show Output
            st.divider()
            
            if st.session_state['actual'] is not None:
                if st.session_state['actual'] == 'rumours':
                    st.write(f"**Ground Truth from Dataset:** RUMOUR")
                else:
                    st.write(f"**Ground Truth from Dataset:** VERIFIED NEWS")

            if pred_class == 1:
                st.error("**RESULT: FAKE NEWS / RUMOUR**")
                st.write("This content matches the patterns typically seen in unverified rumours.")
            else:
                st.success("**RESULT: VERIFIED NEWS**")
                st.write("This content matches the patterns typically seen in verified institutional news.")
                
            # Highlight Confidence Score
            st.metric(label="Model Confidence Score", value=f"{confidence:.2%}")
            st.caption(f"Internal calculated sentiment score: {sentiment_score:.2f}")

if __name__ == "__main__":
    main()
