import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# pyrefly: ignore [missing-import]
import nltk 
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer # type: ignore


# Initialize Sentiment Tool
nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()

def run_content_analysis():
    print("--- Phase 3: Content & Engagement Analytics ---")
    
    # 1. Load the dataset from Step 1
    df = pd.read_csv('pheme_master_v2.csv')
    df['text'] = df['text'].astype(str).fillna('')

    # 2. Sentiment Analysis (Emotional Tone)
    print("Analyzing emotional tone of tweets...")
    df['sentiment_score'] = df['text'].apply(lambda x: sia.polarity_scores(x)['compound'])
    
    # Label sentiment for easier reading
    df['sentiment_type'] = df['sentiment_score'].apply(
        lambda s: 'Positive' if s > 0.05 else ('Negative' if s < -0.05 else 'Neutral')
    )

    # 3. Engagement Analytics: Followers vs. Veracity
    # This addresses the "Social Media Analytics" requirement
    print("Analyzing account authority vs. news type...")
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='veracity', y='followers', hue='veracity', palette='magma', legend=False)
    plt.title("Institutional Authority: Average Followers by News Category")
    plt.ylabel("Avg. Follower Count")
    plt.show()

    # 4. TF-IDF Analysis (Keyword Extraction)
    # Let's see what keywords define 'false' rumours specifically
    print("\n--- TF-IDF Keyword Extraction ---")
    
    # Grouping text by veracity
    # Note: If your data only has 'rumours' vs 'non-rumours', adjust categories accordingly
    categories = df['veracity'].unique()
    
    for cat in categories:
        cat_text = df[df['veracity'] == cat]['text']
        if len(cat_text) > 10:
            # We filter for english stop words and look for top 10 unique terms
            vectorizer = TfidfVectorizer(stop_words='english', max_features=10)
            vectorizer.fit_transform(cat_text)
            terms = vectorizer.get_feature_names_out()
            print(f"Top terms for [{cat}]: {', '.join(terms)}")

    # 5. Sentiment Distribution Visualization
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df, x='veracity', y='sentiment_score', hue='veracity', palette='coolwarm', split=False)
    plt.axhline(0, color='black', linestyle='--', alpha=0.5)
    plt.title("Sentiment Volatility: How Emotional is the Propagation?")
    plt.show()

    # Save enriched data for the Machine Learning step
    df.to_csv('pheme_enriched_v2.csv', index=False)
    print("\nEnriched data saved to 'pheme_enriched_v2.csv'. Ready for Phase 4.")

if __name__ == "__main__":
    run_content_analysis()