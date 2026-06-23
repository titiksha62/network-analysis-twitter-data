import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def run_ml_pipeline():
    print("--- Phase 4: Machine Learning Detection Engine ---")
    
    # 1. Load Step 3 Enriched Data
    df = pd.read_csv('pheme_enriched_v2.csv')
    
    # 2. Load Step 2 Network Metrics
    # Ensure you have the 'network_metrics.csv' from your previous run
    try:
        metrics = pd.read_csv('network_metrics_v2.csv')
        # Convert IDs to string to ensure a perfect match during merge
        df['user_id'] = df['user_id'].astype(str)
        metrics['user_id'] = metrics['user_id'].astype(str)
        
        # Merge content features with network features
        final_df = pd.merge(df, metrics, on='user_id', how='left').fillna(0)
    except FileNotFoundError:
        print("Network metrics not found. Using content features only.")
        final_df = df.fillna(0)

    # 3. Define Features and Target
    # We use a mix of Social Media Analytics (followers), NLP (sentiment), 
    # and Network Science (bridge/influence scores)
    feature_cols = ['followers', 'sentiment_score', 'bridge_score', 'influence_score']
    
    # Target: 1 for rumours, 0 for verified-news
    # (Adjust 'rumours' to match the exact string in your 'thread_type' column)
    X = final_df[feature_cols]
    y = final_df['thread_type'].apply(lambda x: 1 if x == 'rumours' else 0)

    # 4. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. Model Training
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    
    # 6. Evaluation
    preds = model.predict(X_test)
    print("\n--- Model Evaluation ---")
    print(f"Overall Accuracy: {accuracy_score(y_test, preds):.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, preds))

    # 7. FEATURE IMPORTANCE (The 'Explainability' part for your project)
    importances = model.feature_importances_
    feat_importances = pd.Series(importances, index=feature_cols).sort_values(ascending=True)

    plt.figure(figsize=(10, 6))
    feat_importances.plot(kind='barh', color='darkslateblue')
    plt.title("What makes a tweet a Rumour? (Feature Importance)")
    plt.xlabel("Relative Importance Score")
    plt.show()

    # 8. Final Case Study Insights
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', cbar=False)
    plt.title("Confusion Matrix: Misinformation Detection")
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.show()

if __name__ == "__main__":
    run_ml_pipeline()