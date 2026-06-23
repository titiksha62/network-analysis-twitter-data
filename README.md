# Information Integrity & Misinformation Detection Pipeline

This project provides an end-to-end pipeline for analyzing and detecting misinformation (rumours) in social media structures. 

It uses a combination of Natural Language Processing (NLP), Social Network Analysis (SNA), and Machine Learning to process conversational threads (e.g., the PHEME dataset) and classify them based on veracity. The pipeline also includes an interactive Streamlit dashboard to visually explore network propagation, user authority, sentiment, and model performance.

## Visualizing the Network & Architecture

*(To display the images you just shared, create an `images` folder in your project, place your screenshots inside, and update the filenames below)*

<p align="center">
  <img src="images/flowchart.png" alt="Architecture Flowchart" width="800"/>
</p>
<p align="center">
  <img src="images/network_clusters.png" alt="Network Clusters" width="400"/>
  <img src="images/dashboard_filters.png" alt="Dashboard Filters" width="400"/>
</p>

## Project Structure and Pipeline

1. **Extraction (`0_extract.py`)**: 
   Extracts the compressed dataset (`PHEME_veracity.tar.bz2`) into the local `pheme_data/` directory.

2. **Data Flattening (`1_data_flattening_correct.py`)**: 
   Parses the raw JSON files of social media threads. It extracts tweet metadata, user properties (followers, verification status), and interactions (who replied to whom) and consolidates this into `pheme_master_v2.csv`.

3. **Network Analysis (`2_network_analysis.py`)**: 
   Builds a directed interaction graph using `NetworkX`. It computes key metrics for each user:
   - *Betweenness Centrality*: Identifies "Bridge Nodes" connecting different clusters.
   - *Degree Centrality*: Identifies "Influencers" or super-spreaders.
   Saves results to `network_metrics_v2.csv`.

4. **Content Analytics (`3_content_analytics.py`)**: 
   Applies NLP (VADER Sentiment Analysis and TF-IDF) to understand the emotional tone and key terms of the conversation. Saves the combined dataset as `pheme_enriched_v2.csv`.

5. **Misinformation Classifier (`4_misinfo_classifier.py`)**: 
   Trains a Random Forest classifier using merged network metrics and content features to detect 'rumours' vs 'verified-news'. It evaluates accuracy, classification reports, and feature importance.

6. **Final Preparation & Community Detection (`6_prep_final.py`)**: 
   Runs the Louvain community detection algorithm to identify "echo chambers" (`community_data.csv`) and saves the final trained Random Forest model as `veracity_model.joblib` for deployment.

7. **Interactive Dashboard (`final_dashboard.py`)**: 
   An optimized Streamlit application that provides:
   - Interactive Network Propagation mapping (using Pyvis).
   - Social Media Analytics visualizations (using Plotly).
   - Detection Model Performance metrics.

## The Analytical Narrative: Fake News vs. Verified News

When presenting this project, a core focus is explaining *how* our pipeline distinguishes between fake news (rumours) and verified news. Here is how that narrative maps directly to the project flow:

* **In Step 1 (Flattening): The Ground Truth**
  We establish the baseline. By parsing the `veracity-annotations.json` from the PHEME dataset, we explicitly tag threads as `rumours` (Fake/Unverified) or `verified-news`. This gives us the target label we need to analyze their differences.
* **In Step 2 (Network Analysis): Structural Signatures**
  * **Verified News** typically follows a "broadcast" or star-shaped network topology. It spreads outward from authoritative, highly-connected hubs (identifiable by high **Degree Centrality**).
  * **Fake News** often struggles to gain traction from official hubs. Instead, it relies heavily on "super-spreaders" or **Bridge Nodes** (identifiable by high **Betweenness Centrality**) to forcefully push the narrative across disconnected echo chambers.
* **In Step 3 (Content Analytics): Emotional Signatures**
  * **Verified News** language tends to be emotionally neutral and objective. The spreaders generally have higher institutional authority (high follower counts).
  * **Fake News** is designed to trigger engagement. Our VADER sentiment analysis usually shows that fake news has much higher emotional volatility (extremely positive or negative sentiment scores). The spreaders often have lower baseline authority.
* **In Steps 4 & 6 (Machine Learning): The AI's Logic**
  We don't just guess; we prove it. The Random Forest classifier explicitly learns these patterns. When you look at the **Feature Importance** chart in the dashboard, the AI reveals that a combination of [High Bridge Score + High Sentiment Volatility + Low Follower Count] is the mathematical fingerprint of Fake News.

## Setup and Installation

1. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install required dependencies:
   The project includes two dependency files. You can use the standard one, or the "fresh" one which contains updated package versions.
   ```bash
   # Install the standard environment
   pip install -r requirements.txt
   
   # OR install the updated "fresh" environment
   pip install -r requirements_fresh.txt
   ```

3. **Download the Dataset:**
   You can download the dataset from [Figshare: PHEME dataset for Rumour Detection](https://figshare.com/articles/dataset/PHEME_dataset_for_Rumour_Detection_and_Veracity_Classification/6392078).
   Ensure the downloaded archive (`PHEME_veracity.tar.bz2` or `6392078.zip`) is placed in the project root directory before running the scripts.

## Demo Execution Steps

## Demo Execution Steps & Presentation Script

During your demo, you can run these scripts sequentially. Here is exactly what is happening under the hood in each step, including the specific models, libraries, and methodologies used:

### 1. Data Preparation (Extraction & Flattening)
```bash
python 0_extract.py
python 1_data_flattening_correct.py
```
* **What we did:** We programmatically ingested raw social media data (the PHEME dataset) from a compressed `.tar.bz2` archive. The flattening script recursively traverses the nested JSON directory structure, reads the thread topologies (source tweets vs. reactions), and maps complex JSON trees into a tabular pandas DataFrame.
* **Exactly how:** Python's native `tarfile` and `json` libraries were used. We explicitly extracted user metadata (`followers_count`, `verified` status) and interaction topology (`in_reply_to_user_id_str`).
* **Output:** A clean tabular dataset `pheme_master_v2.csv` mapping every reply and user interaction.

### 2. Feature Engineering (Network & NLP Analytics)
```bash
python 2_network_analysis.py
python 3_content_analytics.py
```
* **What we did (Network):** We constructed a mathematical representation of how the rumour propagated.
* **Exactly how & Which Models:** We used the **NetworkX** library to build a Directed Graph (`nx.DiGraph`). For every reply, a directed edge was created from the replying user to the parent user. We then executed two specific algorithms:
   - **Betweenness Centrality:** Calculated to find "Bridge Nodes" (users bridging disparate echo chambers).
   - **Degree Centrality:** Calculated to find "Influencers" (users with the highest absolute volume of direct connections).
* **What we did (NLP Content):** We analyzed the text of the tweets to extract semantic features.
* **Exactly how & Which Models:** 
   - **Sentiment:** We used **NLTK's VADER** (`SentimentIntensityAnalyzer`), a lexicon-based model specifically tuned for social media. We extracted the 'compound' polarity score to measure emotional volatility.
   - **Keywords:** We used **Scikit-learn's `TfidfVectorizer`** (Term Frequency-Inverse Document Frequency) with English stop-words removed to extract the top 10 defining terms of the rumour vs. non-rumour threads.

### 3. Machine Learning Classification
```bash
python 4_misinfo_classifier.py
```
* **What we did:** We trained an algorithmic classifier to predict whether a given thread is a 'rumour' or 'verified-news' based purely on the metadata, network metrics, and sentiment.
* **Which Model:** **Scikit-learn's Random Forest Classifier** (`RandomForestClassifier`).
* **How we trained & Exactly how:** 
   - **Features:** 4 explicitly engineered features: `['followers', 'sentiment_score', 'bridge_score', 'influence_score']`.
   - **Data Split:** We used an **80/20 train-test split** (`test_size=0.2`) on the dataset.
   - **Hyperparameters:** The Random Forest was instantiated with **100 decision trees** (`n_estimators=100`).
   - **Duration:** Because we engineered the data into a clean, low-dimensional tabular format, this model trains instantly on the CPU (typically taking **less than 1-2 seconds** to fit and evaluate).
   - **Evaluation:** The script outputs Overall Accuracy, a full Classification Report (Precision, Recall, F1-score), and extracts the `feature_importances_` to explain *how* the model is making decisions.

### 4. Finalizing Assets & Echo Chamber Detection
```bash
python 6_prep_final.py
```
* **What we did:** We mapped the final network clusters and saved our production model.
* **Exactly how & Which Models:** 
   - **Community Detection:** We used the **Louvain heuristic algorithm** (`community_louvain.best_partition()`) on an undirected version of our graph to explicitly detect clustered sub-networks (Echo Chambers).
   - **Deployment Preparation:** We retrained our Random Forest model (`n_estimators=100`) on the *entire* 100% dataset (no test split) to maximize its knowledge base, and serialized it using **Joblib** (`joblib.dump()`) into a production-ready file (`veracity_model.joblib`).

### 5. Interactive Dashboard (Visual Demo)
```bash
streamlit run final_dashboard.py
```
* **What we did:** We deployed a local web server to dynamically visualize our findings and the model's logic.
* **Exactly how:** We utilized **Streamlit** for the frontend architecture. For the interactive network mapping, we integrated **Pyvis**, running a `forceAtlas2Based` physics solver to visually organize the network graph. Data visualizations were generated using **Plotly Express** for dynamic, hoverable charts.
* **What to show the audience:** Open the local URL (`http://localhost:8501`). Demonstrate the interactive physics graph (mentioning the Pyvis ForceAtlas2 algorithm), the sentiment distribution violin plots, and the Model Performance tab showcasing the Random Forest's feature importance.
