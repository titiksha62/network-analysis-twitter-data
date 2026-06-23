# Information Integrity & Misinformation Detection Pipeline

This project is an end-to-end pipeline designed to analyze and detect misinformation (rumours) in social media structures. It combines Natural Language Processing (NLP), Social Network Analysis (SNA), and Machine Learning to process conversational threads and classify them based on their veracity.

## Architecture & Visuals

<p align="center">
  <img src="images/Architecture%20Flowchart.png" alt="Architecture Flowchart" width="800"/>
</p>
<p align="center">
  <img src="images/Network%20Clusters.png" alt="Network Clusters" width="400"/>
  <img src="images/Dashboard%20Filters.png" alt="Dashboard Filters" width="400"/>
</p>

## End-to-End Pipeline

The project is structured into sequential steps, moving from raw data extraction to interactive visualization:

1. **Data Extraction & Flattening (`0_extract.py` & `1_data_flattening_correct.py`)**
   I start by ingesting raw JSON data from the PHEME dataset. The flattening script maps complex conversational threads (who replied to whom) and user metadata into a structured tabular format (`pheme_master_v2.csv`).

2. **Network Analysis (`2_network_analysis.py`)**
   Using `NetworkX`, I build a directed graph to map how information propagates. Key metrics computed include:
   *   **Betweenness Centrality:** Identifies "Bridge Nodes" that connect different clusters (often used by rumours to cross echo chambers).
   *   **Degree Centrality:** Identifies "Influencers" with high direct connections.

3. **Content Analytics (`3_content_analytics.py`)**
   I apply NLP techniques to analyze the text. `NLTK's VADER` extracts the emotional sentiment (volatility) of the tweets, while `TF-IDF` pulls out the key terms defining the conversation.

4. **Machine Learning Classification (`4_misinfo_classifier.py` & `6_prep_final.py`)**
   A Random Forest classifier is trained on the network and sentiment metrics to predict whether a thread is a 'rumour' or 'verified-news'. The algorithm learns that Fake News often relies on high bridge-node activity, emotional volatility, and lower user authority. The final model is serialized (`veracity_model.joblib`) for deployment.

5. **Interactive Dashboard (`final_dashboard.py`)**
   The pipeline culminates in a Streamlit web application. It features an interactive, physics-based network graph (via `Pyvis`) and interactive data visualizations (via `Plotly`) to explore the propagation of misinformation and model performance in real-time.

## Setup and Installation

1. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install required dependencies:
   ```bash
   # Install the standard environment
   pip install -r requirements.txt
   
   # OR install the updated "fresh" environment
   pip install -r requirements_fresh.txt
   ```

3. **Download the Dataset:**
   You can download the dataset from [Figshare: PHEME dataset for Rumour Detection](https://figshare.com/articles/dataset/PHEME_dataset_for_Rumour_Detection_and_Veracity_Classification/6392078).
   Ensure the downloaded archive (`PHEME_veracity.tar.bz2` or `6392078.zip`) is placed in the project root directory before running the scripts.

## Running the Project

To execute the pipeline end-to-end, run the scripts in numerical order:

```bash
python 0_extract.py
python 1_data_flattening_correct.py
python 2_network_analysis.py
python 3_content_analytics.py
python 4_misinfo_classifier.py
python 6_prep_final.py
```

Finally, launch the interactive dashboard:
```bash
streamlit run final_dashboard.py
```
