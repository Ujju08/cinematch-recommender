# CineMatch: Intelligent Recommendation Engine using Collaborative Filtering

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework-Surprise](https://img.shields.io/badge/Surprise-SVD-orange.svg)](http://surpriselib.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Interactive%20App-red.svg)](https://streamlit.io/)
[![Dataset](https://img.shields.io/badge/Dataset-MovieLens%20Latest--Small-green.svg)](https://grouplens.org/datasets/movielens/)

> **Internship Case Study: CineMatch**  
> In the streaming era (Netflix, Spotify), content abundance causes **"Analysis Paralysis"**. CineMatch builds an Intelligent Recommendation Engine that learns from past ratings and watch histories to predict future user preferences using **Memory-Based** and **Model-Based Collaborative Filtering**.

---

## Table of Contents
- [Project Architecture & Execution Plan](#project-architecture--execution-plan)
- [Key Features & Deliverables](#key-features--deliverables)
- [Experimental Benchmark & Evaluation](#experimental-benchmark--evaluation)
- [Quick Start Guide](#quick-start-guide)
  - [1. Running the Interactive Streamlit Web App](#1-running-the-interactive-streamlit-web-app)
  - [2. Running the CLI Demo](#2-running-the-cli-demo)
  - [3. Running the Jupyter Notebook](#3-running-the-jupyter-notebook)
- [Project Directory Structure](#project-directory-structure)
- [Internship Deliverables Checklist](#internship-deliverables-checklist)

---

## Project Architecture & Execution Plan

```
                              Raw MovieLens Data
                         (Ratings, Movies, User Logs)
                                      |
                                      v
                        [Phase 1: Utility Matrix]
                  Rows: Users (610) | Columns: Movies (9,724)
                           Sparsity Ratio: 98.30%
                                      |
               +----------------------+----------------------+
               |                                             |
               v                                             v
    [Phase 2: Memory-Based CF]                   [Phase 3: Model-Based CF]
  - Manual NumPy Cosine Similarity             - Singular Value Decomposition (SVD)
  - Pearson Centering (User Mean)              - Compress to Latent Embeddings (P, Q)
  - User-User & Item-Item k-NN                 - 2D Latent Space Semantic Mapping
               |                                             |
               +----------------------+----------------------+
                                      |
                                      v
                        [Phase 4: Evaluation & Hybrid Engine]
                  - RMSE, MAE, Precision@5, Recall@5 Benchmarks
                  - Bayesian IMDB Weighted Cold-Start Fallback
                  - Unified recommend_movies(user_id) Function
```

---

## Key Features & Deliverables

### 1. Phase 1: Data Analysis & The Utility Matrix
- **Sparsity Ratio:** Calculated at **98.30%** (out of 5.93M possible user-movie combinations, only 100,836 are observed).
- **The Long Tail Distribution:** Visualized and verified the **Pareto Principle**: just **659 movies (6.78% of catalog)** account for **50% of all ratings**.
- **Rating Positivity Bias:** Quantified that users rate favorably (**61.2% of ratings are $\ge 3.5$ stars**, mode at **4.0 stars**).

### 2. Phase 2: Memory-Based Collaborative Filtering
- **Manual NumPy Cosine Similarity:** Built from first principles without black-box dependencies.
- **Adjusted Cosine (Pearson Centering):** Normalized user vectors by subtracting individual mean ratings $\bar{r}_u$ to eliminate user rating biases.
- **Item-Item Collaborative Filtering:** Answers *"People who liked Toy Story also liked..."* (recommending *Toy Story 2*, *Toy Story 3*, *Aladdin*, *Wallace & Gromit*, and *Beauty and the Beast*).

### 3. Phase 3: Model-Based Collaborative Filtering (Matrix Factorization)
- **Surprise SVD:** Decomposed the sparse utility matrix into dense User Factors $P$ ($610 \times 50$) and Item Factors $Q$ ($9,724 \times 50$) with user and item baseline biases.
- **Latent Space Exploration:** Projected movie vectors onto a 2D plane ($Q_0$ vs $Q_1$), uncovering unsupervised semantic axes (Blockbuster Action vs Arthouse Drama, Animated Family vs Dark Psychological Thrillers).
- **Pedagogical FunkSVD:** Included a pure NumPy SGD matrix factorization implementation for complete theoretical transparency.

### 4. Phase 4: Cold-Start Resolution & Hybridization
- Solved the **Cold Start Problem** for new users ($0$ ratings) using the **Bayesian IMDB Weighted Rating Formula**:
  $$WR = \frac{v}{v + m} R + \frac{m}{v + m} C$$
- Integrated onboarding genre elicitation.

---

## Experimental Benchmark & Evaluation

Evaluated on an 80/20 train-test split ($80,668$ train ratings, $20,168$ test ratings):

| Model | Category | RMSE $\downarrow$ | MAE $\downarrow$ | Precision@5 $\uparrow$ | Recall@5 $\uparrow$ | Inference Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Global Baseline** | Heuristic | 1.0488 | 0.8316 | 63.59% | 42.11% | **0.0007s** |
| **User-User CF** | Memory-Based | 0.8877 | 0.6750 | 56.63% | 78.24% | 0.7446s |
| **Item-Item CF** | Memory-Based | **0.8718** | **0.6607** | **0.5711** | **0.7850** | 0.8727s |
| **Surprise SVD** | Model-Based | 0.8908 | 0.6830 | 55.76% | 77.73% | **0.2091s (4x faster)** |

> **Key Takeaway for Evaluation:** While Item-Item CF achieved slightly lower RMSE on this dataset size, **SVD is vastly superior in production systems**:
> 1. Inference is **$4\times$ faster** via constant-time vector dot products.
> 2. Compresses matrix storage by **$99.4\%$** (from $378$ MB to $2$ MB).
> 3. Does not suffer memory exhaustion on massive catalogs.

---

## Quick Start Guide

### Setup Environment
```bash
# Clone or navigate to the project directory
cd cinematch-recommender

# Install prerequisites
pip install -r requirements.txt
```

### 1. Running the Interactive Streamlit Web App
Launch the full web dashboard with interactive tabs for recommendations, item similarity search, cold-start simulation, and EDA:
```bash
streamlit run app.py
```

### 2. Running the CLI Demo
Execute the terminal demo for any User ID (e.g. User #101):
```bash
# Default demo (User #101 with SVD)
python demo.py

# Custom user and algorithm selection
python demo.py --user 40 --model item_item --top_n 5
```

### 3. Running the Jupyter Notebook
Open the pre-executed analysis notebook in Jupyter:
```bash
jupyter notebook cinematch_recommender.ipynb
```
*Note: The notebook is already fully executed with all embedded charts, formulas, and diagnostic logs.*

---

## Project Directory Structure

```
cinematch-recommender/
├── app.py                         # Interactive Streamlit Web Application
├── demo.py                        # Standalone CLI Demo script
├── build_notebook.py              # Script that built & executed the notebook
├── cinematch_recommender.ipynb    # Executed Jupyter Analysis Notebook
├── requirements.txt               # Dependencies
├── README.md                      # Project documentation
│
├── data/
│   └── ml-latest-small/           # MovieLens dataset (movies.csv, ratings.csv, etc.)
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py             # Data ingestion, cleaning, utility matrix & sparsity
│   ├── eda.py                     # Long-tail, Pareto, and rating distribution plots
│   ├── memory_based.py            # NumPy Cosine Similarity, User-User & Item-Item CF
│   ├── model_based.py             # Surprise SVD, factor tuning & latent space exploration
│   ├── evaluation.py              # RMSE, MAE, Precision@5, Recall@5 benchmarking
│   └── recommender.py             # Unified CineMatchEngine & cold-start fallback
│
├── plots/
│   ├── rating_distribution.png    # Positivity bias chart
│   ├── long_tail_distribution.png # Long tail & Pareto curve
│   ├── user_activity_distribution.png # User rating frequency skew
│   ├── top_rated_movies.png       # Catalog blockbuster rankings
│   ├── svd_latent_space.png       # 2D SVD movie embeddings projection
│   └── model_comparison.png       # Side-by-side benchmark bar chart
│
└── reports/
    └── EVALUATION_REPORT.md       # Comprehensive internship technical report
```

---

## Internship Deliverables Checklist

- [x] **Phase 1: Data Analysis & The Utility Matrix**
  - [x] MovieLens dataset downloaded and preprocessed.
  - [x] Long Tail distribution plotted and analyzed.
  - [x] Rating distribution plotted showing user positivity bias.
  - [x] Utility matrix constructed and Sparsity Ratio ($98.30\%$) calculated.
- [x] **Phase 2: Memory-Based Collaborative Filtering**
  - [x] Manual Cosine Similarity implemented in pure NumPy.
  - [x] Adjusted Cosine (Pearson Centering) implemented to account for rating bias.
  - [x] User-User Collaborative Filtering implemented.
  - [x] Item-Item Collaborative Filtering implemented (*"People who liked Toy Story also liked..."*).
- [x] **Phase 3: Model-Based Collaborative Filtering**
  - [x] SVD implemented using the `Surprise` library.
  - [x] Latent factor tuning evaluated ($k=10, 20, 50, 100$).
  - [x] Latent feature space ($Q$) explored and visualized in 2D with genre semantic interpretation.
  - [x] Pure NumPy FunkSVD SGD implemented for complete mathematical transparency.
- [x] **Phase 4: Evaluation & Hybridization**
  - [x] RMSE and MAE benchmarked across Baseline, User-User, Item-Item, and SVD.
  - [x] Precision@5 and Recall@5 ranking metrics evaluated.
  - [x] Cold-start problem solved with Bayesian IMDB weighted rating.
  - [x] Top-5 `recommend_movies(user_id)` function implemented with explainability.
- [x] **Deliverables:**
  - [x] Fully executed Jupyter Notebook (`cinematch_recommender.ipynb`).
  - [x] Formal Evaluation Report (`reports/EVALUATION_REPORT.md`).
  - [x] Interactive Streamlit Web Application (`app.py`).
  - [x] Working CLI Demo script (`demo.py`).
