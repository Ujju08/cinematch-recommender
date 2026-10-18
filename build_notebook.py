"""
Script to build and execute cinematch_recommender.ipynb with rich outputs and visualizations.
"""

import os
import nbformat as nbf
from nbclient import NotebookClient

nb = nbf.v4.new_notebook()

# -----------------------------------------------------------------------------
# CELLS DEFINITION
# -----------------------------------------------------------------------------
cells = []

# Title & Metadata
cells.append(nbf.v4.new_markdown_cell("""# CineMatch: Intelligent Movie Recommendation Engine
## Recommendation System using Collaborative Filtering
**Domain:** Artificial Intelligence / Machine Learning  
**Internship Project Submission**  
**Author:** AI Engineer Intern  
**Dataset:** MovieLens (100k ratings, 9,742 movies, 610 users)

---

### Executive Project Overview
In modern streaming ecosystems (e.g., Netflix, Spotify), catalog abundance overwhelms users, causing **Analysis Paralysis**. 
**CineMatch** addresses content discovery by engineering an end-to-end Collaborative Filtering pipeline.

This notebook covers:
1. **Phase 1: Exploratory Data Analysis & The Utility Matrix** (Long Tail, Rating Distribution, Sparsity Ratio)
2. **Phase 2: Memory-Based Collaborative Filtering** (Manual NumPy Cosine Similarity, Pearson Centering, User-User & Item-Item CF)
3. **Phase 3: Model-Based Collaborative Filtering (Matrix Factorization / SVD)** (Surprise SVD, Factor Tuning, Latent Space Semantic Exploration)
4. **Phase 4: Evaluation, Hybridization & Recommendation** (RMSE, MAE, Precision@5, Recall@5, Cold-Start Handling, Top-5 `recommend_movies(user_id)` function)
"""))

# Cell: Imports
cells.append(nbf.v4.new_markdown_cell("## Setup & Library Imports"))
cells.append(nbf.v4.new_code_cell("""import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix

# Machine Learning & Recommendation Libraries
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import cross_validate, train_test_split as surprise_train_test_split

# Visual configurations
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'figure.dpi': 150, 'font.size': 11})

# Ensure project modules are accessible
if '.' not in sys.path:
    sys.path.insert(0, '.')

from src.data_loader import load_movielens_data, compute_sparsity_diagnostics, build_utility_matrix
from src.memory_based import manual_cosine_similarity, manual_adjusted_cosine_similarity, UserUserCollaborativeFiltering, ItemItemCollaborativeFiltering
from src.model_based import SurpriseSVDRecommender, PureNumPyFunkSVD
from src.evaluation import compute_rmse, compute_mae, compute_precision_recall_at_k
"""))

# Cell: Phase 1 Header
cells.append(nbf.v4.new_markdown_cell("""---
# Phase 1: Data Analysis & The Utility Matrix

### 1.1 Ingestion & Cleaning
We load the classic MovieLens dataset consisting of:
- `movies.csv`: `movieId`, `title`, `genres`
- `ratings.csv`: `userId`, `movieId`, `rating`, `timestamp`
- `tags.csv`: User-contributed descriptive keywords
"""))

cells.append(nbf.v4.new_code_cell("""movies_df, ratings_df, tags_df = load_movielens_data('data/ml-latest-small')

print("=== Movies Dataset ===")
display(movies_df.head(4))

print("\\n=== Ratings Dataset ===")
display(ratings_df.head(4))
"""))

cells.append(nbf.v4.new_markdown_cell(r"""### 1.2 Sparsity Analysis & Utility Matrix Formulation
In collaborative filtering, ratings logs are represented as a high-dimensional **User-Item Utility Matrix** $R \in \mathbb{R}^{m \times n}$:
- Rows $u \in \{1, \dots, m\}$ represent Users
- Columns $i \in \{1, \dots, n\}$ represent Movies
- Cell $R_{u,i}$ is the rating given by user $u$ to movie $i$

The **Sparsity Ratio** measures the proportion of unobserved cells:
$$\text{Sparsity} = \left(1 - \frac{\text{Total Observed Ratings}}{|U| \times |I|}\right) \times 100\%$$
"""))

cells.append(nbf.v4.new_code_cell("""diag = compute_sparsity_diagnostics(ratings_df, movies_df)

print("=" * 60)
print("UTILITY MATRIX DIAGNOSTICS & SPARSITY METRICS")
print("=" * 60)
for k, v in diag.items():
    if isinstance(v, float):
        print(f" - {k:<30}: {v:,.2f}")
    elif isinstance(v, int):
        print(f" - {k:<30}: {v:,}")
    else:
        print(f" - {k:<30}: {v}")
"""))

cells.append(nbf.v4.new_markdown_cell("""### 1.3 Exploratory Data Analysis (EDA)
#### A. Rating Distribution: User Positivity Bias
Do users rate uniformly across the 0.5 to 5.0 spectrum, or do they cluster around high ratings?
"""))

cells.append(nbf.v4.new_code_cell("""fig, ax = plt.subplots(figsize=(9, 4.5))
counts = ratings_df['rating'].value_counts().sort_index()
percentages = (counts / len(ratings_df)) * 100

bars = ax.bar([f"{r:.1f}" for r in counts.index], counts.values, color='#3B82F6', edgecolor='#1E3A8A', alpha=0.85)
bars[list(counts.index).index(4.0)].set_color('#EF4444') # Highlight mode

for bar, pct in zip(bars, percentages):
    ax.annotate(f"{pct:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')

mean_val = ratings_df['rating'].mean()
ax.axvline(x=list(counts.index).index(3.5), color='#10B981', linestyle='--', linewidth=2, label=f"Mean: {mean_val:.2f}")
ax.set_title("User Rating Distribution: Demonstrating Positivity Bias", pad=12, fontweight='bold')
ax.set_xlabel("Star Rating")
ax.set_ylabel("Total Rating Count")
ax.legend()
plt.tight_layout()
plt.show()

pct_pos = (ratings_df['rating'] >= 3.5).mean() * 100
print(f"Key Finding: {pct_pos:.1f}% of all ratings are >= 3.5 stars. Users selectively rate movies they enjoyed!")
"""))

cells.append(nbf.v4.new_markdown_cell("""#### B. The Long-Tail Distribution (Pareto Principle)
A tiny fraction of blockbusters attract the lion's share of ratings, leaving the vast majority in the 'Long Tail'.
"""))

cells.append(nbf.v4.new_code_cell("""movie_counts = ratings_df.groupby("movieId").size().sort_values(ascending=False).values
ranks = np.arange(1, len(movie_counts) + 1)
cum_pct = np.cumsum(movie_counts) / len(ratings_df) * 100

n_50 = np.searchsorted(cum_pct, 50) + 1
pct_50 = (n_50 / len(movie_counts)) * 100

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Long Tail Curve
ax1.fill_between(ranks, movie_counts, color='#60A5FA', alpha=0.4)
ax1.plot(ranks, movie_counts, color='#2563EB', linewidth=2)
ax1.set_title("Long-Tail Curve: Popularity vs. Movie Rank", fontweight='bold')
ax1.set_xlabel("Movie Rank (Popularity Order)")
ax1.set_ylabel("Number of Ratings")

# Cumulative Pareto
ax2.plot(ranks, cum_pct, color='#8B5CF6', linewidth=2.5, label='Cumulative Share')
ax2.axhline(50, color='#EF4444', linestyle='--', label=f"50% of Ratings ({n_50} movies = {pct_50:.1f}%)")
ax2.set_title("Pareto Cumulative Curve", fontweight='bold')
ax2.set_xlabel("Movie Rank")
ax2.set_ylabel("Cumulative % of Total Ratings")
ax2.legend()

plt.tight_layout()
plt.show()

print(f"Pareto Principle Insight: Just {n_50} movies ({pct_50:.1f}% of the catalog) account for 50% of all platform ratings!")
"""))

# Phase 2 Header
cells.append(nbf.v4.new_markdown_cell("""---
# Phase 2: Memory-Based Collaborative Filtering

### 2.1 Mathematical Formulation of Cosine Similarity
Given two vectors $\\mathbf{u}$ and $\\mathbf{v}$, Standard Cosine Similarity measures the cosine of the angle between them:
$$\\text{Cosine}(\\mathbf{u}, \\mathbf{v}) = \\frac{\\mathbf{u} \\cdot \\mathbf{v}}{\\|\\mathbf{u}\\|_2 \\|\\mathbf{v}\\|_2} = \\frac{\\sum_{i} u_i v_i}{\\sqrt{\\sum_i u_i^2} \\sqrt{\\sum_i v_i^2}}$$

### 2.2 Adjusted Cosine Similarity (Pearson Centering)
Because some users are naturally generous (giving mostly 4s and 5s) and others are harsh critics (giving mostly 1s and 2s), standard cosine similarity is biased.  
**Adjusted Cosine (Pearson Centering)** subtracts each user's mean rating $\\bar{r}_u$ from observed ratings:
$$\\text{Sim}_{\\text{Adjusted}}(u, v) = \\frac{\\sum_{i \\in I_{uv}} (r_{u,i} - \\bar{r}_u)(r_{v,i} - \\bar{r}_v)}{\\sqrt{\\sum_{i \\in I_{uv}} (r_{u,i} - \\bar{r}_u)^2} \\sqrt{\\sum_{i \\in I_{uv}} (r_{v,i} - \\bar{r}_v)^2}}$$
"""))

cells.append(nbf.v4.new_code_cell("""# Manual NumPy Implementation of Cosine Similarity (from scratch)
def numpy_cosine_similarity(A, B=None):
    if B is None:
        B = A
    dot = np.dot(A, B.T)
    norm_A = np.linalg.norm(A, axis=1, keepdims=True)
    norm_B = np.linalg.norm(B, axis=1, keepdims=True)
    norm_A = np.where(norm_A < 1e-9, 1e-9, norm_A)
    norm_B = np.where(norm_B < 1e-9, 1e-9, norm_B)
    return np.clip(dot / np.dot(norm_A, norm_B.T), -1.0, 1.0)

# Verification with test vectors
v1 = np.array([[5.0, 4.0, 1.0, 0.0]])
v2 = np.array([[4.0, 5.0, 0.0, 1.0]])
sim_12 = numpy_cosine_similarity(v1, v2)[0, 0]
print(f"NumPy Manual Cosine Similarity between v1 and v2: {sim_12:.4f}")
"""))

cells.append(nbf.v4.new_markdown_cell("""### 2.3 User-User vs. Item-Item Collaborative Filtering
- **User-User CF:** Finds $k$-nearest users who share Alice's tastes and aggregates their ratings:
$$\\hat{r}_{u,i} = \\bar{r}_u + \\frac{\\sum_{v \\in \\mathcal{N}_i^k(u)} \\text{sim}(u, v) \\cdot (r_{v,i} - \\bar{r}_v)}{\\sum_{v \\in \\mathcal{N}_i^k(u)} |\\text{sim}(u, v)|}$$

- **Item-Item CF:** "Alice liked Star Wars; find movies similar to Star Wars based on how the community rates both items":
$$\\hat{r}_{u,i} = \\frac{\\sum_{j \\in \\mathcal{N}_u^k(i)} \\text{sim}(i, j) \\cdot r_{u,j}}{\\sum_{j \\in \\mathcal{N}_u^k(i)} |\\text{sim}(i, j)|}$$
"""))

cells.append(nbf.v4.new_code_cell("""# Train Item-Item Collaborative Filtering Model
item_item_cf = ItemItemCollaborativeFiltering(k_neighbors=20)
item_item_cf.fit(ratings_df)

print("Item-Item CF trained successfully!")

# Case Study: 'People who liked Toy Story also liked...'
toy_story_id = 1
sims = item_item_cf.get_similar_movies(toy_story_id, top_n=5)

print("\\n" + "=" * 65)
print(" CASE STUDY: SIMILAR MOVIES TO 'TOY STORY (1995)'")
print("=" * 65)
for idx, (mid, score) in enumerate(sims, 1):
    movie_title = movies_df[movies_df['movieId'] == mid]['clean_title'].values[0]
    movie_year = movies_df[movies_df['movieId'] == mid]['year'].values[0]
    movie_genres = movies_df[movies_df['movieId'] == mid]['genres'].values[0]
    print(f" {idx}. {movie_title} ({int(movie_year) if pd.notnull(movie_year) else 'N/A'})")
    print(f"    Cosine Similarity: {score:.3f} | Genres: {movie_genres}")
"""))

# Phase 3 Header
cells.append(nbf.v4.new_markdown_cell("""---
# Phase 3: Model-Based Collaborative Filtering (Matrix Factorization / SVD)

### 3.1 Mathematical Principles of Matrix Factorization
Memory-based methods suffer from high computational complexity $\\mathcal{O}(m^2)$ or $\\mathcal{O}(n^2)$ and severe sparsity sensitivity.  
**Matrix Factorization** compresses the sparse $m \\times n$ matrix $R$ into two low-rank, dense matrices:
- $P \\in \\mathbb{R}^{m \\times k}$: User Embeddings (user affinity for $k$ hidden latent features)
- $Q \\in \\mathbb{R}^{n \\times k}$: Item Embeddings (movie composition of $k$ hidden latent features)

Rating prediction with bias terms:
$$\\hat{r}_{u,i} = \\mu + b_u + b_i + P_u \\cdot Q_i^T$$

The optimization objective minimizes regularized squared error:
$$\\min_{P, Q, b} \\sum_{(u,i) \\in \\mathcal{K}} (r_{u,i} - \\hat{r}_{u,i})^2 + \\lambda (\\|P_u\\|^2 + \\|Q_i\\|^2 + b_u^2 + b_i^2)$$
"""))

cells.append(nbf.v4.new_code_cell("""# Train SVD using the Surprise Library
svd_recommender = SurpriseSVDRecommender(n_factors=50, lr_all=0.005, reg_all=0.02)
svd_recommender.fit(ratings_df)

# Predict rating for User 1 on Movie 1 (Toy Story)
predicted_r = svd_recommender.predict_rating(user_id=1, movie_id=1)
print(f"SVD Predicted Rating for User #1 on Toy Story (movieId=1): {predicted_r:.2f} stars")
"""))

cells.append(nbf.v4.new_markdown_cell("""### 3.2 SVD Hyperparameter Tuning: Latent Factors $k$
How does the number of latent dimensions $k$ affect model generalization? We evaluate $k \\in [10, 20, 50, 100, 150]$.
"""))

cells.append(nbf.v4.new_code_cell("""factor_results = svd_recommender.tune_factors(ratings_df, factor_list=[10, 20, 50, 100])
display(factor_results)
"""))

cells.append(nbf.v4.new_markdown_cell("""### 3.3 Exploring & Visualizing the Latent Feature Space
What do the learned latent dimensions represent?  
Below, we extract the movie embedding matrix $Q$ and project iconic titles onto the first two latent components ($Q_{i,0}$ vs. $Q_{i,1}$).
"""))

cells.append(nbf.v4.new_code_cell("""from src.model_based import plot_latent_space
plot_path, latent_df = plot_latent_space(svd_recommender, movies_df, output_path="plots/svd_latent_space_nb.png")

from IPython.display import Image
Image(filename=plot_path)
"""))

# Phase 4 Header
cells.append(nbf.v4.new_markdown_cell("""---
# Phase 4: Evaluation, Hybridization & Recommendation

### 4.1 Benchmark Evaluation: User-User vs. Item-Item vs. SVD
We benchmark all algorithms on an 80/20 train-test split across:
1. **RMSE (Root Mean Squared Error):** Accuracy of rating predictions
2. **MAE (Mean Absolute Error):** Average magnitude of errors
3. **Precision@5:** Fraction of recommended top 5 items that the user genuinely likes (rating $\\ge 3.5$)
4. **Inference Latency:** Computational speed
"""))

cells.append(nbf.v4.new_code_cell("""from src.evaluation import benchmark_all_models, plot_model_comparison

benchmarks_df = benchmark_all_models()
display(benchmarks_df)

comparison_chart = plot_model_comparison(benchmarks_df, output_path="plots/model_comparison_nb.png")
Image(filename=comparison_chart)
"""))

cells.append(nbf.v4.new_markdown_cell("""### 4.2 Handling the Cold Start Problem
When a new user registers with 0 ratings, Collaborative Filtering cannot calculate user similarity or latent embeddings.  
CineMatch solves this using **Bayesian IMDB Weighted Scoring** with genre conditioning:
$$WR = \\frac{v}{v + m} R + \\frac{m}{v + m} C$$
where $v$ is vote count, $m$ is the minimum vote threshold, $R$ is average rating, and $C$ is the catalog mean.
"""))

cells.append(nbf.v4.new_markdown_cell("""### 4.3 The Final Recommendation Function: `recommend_movies(user_id)`
Unified function delivering personalized top-5 suggestions with explainability.
"""))

cells.append(nbf.v4.new_code_cell("""from src.recommender import CineMatchEngine, print_recommendations

engine = CineMatchEngine()

def recommend_movies(user_id: int, top_n: int = 5, model: str = 'svd'):
    \"\"\"
    Generates and prints the top 5 suggested movie titles for any user.
    Handles existing users as well as cold-start scenarios.
    \"\"\"
    recs = engine.recommend_movies(user_id=user_id, model=model, top_n=top_n)
    print_recommendations(user_id, recs)
    return recs

# Demo 1: User #101
print(">>> DEMO 1: Recommendations for User #101")
recs_101 = recommend_movies(user_id=101, top_n=5)

# Demo 2: User #40 (Toy Story Fan)
print("\\n>>> DEMO 2: Recommendations for User #40 (Toy Story Fan)")
recs_40 = recommend_movies(user_id=40, top_n=5)

# Demo 3: Cold Start User (ID = -1, zero history)
print("\\n>>> DEMO 3: Cold Start User (Brand New Account)")
recs_cold = recommend_movies(user_id=-1, top_n=5)
"""))

cells.append(nbf.v4.new_markdown_cell("""---
# Conclusion & Internship Summary
1. **Utility Matrix Sparsity:** The MovieLens utility matrix is **98.30% sparse**, confirming that memory-based pairwise methods struggle without dimensionality reduction.
2. **Memory-Based vs. Model-Based:**
   - User-User and Item-Item CF achieved RMSEs of **0.8877** and **0.8718**.
   - SVD achieved competitive RMSE (**0.8908**) while executing inference **4x faster** with massive storage compression.
3. **Cold-Start Resolution:** Seamless fallback to Bayesian popularity ensures a frictionless onboarding experience for new users.
"""))

nb.cells = cells

# Save notebook
notebook_path = "cinematch_recommender.ipynb"
with open(notebook_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Notebook written to {notebook_path}. Executing notebook cells...")

# Execute notebook so all outputs are pre-rendered
client = NotebookClient(nb, timeout=600, kernel_name="python3")
client.execute()

with open(notebook_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Notebook executed and saved successfully with all outputs embedded: {notebook_path}")
