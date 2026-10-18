"""
CineMatch - Interactive Streamlit Web Application
An interactive dashboard for exploring MovieLens data, comparing Collaborative Filtering
algorithms, testing recommendations for any User ID, and discovering similar movies.
"""

import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

from src.data_loader import load_movielens_data, compute_sparsity_diagnostics
from src.recommender import CineMatchEngine

# Streamlit Page Config
st.set_page_config(
    page_title="CineMatch | AI Recommendation Engine",
    page_icon="\U0001F3AC",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 800;
        color: #E11D48;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
    }
    .movie-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        transition: transform 0.2s;
    }
    .movie-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #0F172A;
    }
    .rating-badge {
        background-color: #FEF3C7;
        color: #B45309;
        font-weight: bold;
        padding: 3px 8px;
        border-radius: 5px;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_engine():
    return CineMatchEngine()


engine = get_engine()
movies_df, ratings_df, _ = engine.movies_df, engine.ratings_df, engine.tags_df
sparsity_stats = compute_sparsity_diagnostics(ratings_df, movies_df)

# Sidebar Navigation
st.sidebar.markdown("# \U0001F3AC CineMatch")
st.sidebar.markdown("**Collaborative Filtering Engine**")
st.sidebar.caption("Internship Case Study & Demo")

nav_choice = st.sidebar.radio(
    "Navigation",
    [
        "Personalized Recommender",
        "Item-Item Similarity Finder",
        "Cold Start Simulator",
        "Exploratory Data Analysis (EDA)",
        "Model Benchmarks & Latent Space"
    ]
)

st.sidebar.divider()
st.sidebar.markdown("### Matrix Sparsity Stats")
st.sidebar.metric("Sparsity Ratio", f"{sparsity_stats['sparsity_ratio_active']:.2f}%")
st.sidebar.metric("Total Ratings", f"{sparsity_stats['total_ratings']:,}")
st.sidebar.metric("Catalog Movies", f"{sparsity_stats['n_total_movies']:,}")
st.sidebar.metric("Active Users", f"{sparsity_stats['n_users']:,}")


# -----------------------------------------------------------------------------
# TAB 1: PERSONALIZED RECOMMENDER
# -----------------------------------------------------------------------------
if nav_choice == "Personalized Recommender":
    st.markdown('<div class="main-header">\U0001F3AC Personalized Movie Recommendations</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Select a user profile to predict what they will enjoy next based on collaborative filtering.</div>', unsafe_allow_html=True)

    col_ctrl1, col_ctrl2, col_ctrl3, col_ctrl4 = st.columns([1.5, 1.5, 1, 1.5])
    with col_ctrl1:
        user_id = st.number_input("User ID", min_value=1, max_value=int(ratings_df["userId"].max()), value=101, step=1)
    with col_ctrl2:
        model_name = st.selectbox(
            "Algorithm",
            ["svd", "item_item", "user_user"],
            format_func=lambda x: "SVD Matrix Factorization" if x == "svd" else ("Item-Item CF" if x == "item_item" else "User-User CF")
        )
    with col_ctrl3:
        top_k = st.slider("Top N", min_value=3, max_value=15, value=5)
    with col_ctrl4:
        genre_filter = st.selectbox(
            "Filter by Genre (Optional)",
            ["All Genres", "Action", "Adventure", "Animation", "Comedy", "Crime", "Drama", "Fantasy", "Horror", "Mystery", "Romance", "Sci-Fi", "Thriller"]
        )
        selected_genre = None if genre_filter == "All Genres" else genre_filter

    user_history = engine.get_user_history(user_id)
    
    col_hist, col_recs = st.columns([1, 1.3])

    with col_hist:
        st.subheader(f"\U0001F4DA User #{user_id}'s Favorites")
        if user_history.empty:
            st.warning("No ratings recorded for this user.")
        else:
            st.caption(f"Showing top rated movies out of {len(user_history)} total ratings:")
            for _, row in user_history.head(5).iterrows():
                st.markdown(f"""
                <div class="movie-card">
                    <div class="movie-title">{row['clean_title']} ({row['year']})</div>
                    <small><b>Genres:</b> {row['genres']}</small><br>
                    <span class="rating-badge">\u2605 {row['rating']:.1f} / 5.0</span>
                </div>
                """, unsafe_allow_html=True)

    with col_recs:
        st.subheader(f"\u2728 CineMatch Suggestions ({model_name.upper()})")
        with st.spinner("Calculating personalized recommendations..."):
            recs = engine.recommend_movies(
                user_id=user_id,
                model=model_name,
                top_n=top_k,
                genre_filter=selected_genre
            )
        
        for idx, rec in enumerate(recs, 1):
            imdb_link = rec.get("imdb_url", f"https://www.imdb.com/find?q={rec['title']}")
            trailer_link = rec.get("trailer_url", f"https://www.youtube.com/results?search_query={rec['title']}+trailer")
            
            st.markdown(f"""
            <div class="movie-card" style="border-left: 4px solid #E11D48;">
                <div class="movie-title">#{idx}. <a href="{imdb_link}" target="_blank" style="color: #0F172A; text-decoration: none;">{rec['title']} ({rec['year']}) ↗</a></div>
                <small><b>Genres:</b> {rec['genres']}</small><br>
                <span class="rating-badge" style="background-color: #DCFCE7; color: #166534;">
                    Predicted Rating: {rec['predicted_rating']} / 5.0
                </span>
                <p style="margin-top: 6px; font-size: 0.85rem; color: #475569;">
                    <b>Why:</b> {rec['reason']}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            b1, b2 = st.columns(2)
            with b1:
                st.link_button(f"🎬 View on IMDb", imdb_link, use_container_width=True)
            with b2:
                st.link_button("▶ Watch Trailer (YouTube)", trailer_link, use_container_width=True)
            st.markdown("<div style='margin-bottom: 12px;'></div>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# TAB 2: ITEM-ITEM SIMILARITY FINDER
# -----------------------------------------------------------------------------
elif nav_choice == "Item-Item Similarity Finder":
    st.markdown('<div class="main-header">\U0001F50D "People Who Liked This Also Liked..."</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Discover mathematically similar movies based on pairwise rating co-occurrence patterns.</div>', unsafe_allow_html=True)

    sample_movies = ["Toy Story", "Star Wars: Episode IV - A New Hope", "Matrix, The", "Godfather, The", "Pulp Fiction", "Lion King, The", "Fight Club"]
    
    col_input1, col_input2 = st.columns([2, 1])
    with col_input1:
        query_title = st.selectbox("Select a benchmark movie (or type custom below):", sample_movies)
        custom_query = st.text_input("Or search any title in catalog:", "")
        target_title = custom_query.strip() if custom_query.strip() else query_title
    with col_input2:
        top_n_sim = st.slider("Number of Similar Movies", min_value=3, max_value=12, value=5)

    with st.spinner("Computing cosine similarities across user vectors..."):
        sim_results = engine.get_similar_movies(target_title, top_n=top_n_sim)

    if not sim_results:
        st.error(f"Movie matching '{target_title}' not found. Please try another title.")
    else:
        st.subheader(f"Movies most similar to: {target_title}")
        cols = st.columns(min(len(sim_results), 3))
        for idx, res in enumerate(sim_results):
            col_target = cols[idx % 3]
            s_imdb = res.get("imdb_url", f"https://www.imdb.com/find?q={res['title']}")
            s_trailer = res.get("trailer_url", f"https://www.youtube.com/results?search_query={res['title']}+trailer")
            with col_target:
                st.markdown(f"""
                <div class="movie-card" style="border-top: 4px solid #3B82F6;">
                    <div class="movie-title"><a href="{s_imdb}" target="_blank" style="color: #0F172A; text-decoration: none;">{res['title']} ({res['year']}) ↗</a></div>
                    <small><b>Genres:</b> {res['genres']}</small><br><br>
                    <span class="rating-badge" style="background-color: #DBEAFE; color: #1E40AF;">
                        Cosine Sim: {res['similarity_score']:.3f}
                    </span>
                </div>
                """, unsafe_allow_html=True)
                st.link_button("🎬 IMDb Page", s_imdb, use_container_width=True)
                st.link_button("▶ Trailer", s_trailer, use_container_width=True)


# -----------------------------------------------------------------------------
# TAB 3: COLD START SIMULATOR
# -----------------------------------------------------------------------------
elif nav_choice == "Cold Start Simulator":
    st.markdown('<div class="main-header">\u2744\ufe0f Cold Start Problem & Hybrid Fallback</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Simulate onboarding for a brand new user with 0 ratings using Bayesian popularity & preference elicitation.</div>', unsafe_allow_html=True)

    st.markdown("""
    > **The Cold-Start Challenge:** When a user first registers on CineMatch, they have zero interaction logs.
    > Pure collaborative filtering cannot compute similarities without history.
    > CineMatch addresses this using **Bayesian IMDB Weighted Scoring**:
    > $$WR = \\frac{v}{v + m} R + \\frac{m}{v + m} C$$
    """)

    col_cs1, col_cs2 = st.columns([1.5, 1])
    with col_cs1:
        preferred_genres = st.multiselect(
            "Select Initial Genre Preferences:",
            ["Action", "Adventure", "Animation", "Comedy", "Crime", "Drama", "Fantasy", "Horror", "Mystery", "Romance", "Sci-Fi", "Thriller"],
            default=["Animation"]
        )
    with col_cs2:
        cold_top_n = st.slider("Suggestions Count", 3, 10, 5)

    genre_regex = "|".join(preferred_genres) if preferred_genres else None
    cold_recs = engine.recommend_movies(user_id=-1, top_n=cold_top_n, genre_filter=genre_regex)

    st.subheader("\U0001F31F Onboarding Recommendations")
    for idx, rec in enumerate(cold_recs, 1):
        cs_imdb = rec.get("imdb_url", f"https://www.imdb.com/find?q={rec['title']}")
        cs_trailer = rec.get("trailer_url", f"https://www.youtube.com/results?search_query={rec['title']}+trailer")
        st.markdown(f"""
        <div class="movie-card" style="border-left: 4px solid #10B981;">
            <div class="movie-title">#{idx}. <a href="{cs_imdb}" target="_blank" style="color: #0F172A; text-decoration: none;">{rec['title']} ({rec['year']}) ↗</a></div>
            <small><b>Genres:</b> {rec['genres']}</small><br>
            <span class="rating-badge" style="background-color: #FEF3C7; color: #92400E;">
                Community Weighted Score: {rec['predicted_rating']} / 5.0
            </span>
            <p style="margin-top: 5px; font-size: 0.85rem; color: #475569;">
                <b>Strategy:</b> {rec['reason']}
            </p>
        </div>
        """, unsafe_allow_html=True)
        cs1, cs2 = st.columns(2)
        with cs1:
            st.link_button("🎬 IMDb Page", cs_imdb, use_container_width=True)
        with cs2:
            st.link_button("▶ Watch Trailer", cs_trailer, use_container_width=True)
        st.markdown("<div style='margin-bottom: 12px;'></div>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# TAB 4: EXPLORATORY DATA ANALYSIS
# -----------------------------------------------------------------------------
elif nav_choice == "Exploratory Data Analysis (EDA)":
    st.markdown('<div class="main-header">\U0001F4CA Exploratory Data Analysis & Utility Matrix</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Visualizing the Long-Tail distribution, user positivity bias, and extreme sparsity.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("1. Rating Distribution (Positivity Bias)")
        st.image("plots/rating_distribution.png", use_container_width=True)
        st.caption("Observation: Ratings are heavily concentrated at 4.0 and 5.0. Over 61% of ratings are >= 3.5.")

    with col2:
        st.subheader("2. The Long-Tail Distribution (Pareto Principle)")
        st.image("plots/long_tail_distribution.png", use_container_width=True)
        st.caption("Observation: Just 659 movies (6.8% of the catalog) generate 50% of all ratings.")

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("3. User Activity Distribution")
        st.image("plots/user_activity_distribution.png", use_container_width=True)
    with col4:
        st.subheader("4. Top Rated Blockbusters")
        st.image("plots/top_rated_movies.png", use_container_width=True)


# -----------------------------------------------------------------------------
# TAB 5: MODEL BENCHMARKS & LATENT SPACE
# -----------------------------------------------------------------------------
elif nav_choice == "Model Benchmarks & Latent Space":
    st.markdown('<div class="main-header">\U0001F3C6 Model Evaluation & SVD Latent Space</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Benchmarking RMSE, MAE, Precision@K, and exploring learned latent representations.</div>', unsafe_allow_html=True)

    col_b1, col_b2 = st.columns([1.2, 1])
    with col_b1:
        st.subheader("Model Evaluation Benchmark")
        if os.path.exists("plots/model_comparison.png"):
            st.image("plots/model_comparison.png", use_container_width=True)
        
        benchmark_table = pd.DataFrame([
            {"Model": "Global Baseline", "Type": "Heuristic", "RMSE": 1.0488, "MAE": 0.8316, "Precision@5": "63.6%", "Inference Speed": "Instant"},
            {"Model": "User-User CF", "Type": "Memory-Based", "RMSE": 0.8877, "MAE": 0.6750, "Precision@5": "56.6%", "Inference Speed": "0.74s"},
            {"Model": "Item-Item CF", "Type": "Memory-Based", "RMSE": 0.8718, "MAE": 0.6607, "Precision@5": "57.1%", "Inference Speed": "0.87s"},
            {"Model": "Surprise SVD", "Type": "Model-Based", "RMSE": 0.8908, "MAE": 0.6830, "Precision@5": "55.8%", "Inference Speed": "0.21s (4x faster)"},
        ])
        st.dataframe(benchmark_table, use_container_width=True)

    with col_b2:
        st.subheader("SVD Latent Space: Movie Embeddings ($Q$)")
        if os.path.exists("plots/svd_latent_space.png"):
            st.image("plots/svd_latent_space.png", use_container_width=True)
        st.caption("Visualizing the 2D projection of movie embeddings: clusters reveal latent dimensions capturing genre polarity (Animation vs Action vs Crime/Drama).")
