"""
CineMatch - Command-Line Interface (CLI) Demo
Generates interactive logs and demonstrations for internship presentations.
Supports:
  - Personalized recommendations for any User ID (e.g. User #101)
  - Similar movie discovery ('People who liked Toy Story also liked...')
  - Custom user taste profile simulator
  - Cold-start handling for new users
  - Model benchmarking summary
"""

import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import argparse
from src.recommender import CineMatchEngine, print_recommendations


def run_demo(user_id: int = 101, model: str = "svd", top_n: int = 5):
    print("=" * 70)
    print("   CINEMATCH: INTELLIGENT COLLABORATIVE FILTERING ENGINE   ")
    print("                 Internship Project Demo                   ")
    print("=" * 70)

    engine = CineMatchEngine()

    # 1. Existing User Demo (User #101)
    print(f"\n>>> [DEMO 1] Personalized Recommendations for User #{user_id} ({model.upper()} Model)")
    history = engine.get_user_history(user_id)
    if not history.empty:
        print(f"Top 3 Rated Movies by User #{user_id}:")
        for _, row in history.head(3).iterrows():
            print(f"   * {row['clean_title']} ({row['year']}) -> {row['rating']} stars [{row['genres']}]")
    
    recs = engine.recommend_movies(user_id=user_id, model=model, top_n=top_n)
    print_recommendations(user_id, recs)

    # 2. Item-Item Similarity Demo (Toy Story Case Study)
    print("=" * 70)
    print(">>> [DEMO 2] Item-Item Collaborative Filtering ('People who liked Toy Story also liked...')")
    toy_sims = engine.get_similar_movies("Toy Story", top_n=top_n)
    print(f"Movies most similar to 'Toy Story' based on rating patterns:")
    for i, s in enumerate(toy_sims, 1):
        print(f"  {i}. {s['title']} ({s['year']}) | Similarity: {s['similarity_score']:.3f} | Genres: {s['genres']}")

    # 3. Custom Profile: Animation Lover (Toy Story & Lion King fan)
    print("\n" + "=" * 70)
    print(">>> [DEMO 3] Taste Profile Simulator (User who loves Toy Story & Lion King)")
    custom_recs = engine.recommend_for_custom_profile(["Toy Story", "Lion King"], top_n=top_n)
    print(f"Generated Suggestions for Profile liking ['Toy Story', 'Lion King']:")
    for i, r in enumerate(custom_recs, 1):
        print(f"  {i}. {r['title']} ({r['year']}) | Score: {r['predicted_rating']} | Reason: {r['reason']}")

    # 4. Cold-Start Demo (New user with zero history)
    print("\n" + "=" * 70)
    print(">>> [DEMO 4] Cold-Start Problem Handling (Brand New User with 0 History)")
    cold_recs = engine.recommend_movies(user_id=-1, top_n=top_n)
    print("Bayesian Popularity Fallback (Weighted IMDB formula):")
    for i, r in enumerate(cold_recs, 1):
        print(f"  {i}. {r['title']} ({r['year']}) | Score: {r['predicted_rating']} | {r['reason']}")

    print("\n" + "=" * 70)
    print("Demo successfully completed!")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CineMatch Recommendation Engine CLI")
    parser.add_argument("--user", type=int, default=101, help="User ID to generate recommendations for (default: 101)")
    parser.add_argument("--model", type=str, default="svd", choices=["svd", "item_item", "user_user"], help="Model type")
    parser.add_argument("--top_n", type=int, default=5, help="Number of recommendations (default: 5)")
    args = parser.parse_args()

    run_demo(user_id=args.user, model=args.model, top_n=args.top_n)
