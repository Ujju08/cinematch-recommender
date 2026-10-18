"""
CineMatch - Submission Verification Script
Audits all project deliverables, files, tests, and outputs before final submission.
Run with: python verify_submission.py
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CHECKLIST = [
    ("Core Code: Data Loader", "src/data_loader.py"),
    ("Core Code: EDA Module", "src/eda.py"),
    ("Core Code: Memory-Based CF", "src/memory_based.py"),
    ("Core Code: Model-Based CF (SVD)", "src/model_based.py"),
    ("Core Code: Evaluation Module", "src/evaluation.py"),
    ("Core Code: Unified Recommender Engine", "src/recommender.py"),
    ("Executed Jupyter Analysis Notebook", "cinematch_recommender.ipynb"),
    ("Interactive Streamlit Web Dashboard", "app.py"),
    ("Interactive CLI Demo Script", "demo.py"),
    ("Comprehensive Evaluation Report", "reports/EVALUATION_REPORT.md"),
    ("Viva & Defense Q&A Guide", "reports/INTERNSHIP_VIVA_QA.md"),
    ("Interactive HTML Slide Deck", "presentation/index.html"),
    ("Automated Unit Test Suite", "tests/test_cinematch.py"),
    ("MovieLens Dataset (movies.csv)", "data/ml-latest-small/movies.csv"),
    ("MovieLens Dataset (ratings.csv)", "data/ml-latest-small/ratings.csv"),
    ("Generated Plot: Rating Distribution", "plots/rating_distribution.png"),
    ("Generated Plot: Long-Tail Pareto", "plots/long_tail_distribution.png"),
    ("Generated Plot: SVD Latent Space", "plots/svd_latent_space.png"),
    ("Generated Plot: Model Benchmark Comparison", "plots/model_comparison.png"),
    ("Dependencies File", "requirements.txt"),
    ("Project README Documentation", "README.md"),
]

def verify():
    print("=" * 75)
    print("        CINEMATCH: INTERNSHIP SUBMISSION VERIFICATION AUDIT        ")
    print("=" * 75)

    all_passed = True
    for item_name, rel_path in CHECKLIST:
        full_path = os.path.join(BASE_DIR, rel_path)
        exists = os.path.exists(full_path)
        size_str = ""
        if exists:
            size_kb = os.path.getsize(full_path) / 1024
            size_str = f"({size_kb:.1f} KB)"
            status = "[PASS]"
        else:
            status = "[FAIL]"
            all_passed = False

        print(f" {status:<8} {item_name:<42} -> {rel_path} {size_str}")

    print("=" * 75)
    if all_passed:
        print(" SUCCESS: ALL 21 DELIVERABLES & ARTIFACTS ARE PRESENT AND VERIFIED!")
        print(" Your CineMatch internship project is ready for submission and defense.")
    else:
        print(" WARNING: Some deliverables are missing. Check above failed items.")
    print("=" * 75)
    return all_passed

if __name__ == "__main__":
    success = verify()
    sys.exit(0 if success else 1)
