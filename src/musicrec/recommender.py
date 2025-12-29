"""
Compatibility façade for recommender modules.

Why this exists:
- The repo has two recommender modules:
    - recommender_knn.py
    - recommender_hybrid.py
- Some parts of the codebase/tests may still import:
    from musicrec.recommender import HybridRecommender, SimilarQuery, ...

This file keeps those imports stable by re-exporting the public API.
"""

from __future__ import annotations

# KNN / cosine baseline
from musicrec.recommender_knn import (  # noqa: F401
    KNNRecommender,
    SimilarQuery,
    SimilarTrack,
)

# Hybrid ranker (similarity + momentum + popularity + freshness)
from musicrec.recommender_hybrid import (  # noqa: F401
    HybridRecommender,
    HybridQuery,
    HybridWeights,
    HybridTrack,
)

__all__ = [
    # KNN
    "KNNRecommender",
    "SimilarQuery",
    "SimilarTrack",
    # Hybrid
    "HybridRecommender",
    "HybridQuery",
    "HybridWeights",
    "HybridTrack",
]