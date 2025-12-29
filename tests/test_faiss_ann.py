import numpy as np
import pytest

from musicrec.ann_faiss import FaissANN


def test_faiss_flat_cosine_self_neighbor():
    # If faiss isn't available in some envs, skip (keeps CI flexible)
    try:
        import faiss  # noqa: F401
    except Exception:
        pytest.skip("faiss not installed in this environment")

    rng = np.random.default_rng(42)
    X = rng.normal(size=(200, 32)).astype(np.float32)

    ann = FaissANN.build_flat_cosine(X)

    # query with an exact item vector -> nearest neighbor should be itself
    q_idx = 7
    q = X[q_idx]

    scores, idxs = ann.search(q, k=5)
    idxs = idxs[0].tolist()

    assert q_idx in idxs, f"Expected self index {q_idx} in top-5 neighbors, got {idxs}"


def test_faiss_dim_mismatch_raises():
    try:
        import faiss  # noqa: F401
    except Exception:
        pytest.skip("faiss not installed in this environment")

    X = np.random.normal(size=(50, 16)).astype(np.float32)
    ann = FaissANN.build_flat_cosine(X)

    bad_q = np.random.normal(size=(8,)).astype(np.float32)  # wrong dim
    with pytest.raises(ValueError):
        ann.search(bad_q, k=5)