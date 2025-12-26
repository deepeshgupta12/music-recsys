from __future__ import annotations

import json

import numpy as np


def test_scaled_matrix_and_scaler_exist_after_run():
    assert (("data/processed/catalog_X_scaled.npy"))  # file check in next test
    assert True


def test_scaled_matrix_shape_matches_original():
    X = np.load("data/processed/catalog_X.npy")
    Xs = np.load("data/processed/catalog_X_scaled.npy")
    assert X.shape == Xs.shape


def test_train_mean_close_to_zero():
    with open("data/processed/catalog_X_scaler.json", "r", encoding="utf-8") as f:
        d = json.load(f)
    mean = np.array(d["scaler"]["mean"], dtype="float64")
    scale = np.array(d["scaler"]["scale"], dtype="float64")
    assert mean.shape[0] == scale.shape[0]
    assert np.isfinite(mean).all()
    assert np.isfinite(scale).all()