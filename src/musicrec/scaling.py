from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class StandardScaler:
    mean_: np.ndarray
    scale_: np.ndarray

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.scale_


def fit_standard_scaler(X: np.ndarray, eps: float = 1e-12) -> StandardScaler:
    """
    Fit a standard scaler (z-score) on X.
    scale_ uses std with ddof=0; zeros are clamped to eps.
    """
    mean_ = X.mean(axis=0)
    scale_ = X.std(axis=0)
    scale_ = np.where(scale_ < eps, eps, scale_)
    return StandardScaler(mean_=mean_, scale_=scale_)


def scaler_to_jsonable(s: StandardScaler) -> Dict[str, object]:
    return {
        "mean": s.mean_.tolist(),
        "scale": s.scale_.tolist(),
    }


def scaler_from_jsonable(d: Dict[str, object]) -> StandardScaler:
    mean = np.array(d["mean"], dtype="float64")
    scale = np.array(d["scale"], dtype="float64")
    return StandardScaler(mean_=mean, scale_=scale)