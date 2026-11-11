"""Ringer data loading, v2 ring slicing, L1 normalization and SP index.

Mirrors cnn-optimization (commit f05e91a): src/metrics.py (slice_rings_by_version, norm1,
compute_sp_with_threshold) and the data cell of notebooks/01_quantized_cnn1d_optimization.ipynb
(balancing by truncation to the minority class, 20 % stratified holdout, seed 42).
"""

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split

ESTRUTURA_BASE: Dict[str, int] = {"PS": 8, "EM1": 64, "EM2": 8, "EM3": 8, "HAD1": 4, "HAD2": 4, "HAD3": 4}
V2_FACTOR = 4  # v2 keeps the first 1/4 rings of each sampling layer -> 25 rings


def slice_rings_v2(matriz_v0: np.ndarray) -> np.ndarray:
    idx, ptr = [], 0
    for n in ESTRUTURA_BASE.values():
        idx.extend(range(ptr, ptr + n // V2_FACTOR))
        ptr += n
    return matriz_v0[:, idx]


def norm1(data: np.ndarray) -> np.ndarray:
    norms = np.abs(data.sum(axis=1, keepdims=True))
    norms[norms == 0] = 1.0
    return data / norms


def load_ringer_v2(h5_path: str, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns X_dev, X_holdout, y_dev, y_holdout exactly as in the cnn-optimization notebook."""
    df = pd.read_hdf(h5_path, key="data")
    ring_cols = sorted([c for c in df.columns if c.startswith("trig_L2_cl_ring_")], key=lambda c: int(c.split("_")[-1]))
    sig = df[df["target"] == 1][ring_cols].values
    bkg = df[df["target"] == 0][ring_cols].values
    ns = min(len(sig), len(bkg))
    sig_n = norm1(slice_rings_v2(sig[:ns]))
    bkg_n = norm1(slice_rings_v2(bkg[:ns]))
    X = np.vstack([sig_n, bkg_n]).astype(np.float32)
    y = np.concatenate([np.ones(ns, dtype=np.float32), np.zeros(ns, dtype=np.float32)])
    return train_test_split(X, y, test_size=0.20, stratify=y, random_state=seed)


def compute_sp_with_threshold(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float, float]:
    """SP = sqrt( sqrt(PD (1-FA)) * (PD + 1 - FA)/2 ) at the ROC knee. Returns (sp, pd, fa, threshold)."""
    fa, pd_, thr = roc_curve(y_true, y_pred)
    sp = np.sqrt(np.sqrt(np.clip(pd_ * (1.0 - fa), 0.0, 1.0)) * 0.5 * (pd_ + 1.0 - fa))
    k = int(np.nanargmax(sp))
    return float(sp[k]), float(pd_[k]), float(fa[k]), float(thr[k])


def sp_at_threshold(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> Tuple[float, float, float]:
    """SP / PD / FA at a fixed decision threshold (no re-optimization on the evaluated set)."""
    sel = y_pred >= threshold
    pd_ = float(np.mean(sel[y_true == 1]))
    fa = float(np.mean(sel[y_true == 0]))
    sp = float(np.sqrt(np.sqrt(pd_ * (1.0 - fa)) * 0.5 * (pd_ + 1.0 - fa)))
    return sp, pd_, fa
