
from __future__ import annotations
import os, re, glob
import numpy as np
from typing import List, Tuple

def _extract_subject_id(fname: str) -> str:
    base = os.path.basename(fname)
    m = re.search(r"_(\d+)_rois(?:_|\.|$)", base)
    if m:
        return m.group(1)
    m2 = list(re.finditer(r"(\d+)(?=[^0-9]*$)", base))
    if m2:
        return m2[-1].group(1)
    return os.path.splitext(base)[0]

def _read_1d(path: str) -> np.ndarray:
    return np.loadtxt(path)

def _robust_corrcoef(ts: np.ndarray) -> np.ndarray:
    # Center
    X = ts - np.nanmean(ts, axis=0, keepdims=True)
    # Sample std (ddof=1); mark zero-variance as NaN to avoid divide-by-zero
    sd = np.nanstd(X, axis=0, ddof=1)
    sd[sd == 0] = np.nan
    # Sample covariance
    n = X.shape[0]
    C = (X.T @ X) / max(n - 1, 1)
    denom = np.outer(sd, sd)
    R = C / denom
    # Diagonal exactly 1.0
    np.fill_diagonal(R, 1.0)
    # Replace NaN/Inf with 0 (no correlation if undefined)
    R = np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)
    return R

def _fisher_z(r: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    r = np.clip(r, -1 + eps, 1 - eps)
    return np.arctanh(r)

def _vec_upper(mat: np.ndarray) -> np.ndarray:
    iu = np.triu_indices_from(mat, k=1)
    return mat[iu]

def build_features_from_1d(roi_dir: str, pattern: str="*.1D", fisher: bool=True
                          ) -> Tuple[np.ndarray, np.ndarray]:
    files = sorted(glob.glob(os.path.join(roi_dir, pattern)))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern} under {roi_dir}")
    X_rows = []
    sub_ids = []
    n_roi_ref = None
    for fp in files:
        ts = _read_1d(fp)
        if ts.ndim != 2:
            raise ValueError(f"{fp} did not load as 2D array")
        if n_roi_ref is None:
            n_roi_ref = ts.shape[1]
        elif ts.shape[1] != n_roi_ref:
            raise ValueError(f"ROI count mismatch: expected {n_roi_ref}, got {ts.shape[1]} in {fp}")
        R = _robust_corrcoef(ts)
        Z = _fisher_z(R) if fisher else R
        feat = _vec_upper(Z)
        # Final safety net: no NaN/Inf in features
        feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
        X_rows.append(feat)
        sub_ids.append(_extract_subject_id(fp))
    X = np.vstack(X_rows)
    return X, np.array(sub_ids, dtype=object)
