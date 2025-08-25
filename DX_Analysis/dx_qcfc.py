from __future__ import annotations
import numpy as np, pandas as pd
from typing import Dict
from scipy.stats import spearmanr
from dx_config import DXConfig

def qcfc_by_group(X: np.ndarray, ph: pd.DataFrame, cfg: DXConfig) -> Dict[str, float]:
    """Correlate each edge with FD within ASD and Control separately; return med|r| and FDR %."""
    out = {}
    for label, name in [(1, "ASD"), (0, "Control")]:
        idx = ph["DX_BIN"].values == label
        if idx.sum() < 10:
            out[f"{name}_median_abs_r"] = np.nan
            out[f"{name}_fdr_pct"] = np.nan
            continue
        r = np.zeros(X.shape[1], dtype=float)
        p = np.ones(X.shape[1], dtype=float)
        fd = ph.loc[idx, cfg.fd_col].values
        Xm = X[idx]
        for j in range(X.shape[1]):
            rr, pp = spearmanr(Xm[:, j], fd, nan_policy='omit')
            r[j], p[j] = (rr if np.isfinite(rr) else 0.0), (pp if np.isfinite(pp) else 1.0)
        # Benjamini–Hochberg within group
        from statsmodels.stats.multitest import multipletests
        rej, q, *_ = multipletests(p, alpha=0.05, method='fdr_bh')
        out[f"{name}_median_abs_r"] = float(np.nanmedian(np.abs(r)))
        out[f"{name}_fdr_pct"] = float(100.0 * rej.mean())
    return out
