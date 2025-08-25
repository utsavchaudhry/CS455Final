from __future__ import annotations
import numpy as np
import pandas as pd
from dx_config import DXConfig

def combat_harmonize_preserve_dx(X: np.ndarray, ph: pd.DataFrame, cfg: DXConfig) -> np.ndarray:
    """
    ComBat using the maintained 'neuroCombat' (no sklearn dependency).
    - Batch: SITE_ID
    - Covariates to preserve: Age (continuous), SexF (binary), DX_BIN (binary)
    Returns an array with the same shape as X (subjects × edges).
    """
    try:
        from neuroCombat import neuroCombat

        # Build covariate DataFrame (N subjects × K covariates)
        covars = pd.DataFrame({
            "batch": ph[cfg.site_col].astype(str).values,   # site label (categorical)
            "Age":   pd.to_numeric(ph[cfg.age_col], errors="coerce").fillna(0.0).values,
            "SexF":  pd.to_numeric(ph["SexF"], errors="coerce").fillna(0.0).values,
            "DX_BIN":pd.to_numeric(ph["DX_BIN"], errors="coerce").fillna(0.0).values,
        })

        # Features must be (features × samples). Our X is (samples × edges).
        dat = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).T  # (P × N)

        # SexF and DX_BIN are categorical; Age is continuous; 'batch' is the batch column name.
        out = neuroCombat(
            dat=dat,
            covars=covars,
            batch_col="batch",
            categorical_cols=["SexF", "DX_BIN"],   # including DX preserves its effect
            parametric=True,
            mean_only=False,
        )
        Xc = out["data"].T  # back to (N × P)
        return Xc

    except Exception as e:
        print("[WARN] ComBat not available or failed (", e, ") → proceeding without harmonization.")
        return X.copy()
