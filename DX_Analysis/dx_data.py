
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple
from dx_config import DXConfig

def load_phenotype(csv_path: str, cfg: DXConfig) -> pd.DataFrame:
    ph = pd.read_csv(csv_path)
    ph = ph[ph[cfg.dx_col].isin([1,2])].copy()
    ph["DX_BIN"] = (ph[cfg.dx_col] == 2).astype(int)
    ph["SexF"] = (ph[cfg.sex_col] == 2).astype(int)
    # Normalize SUB_ID to string sans leading zeros for robust matching
    ph["_SUB_STR"] = ph[cfg.sub_id_col].astype(str).str.lstrip("0")
    return ph

def load_features(x_path: str, subids_path: str) -> Tuple[np.ndarray, np.ndarray]:
    X = np.load(x_path)
    sub_ids = np.load(subids_path)
    if X.shape[0] != len(sub_ids):
        raise ValueError(f"X rows ({X.shape[0]}) != sub_ids length ({len(sub_ids)})")
    return X, sub_ids

def align_by_subject(ph: pd.DataFrame, sub_ids: np.ndarray, cfg: DXConfig) -> pd.DataFrame:
    # Coerce sub_ids to normalized string ids (strip leading zeros)
    sub_norm = pd.Series(sub_ids).astype(str).str.lstrip("0")
    ph = ph.set_index("_SUB_STR")
    missing = set(sub_norm) - set(ph.index)
    if missing:
        raise KeyError(f"{len(missing)} subject IDs missing in phenotype: e.g., {list(missing)[:5]}")
    out = ph.loc[sub_norm].reset_index(drop=True)
    return out

def qc_filter(ph: pd.DataFrame, cfg: DXConfig) -> np.ndarray:
    mask = (
        ph[cfg.age_col].notna() &
        ph[cfg.sex_col].notna() &
        ph[cfg.fd_col].notna() &
        (ph[cfg.fd_col] <= cfg.max_fd)
    )
    return mask.values
