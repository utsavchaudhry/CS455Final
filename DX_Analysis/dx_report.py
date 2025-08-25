from __future__ import annotations
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Tuple
from dx_config import DXConfig

def save_volcano(beta: np.ndarray, qvals: np.ndarray, out_png: str):
    x = beta
    y = -np.log10(np.clip(qvals, 1e-300, None))
    plt.figure()
    plt.scatter(x, y, s=4)  # default color; single-plot per figure
    plt.xlabel(r"$\hat{\beta}_{DX}$")
    plt.ylabel(r"$-\log_{10}(q)$")
    plt.title("Volcano plot: DX effect per edge")
    plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close()

def save_histogram(vals: np.ndarray, out_png: str, title: str):
    plt.figure()
    plt.hist(vals, bins=60)
    plt.xlabel("Value"); plt.ylabel("Count"); plt.title(title)
    plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close()

def write_table(df: pd.DataFrame, out_csv: str):
    df.to_csv(out_csv, index=False)

def cohort_table(ph: pd.DataFrame, cfg: DXConfig) -> pd.DataFrame:
    grp = ph.groupby([cfg.site_col, "DX_BIN"]).size().rename("n").reset_index()
    return grp.pivot(index=cfg.site_col, columns="DX_BIN", values="n").fillna(0).astype(int).reset_index().rename(columns={0:"Control",1:"ASD"})
