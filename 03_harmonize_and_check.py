#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Harmonize ABIDE connectomes across sites with ComBat (neuroHarmonize) and verify:

- PCA scatter pre vs post ComBat (color by SITE_ID)
- One-way ANOVA (feature ~ SITE_ID): median F-statistic drop

Inputs
------
--merged     : Parquet from step 02 (post-QC). Must include columns 'SITE_ID', covariates, and edge features (e*)
--out-prefix : Path prefix for outputs (adjusted parquet + PNGs + CSV)
--covars     : Biological covariates to preserve (default: AGE_AT_SCAN SEX)

Example
-------
python 03_harmonize_and_check.py \
  --merged ./derivatives/abide_cc200_merged_fd02.parquet \
  --out-prefix ./derivatives/abide_cc200 \
  --covars AGE_AT_SCAN SEX
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif
import matplotlib.pyplot as plt



def run_combat(df: pd.DataFrame, feat_cols: list[str], covar_cols: list[str]) -> pd.DataFrame:
    """
    Harmonize with neuroHarmonize if available; otherwise use neuroCombat.
    Ensures dtype compatibility with the float32 connectome columns.
    """
    import traceback
    import numpy as np
    import pandas as pd

    cov = df[["SITE_ID"] + covar_cols].copy()
    categorical_cols = [c for c in covar_cols if df[c].dtype == "object"]

    # Complete-case on covariates (required by ComBat)
    ok = cov.notna().all(axis=1)
    df_use = df.loc[ok].copy()
    cov_use = cov.loc[ok].copy()

    # features x subjects (ComBat API)
    X = df_use[feat_cols].to_numpy(dtype=float).T

    # --- Try neuroHarmonize first ------------------------------------------------
    try:
        
        # --- Try neuroHarmonize first (optional) ---
        from neuroHarmonize import harmonizationLearn  # correct public API
        cov_nh = df_use[["SITE_ID"] + covar_cols].rename(columns={"SITE_ID": "SITE"}).copy()

        # ensure numeric covariates (e.g., SEX -> {0,1})
        for c in covar_cols:
            cov_nh[c] = pd.to_numeric(cov_nh[c], errors="coerce")

        # neuroHarmonize expects N_samples x N_features
        X_samples_by_feat = df_use[feat_cols].to_numpy(dtype=float)  # (subjects x features)
        model, X_adj_samples_by_feat = harmonizationLearn(X_samples_by_feat, cov_nh)

        X_adj = X_adj_samples_by_feat.T.astype("float32")  # back to features x subjects for consistency


    except Exception as e:
        print("[warn] neuroHarmonize import/use failed:\n", repr(e))
        traceback.print_exc()
        print("[warn] falling back to neuroCombat…")

        # --- Fallback: neuroCombat returns a dict since v0.2.10 ------------------
        from neuroCombat import neuroCombat
        res = neuroCombat(
            dat=X, covars=cov_use, batch_col="SITE_ID", categorical_cols=categorical_cols
        )
        X_adj = res["data"] if isinstance(res, dict) else res  # <- Important per API

    # Cast to float32 before inserting into a float32-backed DataFrame block
    X_adj = np.asarray(X_adj, dtype=np.float32, order="C")  # (features x subjects)

    # Assign via a DataFrame to avoid block-wise dtype conflicts
    df_out = df_use.copy()
    df_out.loc[:, feat_cols] = pd.DataFrame(
        X_adj.T, index=df_use.index, columns=feat_cols, dtype=np.float32
    )
    return df_out




def pca_scatter(X: np.ndarray, sites: pd.Series, title: str, out_png: Path):
    Z = StandardScaler(with_mean=True, with_std=True).fit_transform(X)
    pcs = PCA(n_components=2, random_state=0).fit_transform(Z)

    fig, ax = plt.subplots(figsize=(6, 5))
    # Group by site to overlay points
    site_vals = sites.astype(str).values
    unique_sites = pd.unique(site_vals)
    for s in unique_sites:
        mask = (site_vals == s)
        ax.scatter(pcs[mask, 0], pcs[mask, 1], s=10, alpha=0.7, label=s)
    ax.set_title(title)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.legend(loc="best", fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

def one_way_site_anova(df: pd.DataFrame, feat_cols: list[str], site: pd.Series) -> pd.DataFrame:
    # Encode site as integer classes for f_classif
    y = site.astype("category").cat.codes.to_numpy()
    X = df[feat_cols].to_numpy()
    F, p = f_classif(X, y)
    return pd.DataFrame({"F": F, "p": p})

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", required=True)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--covars", nargs="*", default=["AGE_AT_SCAN", "SEX"])
    args = ap.parse_args()

    df = pd.read_parquet(args.merged)
    feat_cols = [c for c in df.columns if c.startswith("e")]
    if not feat_cols:
        raise SystemExit("No edge features (columns starting with 'e') found in merged file.")

    # Save baseline PCA/ANOVA
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    pca_scatter(df[feat_cols].to_numpy(), df["SITE_ID"], "PCA (raw)", out_prefix.with_name(out_prefix.stem + "_PCA_raw.png"))

    anova_raw = one_way_site_anova(df, feat_cols, df["SITE_ID"])
    anova_raw.to_csv(out_prefix.with_name(out_prefix.stem + "_anova_site_raw.csv"), index=False)
    print("[info] Median F (site) RAW:", float(np.median(anova_raw["F"])))

    # Run ComBat
    df_adj = run_combat(df, feat_cols, args.covars)
    df_adj.to_parquet(out_prefix.with_name(out_prefix.stem + "_combat.parquet"), compression="zstd")

    # Post-ComBat PCA/ANOVA
    pca_scatter(df_adj[feat_cols].to_numpy(), df_adj["SITE_ID"], "PCA (ComBat)", out_prefix.with_name(out_prefix.stem + "_PCA_combat.png"))

    anova_ad = one_way_site_anova(df_adj, feat_cols, df_adj["SITE_ID"])
    anova_ad.to_csv(out_prefix.with_name(out_prefix.stem + "_anova_site_combat.csv"), index=False)
    print("[info] Median F (site) ComBat:", float(np.median(anova_ad["F"])))

    # Simple summary of reduction
    delta = float(np.median(anova_raw["F"]) - np.median(anova_ad["F"]))
    print(f"[info] Median F reduction (site): {delta:.3f}")

if __name__ == "__main__":
    main()
