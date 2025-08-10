#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse, os, json, time, random, math
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils import check_random_state

# ---------------------------
# Utilities
# ---------------------------

META_LIKE = {
    "SUB_ID","FILE_ID","SITE_ID","DX_GROUP","AGE_AT_SCAN","SEX",
    "func_mean_fd","func_num_fd_ge_0p2","func_num_fd_ge_0p3"
}

def set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)

def infer_feature_cols(df: pd.DataFrame):
    # Prefer explicit connectome naming if present
    cand = [c for c in df.columns if str(c).startswith(("edge_","e","conn_","corr_"))]
    if len(cand) >= 1000:
        return cand
    # Fallback: all float columns not obviously meta
    fcols = [c for c in df.columns
             if (c not in META_LIKE) and (pd.api.types.is_float_dtype(df[c]))]
    return fcols

def cap_per_site(df, site_col, max_per_site, seed=0):
    if max_per_site is None or max_per_site <= 0:
        return df
    rng = check_random_state(seed)
    parts = []
    for s, d in df.groupby(site_col):
        if len(d) > max_per_site:
            parts.append(d.sample(n=max_per_site, random_state=rng))
        else:
            parts.append(d)
    return pd.concat(parts, axis=0).sample(frac=1.0, random_state=rng).reset_index(drop=True)

def drop_all_nan_features(df, feat_cols):
    f = df[feat_cols]
    keep = f.columns[f.notna().any(axis=0)]
    dropped = len(f.columns) - len(keep)
    return list(keep), dropped

def build_transform(seed, n_comp):
    # Each fold trains its own imputer+scaler+pca (cached)
    imp = SimpleImputer(strategy="median")
    sca = StandardScaler(with_mean=True, with_std=True)
    pca = PCA(n_components=n_comp, svd_solver="randomized", random_state=seed)
    return imp, sca, pca

def fit_transform_chain(imp, sca, pca, X_tr, X_te):
    X_tr = X_tr.astype(np.float32, copy=False)
    X_te = X_te.astype(np.float32, copy=False)
    X_tr = imp.fit_transform(X_tr)
    X_tr = sca.fit_transform(X_tr)
    Z_tr = pca.fit_transform(X_tr)
    # apply to test
    X_te = imp.transform(X_te.astype(np.float32, copy=False))
    X_te = sca.transform(X_te)
    Z_te = pca.transform(X_te)
    return Z_tr, Z_te

def fit_clf_and_predict(Z_tr, y_tr, Z_te, seed):
    clf = LogisticRegression(
        solver="saga", penalty="l2", C=1.0,
        max_iter=10000, random_state=seed, n_jobs=None # keep default threads
    )
    clf.fit(Z_tr, y_tr)
    y_pred = clf.predict(Z_te)
    return y_pred

def scores(y_true, y_pred):
    return accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average="macro")

# ---------------------------
# LOSO core with caching
# ---------------------------

def loso_prepare_cached(df, feat_cols, label_col, site_col, pca_components, seed):
    """Prepare per-site cached transforms + transformed matrices for fast permutations."""
    set_seed(seed)
    X = df[feat_cols].to_numpy(dtype=np.float32, copy=False)
    y = df[label_col].to_numpy()
    sites = df[site_col].astype(str).to_numpy()

    # unique held-out sites
    site_list = sorted(np.unique(sites))
    cache = {}  # site -> dict(Z_tr,Z_te,y_tr,y_te, idx_tr, idx_te)
    for s in site_list:
        te_idx = np.where(sites == s)[0]
        tr_idx = np.where(sites != s)[0]
        # build transforms
        imp, sca, pca = build_transform(seed, pca_components)
        Z_tr, Z_te = fit_transform_chain(imp, sca, pca, X[tr_idx], X[te_idx])
        cache[s] = {
            "Z_tr": Z_tr, "Z_te": Z_te,
            "y_tr": y[tr_idx].copy(), "y_te": y[te_idx].copy()
        }
    return cache, site_list

def loso_once_from_cache(cache, site_list, seed):
    accs, f1s = [], []
    rng = check_random_state(seed)
    for s in site_list:
        Z_tr = cache[s]["Z_tr"]; y_tr = cache[s]["y_tr"]
        Z_te = cache[s]["Z_te"]; y_te = cache[s]["y_te"]
        # fresh classifier each fold
        y_pred = fit_clf_and_predict(Z_tr, y_tr, Z_te, seed=rng.randint(0, 2**31-1))
        a, f = scores(y_te, y_pred)
        accs.append(a); f1s.append(f)
    return float(np.mean(accs)), float(np.mean(f1s))

def loso_permute_from_cache(cache, site_list, n_perm, seed):
    rng = check_random_state(seed)
    accs, f1s = [], []
    for i in range(n_perm):
        acc_i, f1_i = [], []
        for s in site_list:
            Z_tr = cache[s]["Z_tr"]; y_tr = cache[s]["y_tr"].copy()
            Z_te = cache[s]["Z_te"]; y_te = cache[s]["y_te"]
            rng.shuffle(y_tr)  # permute training labels only
            y_pred = fit_clf_and_predict(Z_tr, y_tr, Z_te, seed=rng.randint(0, 2**31-1))
            a, f = scores(y_te, y_pred)
            acc_i.append(a); f1_i.append(f)
        accs.append(np.mean(acc_i)); f1s.append(np.mean(f1_i))
    return np.asarray(accs), np.asarray(f1s)

def permutation_pvalue(obs, null_dist):
    # upper-tail p-value with 1/(N+1) correction
    return float((1 + np.sum(null_dist >= obs)) / (len(null_dist) + 1))

# ---------------------------
# High-level run
# ---------------------------

def run_one(df, tag, outdir, args):
    print(f"[{tag}] features detected:", end=" ")
    feat_cols = infer_feature_cols(df)
    feat_cols, dropped = drop_all_nan_features(df, feat_cols)
    print(f"{len(feat_cols)} (dropped all-NaN: {dropped})")

    # Keep only rows with known label & site
    df = df.dropna(subset=[args.label_col, args.site_col]).copy()

    # Balance per-site for fairness/speed
    df = cap_per_site(df, args.site_col, args.max_per_site, seed=args.seed)

    # Cache transforms per held-out site (fit ONCE per site)
    t0 = time.time()
    cache, site_list = loso_prepare_cached(
        df, feat_cols, args.label_col, args.site_col,
        pca_components=args.pca, seed=args.seed
    )
    print(f"[{tag}] cached transforms for {len(site_list)} sites in {time.time()-t0:.1f}s")

    # Observed LOSO score
    obs_acc, obs_f1 = loso_once_from_cache(cache, site_list, seed=args.seed)
    print(f"[{tag}] LOSO  acc={obs_acc:.3f}  f1={obs_f1:.3f}")

    # Permutations (optional)
    p_acc = p_f1 = None
    if args.perm_n and args.perm_n > 0:
        t0 = time.time()
        null_acc, null_f1 = loso_permute_from_cache(cache, site_list, n_perm=args.perm_n, seed=args.seed+123)
        p_acc = permutation_pvalue(obs_acc, null_acc)
        p_f1  = permutation_pvalue(obs_f1,  null_f1)
        print(f"[{tag}] perm (n={args.perm_n})  p_acc={p_acc:.5f}  p_f1={p_f1:.5f}  ({time.time()-t0:.1f}s)")

    # Write JSON
    os.makedirs(outdir, exist_ok=True)
    out = {
        "tag": tag,
        "n_sites": len(site_list),
        "n_perm": int(args.perm_n or 0),
        "pca_components": int(args.pca),
        "max_per_site": int(args.max_per_site or 0),
        "metrics": {
            "acc": float(obs_acc),
            "f1_macro": float(obs_f1),
            "p_acc": None if p_acc is None else float(p_acc),
            "p_f1_macro": None if p_f1 is None else float(p_f1),
        }
    }
    with open(os.path.join(outdir, f"loso_{tag}.json"), "w") as f:
        json.dump(out, f, indent=2)
    return out

# ---------------------------
# CLI
# ---------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="ASD classification with LOSO by SITE_ID (fast permutations via cached PCA).")
    ap.add_argument("--raw", type=str, required=True, help="Merged parquet (raw)")
    ap.add_argument("--combat", type=str, required=False, help="ComBat parquet (optional)")
    ap.add_argument("--outdir", type=str, required=True)

    ap.add_argument("--label-col", type=str, default="DX_GROUP")
    ap.add_argument("--site-col",  type=str, default="SITE_ID")

    ap.add_argument("--pca", type=int, default=50)
    ap.add_argument("--max-per-site", type=int, default=80)

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--perm-n", type=int, default=0, help="Number of permutations (0 to skip)")

    return ap.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    # RAW
    print(f"[info] loading raw: {args.raw}")
    df_raw = pd.read_parquet(args.raw)
    run_one(df_raw, tag="raw", outdir=args.outdir, args=args)

    # ComBat (optional)
    if args.combat and os.path.exists(args.combat):
        print(f"[info] loading combat: {args.combat}")
        df_cb = pd.read_parquet(args.combat)
        run_one(df_cb, tag="combat", outdir=args.outdir, args=args)

if __name__ == "__main__":
    main()
