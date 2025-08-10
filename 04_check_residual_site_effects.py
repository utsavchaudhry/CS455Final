#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
04_check_residual_site_effects.py

Diagnostics for residual site effects before vs. after harmonization.

Adds:
  - Permutation tests with shared CV splits for speed
  - Separate PCA/downsampling knobs used only during permutations
  - Ability to select which models get permutations (default: knn,ridge,svc)
  - Avoids nested parallelism for LR during permutations

Outputs (by <tag> = raw | combat | covbat):
  site_pred_{model}_{tag}.json  # acc/f1 (+ p_acc, p_f1 and null arrays if permutations requested)
  silhouette_{tag}.json
  qcfct_{tag}.csv
"""

import argparse, json
from pathlib import Path
import numpy as np, pandas as pd

from scipy import stats
from statsmodels.stats.multitest import multipletests

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, silhouette_score


# ---------- IO ----------

def load_df(path):
    df = pd.read_parquet(path)
    feat_cols = [c for c in df.columns if c.startswith("e")]
    if "SITE_ID" not in df or "func_mean_fd" not in df:
        raise ValueError("Input parquet must contain SITE_ID and func_mean_fd columns.")
    return df, feat_cols


# ---------- Helpers ----------

def _balance_per_site(df, label, max_per_site=40, seed=0):
    """Cap samples per site to reduce imbalance & tiny-fold issues."""
    parts = []
    for _, block in df.groupby(label):
        if len(block) > max_per_site:
            parts.append(block.sample(n=max_per_site, random_state=seed))
        else:
            parts.append(block)
    out = pd.concat(parts, axis=0)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)

def _stratified_subsample(X, y, max_n=600, seed=0):
    """Subsample up to max_n rows, roughly balanced across labels."""
    n = X.shape[0]
    if n <= max_n:
        return X, y
    rng = np.random.default_rng(seed)
    uniq = np.unique(y)
    per_site = max(2, max_n // len(uniq))
    idx_take = []
    for s in uniq:
        idx = np.flatnonzero(y == s)
        k = min(per_site, len(idx))
        idx_take.extend(rng.choice(idx, size=k, replace=False))
    idx_take = np.array(sorted(idx_take))
    return X[idx_take], y[idx_take]

def _chance(n_sites: int) -> float:
    return 1.0 / max(1, n_sites)


# ---------- PCA preparation ----------

def _prep_Z(df, feat_cols, n_components, seed, max_per_site):
    dfb = _balance_per_site(df, "SITE_ID", max_per_site=max_per_site, seed=seed)
    y = LabelEncoder().fit_transform(dfb["SITE_ID"].astype(str).to_numpy())
    X = dfb[feat_cols].to_numpy(dtype=np.float32)
    Z = StandardScaler().fit_transform(X)
    Z = PCA(n_components=min(n_components, Z.shape[1]), random_state=seed).fit_transform(Z)
    return Z, y


# ---------- Cross-validated scorers given Z, y and (optional) precomputed splits ----------

def _cv_scores_lr(Z, y, splits=None, seed=0, n_splits=5, C=0.5, max_iter=10000, solver="saga", n_jobs=-1):
    if splits is None:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = list(skf.split(Z, y))
    accs, f1s, used = [], [], 0
    for tr, te in splits:
        if len(np.unique(y[tr])) < len(np.unique(y)):  # guard
            continue
        clf = LogisticRegression(
            solver=solver, penalty="l2", C=C,
            class_weight="balanced", max_iter=max_iter,
            n_jobs=n_jobs, random_state=seed
        )
        clf.fit(Z[tr], y[tr])
        yhat = clf.predict(Z[te])
        accs.append(accuracy_score(y[te], yhat))
        f1s.append(f1_score(y[te], yhat, average="macro"))
        used += 1
    if used == 0:
        raise RuntimeError("No valid CV folds.")
    return float(np.mean(accs)), float(np.std(accs)), float(np.mean(f1s)), float(np.std(f1s))

def _cv_scores_knn(Z, y, splits=None, seed=0, n_splits=5, k=5):
    if splits is None:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = list(skf.split(Z, y))
    accs, f1s = [], []
    for tr, te in splits:
        clf = KNeighborsClassifier(n_neighbors=k, weights="distance")
        clf.fit(Z[tr], y[tr])
        yhat = clf.predict(Z[te])
        accs.append(accuracy_score(y[te], yhat))
        f1s.append(f1_score(y[te], yhat, average="macro"))
    return float(np.mean(accs)), float(np.std(accs)), float(np.mean(f1s)), float(np.std(f1s))

def _cv_scores_ridge(Z, y, splits=None, seed=0, n_splits=5, alpha=1.0):
    if splits is None:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = list(skf.split(Z, y))
    accs, f1s = [], []
    for tr, te in splits:
        classes, counts = np.unique(y[tr], return_counts=True)
        inv_freq = {c: (len(y[tr]) / (len(classes) * cnt)) for c, cnt in zip(classes, counts)}
        w = np.array([inv_freq[c] for c in y[tr]], dtype=np.float64)
        clf = RidgeClassifier(alpha=alpha, random_state=seed)
        clf.fit(Z[tr], y[tr], sample_weight=w)
        yhat = clf.predict(Z[te])
        accs.append(accuracy_score(y[te], yhat))
        f1s.append(f1_score(y[te], yhat, average="macro"))
    return float(np.mean(accs)), float(np.std(accs)), float(np.mean(f1s)), float(np.std(f1s))

def _cv_scores_linsvc(Z, y, splits=None, seed=0, n_splits=5, C=0.5, max_iter=20000):
    if splits is None:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = list(skf.split(Z, y))
    accs, f1s = [], []
    for tr, te in splits:
        clf = LinearSVC(C=C, class_weight="balanced", dual=False,
                        max_iter=max_iter, random_state=seed)
        clf.fit(Z[tr], y[tr])
        yhat = clf.predict(Z[te])
        accs.append(accuracy_score(y[te], yhat))
        f1s.append(f1_score(y[te], yhat, average="macro"))
    return float(np.mean(accs)), float(np.std(accs)), float(np.mean(f1s)), float(np.std(f1s))


# ---------- Permutation test (label-shuffle), with shared CV splits ----------

def _perm_pvals(Z, y, scorer, splits, n_perm=0, seed=0):
    """
    scorer: function (Z, y, splits=...) -> (acc_mean, acc_std, f1_mean, f1_std)
    """
    acc_obs, acc_sd, f1_obs, f1_sd = scorer(Z, y, splits=splits)
    out = {"acc_mean": acc_obs, "acc_std": acc_sd,
           "f1_macro_mean": f1_obs, "f1_macro_std": f1_sd}

    if n_perm and n_perm > 0:
        rng = np.random.default_rng(seed)
        acc_null = np.empty(n_perm, dtype=float)
        f1_null  = np.empty(n_perm, dtype=float)
        for i in range(n_perm):
            y_perm = y.copy()
            rng.shuffle(y_perm)
            acc_i, _, f1_i, _ = scorer(Z, y_perm, splits=splits)
            acc_null[i] = acc_i
            f1_null[i]  = f1_i
        # upper-tailed p-value: is observed ≥ null?
        p_acc = (1.0 + np.sum(acc_null >= acc_obs)) / (n_perm + 1.0)
        p_f1  = (1.0 + np.sum(f1_null  >= f1_obs)) / (n_perm + 1.0)
        out.update({
            "p_acc": float(p_acc),
            "p_f1":  float(p_f1),
            "perm_acc": acc_null.tolist(),
            "perm_f1":  f1_null.tolist()
        })
    return out


# ---------- Silhouette (efficient) ----------

def silhouette_by_site(df, feat_cols, label="SITE_ID",
                       n_components=50, max_n=600, metric="euclidean", random_state=0) -> float:
    X = df[feat_cols].to_numpy(dtype=np.float32)
    y = df[label].astype("category").cat.codes.to_numpy()
    Z = StandardScaler().fit_transform(X)
    if n_components and n_components < Z.shape[1]:
        Z = PCA(n_components=n_components, random_state=random_state).fit_transform(Z)
    Z_sub, y_sub = _stratified_subsample(Z, y, max_n=max_n, seed=random_state)
    return float(silhouette_score(Z_sub, y_sub, metric=metric))


# ---------- QC–FC ----------

def qcfct(df, feat_cols):
    fd = pd.to_numeric(df["func_mean_fd"], errors="coerce").to_numpy()
    X = df[feat_cols].to_numpy(dtype=np.float32)
    r = np.zeros(X.shape[1], dtype=np.float64)
    p = np.zeros_like(r)
    for j in range(X.shape[1]):
        r[j], p[j] = stats.pearsonr(X[:, j], fd)
    _, q, _, _ = multipletests(p, alpha=0.05, method="fdr_bh")
    return pd.DataFrame({"edge": feat_cols, "r": r, "p": p, "q": q})


# ---------- Driver ----------

def compute_all(tag: str, df: pd.DataFrame, feats: list[str], out: Path, args):

    # Standard evaluation prep (used for the reported CV means/stds)
    Z_lr,   y_lr   = _prep_Z(df, feats, args.lr_pca_components,    args.seed, args.max_per_site)
    Z_knn,  y_knn  = _prep_Z(df, feats, args.knn_pca_components,   args.seed, args.max_per_site)
    Z_rdg,  y_rdg  = _prep_Z(df, feats, args.ridge_pca_components, args.seed, args.max_per_site)
    Z_lsvc, y_lsvc = _prep_Z(df, feats, args.linsvc_pca_components,args.seed, args.max_per_site)

    # Shared CV splits for each model (reused in permutations)
    skf_lr   = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    skf_knn  = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    skf_rdg  = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    skf_lsvc = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    splits_lr   = list(skf_lr.split(Z_lr, y_lr))
    splits_knn  = list(skf_knn.split(Z_knn, y_knn))
    splits_rdg  = list(skf_rdg.split(Z_rdg, y_rdg))
    splits_lsvc = list(skf_lsvc.split(Z_lsvc, y_lsvc))

    # Observed CV performance (fast)
    lr_obs   = _cv_scores_lr   (Z_lr,   y_lr,   splits_lr,   seed=args.seed, C=args.lr_C,     max_iter=args.lr_max_iter,    solver=args.lr_solver, n_jobs=-1)
    knn_obs  = _cv_scores_knn  (Z_knn,  y_knn,  splits_knn,  seed=args.seed, k=args.knn_k)
    rdg_obs  = _cv_scores_ridge(Z_rdg,  y_rdg,  splits_rdg,  seed=args.seed, alpha=args.ridge_alpha)
    lsvc_obs = _cv_scores_linsvc(Z_lsvc, y_lsvc, splits_lsvc, seed=args.seed, C=args.linsvc_C, max_iter=args.linsvc_max_iter)

    # Permutation prep (optionally smaller PCA and stronger downsampling just for speed)
    perm_models = set(m.strip().lower() for m in args.perm_models.split(",")) if args.perm_n > 0 else set()
    if args.perm_n > 0:
        Zp_lr,   yp_lr   = _prep_Z(df, feats, args.perm_pca_components, args.seed, args.perm_max_per_site)
        Zp_knn,  yp_knn  = _prep_Z(df, feats, args.perm_pca_components, args.seed, args.perm_max_per_site)
        Zp_rdg,  yp_rdg  = _prep_Z(df, feats, args.perm_pca_components, args.seed, args.perm_max_per_site)
        Zp_lsvc, yp_lsvc = _prep_Z(df, feats, args.perm_pca_components, args.seed, args.perm_max_per_site)
        # reuse same number of folds, but recompute splits for the perm-sized sets
        splitsp_lr   = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed).split(Zp_lr,   yp_lr))
        splitsp_knn  = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed).split(Zp_knn,  yp_knn))
        splitsp_rdg  = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed).split(Zp_rdg,  yp_rdg))
        splitsp_lsvc = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed).split(Zp_lsvc, yp_lsvc))
    else:
        Zp_lr=yp_lr=Zp_knn=yp_knn=Zp_rdg=yp_rdg=Zp_lsvc=yp_lsvc=None
        splitsp_lr=splitsp_knn=splitsp_rdg=splitsp_lsvc=None

    # Build result dicts; add p-values only for requested models
    def with_perm(tag_model, obs_tuple, do_perm, perm_fun):
        acc_mean, acc_std, f1_mean, f1_std = obs_tuple
        res = {"acc_mean": float(acc_mean), "acc_std": float(acc_std),
               "f1_macro_mean": float(f1_mean), "f1_macro_std": float(f1_std)}
        if args.perm_n > 0 and do_perm:
            res = perm_fun(res)  # add p-values & nulls
        return res

    # LR permutations (optional; heavy) – during permutations force n_jobs=1 to avoid nested parallelism
    lr_do_perm = ("lr" in perm_models)
    def _perm_lr(res):
        tmp = _perm_pvals(
            Zp_lr, yp_lr,
            lambda Z,Y, splits=None: _cv_scores_lr(Z, Y, splits, seed=args.seed, C=args.lr_C,
                                                   max_iter=args.lr_max_iter, solver=args.lr_solver,
                                                   n_jobs=1),  # <- critical for speed/stability
            splitsp_lr, n_perm=args.perm_n, seed=args.seed
        )
        return tmp
    lr_res = with_perm("lr", lr_obs, lr_do_perm, _perm_lr)

    # kNN permutations (fast)
    knn_do_perm = ("knn" in perm_models)
    def _perm_knn(res):
        return _perm_pvals(
            Zp_knn, yp_knn,
            lambda Z,Y, splits=None: _cv_scores_knn(Z, Y, splits, seed=args.seed, k=args.knn_k),
            splitsp_knn, n_perm=args.perm_n, seed=args.seed
        )
    knn_res = with_perm("knn", knn_obs, knn_do_perm, _perm_knn)

    # Ridge permutations (fast)
    rdg_do_perm = ("ridge" in perm_models)
    def _perm_rdg(res):
        return _perm_pvals(
            Zp_rdg, yp_rdg,
            lambda Z,Y, splits=None: _cv_scores_ridge(Z, Y, splits, seed=args.seed, alpha=args.ridge_alpha),
            splitsp_rdg, n_perm=args.perm_n, seed=args.seed
        )
    rdg_res = with_perm("ridge", rdg_obs, rdg_do_perm, _perm_rdg)

    # LinearSVC permutations (fast)
    lsvc_do_perm = ("svc" in perm_models or "linsvc" in perm_models)
    def _perm_lsvc(res):
        return _perm_pvals(
            Zp_lsvc, yp_lsvc,
            lambda Z,Y, splits=None: _cv_scores_linsvc(Z, Y, splits, seed=args.seed, C=args.linsvc_C, max_iter=args.linsvc_max_iter),
            splitsp_lsvc, n_perm=args.perm_n, seed=args.seed
        )
    lsvc_res = with_perm("linsvc", lsvc_obs, lsvc_do_perm, _perm_lsvc)

    # Print & save
    n_sites = int(len(np.unique(y_lr)))
    chance = _chance(n_sites)
    def _fmt(res):
        s = [f"acc={res['acc_mean']:.3f}", f"f1={res['f1_macro_mean']:.3f}"]
        if "p_acc" in res: s.append(f"p_acc={res['p_acc']:.3g}")
        if "p_f1"  in res: s.append(f"p_f1={res['p_f1']:.3g}")
        return "  ".join(s)

    print(f"[site-pred][{tag}] chance ≈ {chance:.3f} (1/{n_sites})")
    print(f"[LR+PCA    ][{tag}] {_fmt(lr_res)}")
    print(f"[kNN+PCA   ][{tag}] {_fmt(knn_res)}")
    print(f"[Ridge+PCA ][{tag}] {_fmt(rdg_res)}")
    print(f"[LinSVC+PCA][{tag}] {_fmt(lsvc_res)}")

    (out / f"site_pred_lr_pca_{tag}.json").write_text(json.dumps(lr_res,  indent=2))
    (out / f"site_pred_knn_pca_{tag}.json").write_text(json.dumps(knn_res, indent=2))
    (out / f"site_pred_ridge_pca_{tag}.json").write_text(json.dumps(rdg_res,indent=2))
    (out / f"site_pred_linsvc_pca_{tag}.json").write_text(json.dumps(lsvc_res,indent=2))

    # Silhouette
    s = silhouette_by_site(df, feats, n_components=args.sil_n_components,
                           max_n=args.sil_max_n, metric=args.sil_metric,
                           random_state=args.seed)
    (out / f"silhouette_{tag}.json").write_text(json.dumps({"silhouette": s}, indent=2))
    print(f"[silhouette][{tag}] = {s:.6f}")

    # QC–FC
    qcdf = qcfct(df, feats)
    qcdf.to_csv(out / f"qcfct_{tag}.csv", index=False)
    med_abs = float(np.median(np.abs(qcdf["r"].to_numpy())))
    pct_sig = float(np.mean(qcdf["q"].to_numpy() < 0.05) * 100.0)
    print(f"[QC–FC][{tag}] median |r|={med_abs:.3f}, %FDR<0.05={pct_sig:.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--combat", required=True)
    ap.add_argument("--covbat", default=None)
    ap.add_argument("--outdir", required=True)

    # silhouette knobs
    ap.add_argument("--sil-n-components", type=int, default=50)
    ap.add_argument("--sil-max-n", type=int, default=600)
    ap.add_argument("--sil-metric", type=str, default="euclidean")

    # PCA caps for models (observed CV)
    ap.add_argument("--lr-pca-components", type=int, default=50)
    ap.add_argument("--knn-pca-components", type=int, default=50)
    ap.add_argument("--ridge-pca-components", type=int, default=50)
    ap.add_argument("--linsvc-pca-components", type=int, default=50)

    # model hyperparams
    ap.add_argument("--knn-k", type=int, default=5)
    ap.add_argument("--max-per-site", type=int, default=40)

    ap.add_argument("--lr-C", type=float, default=0.5)
    ap.add_argument("--lr-max-iter", type=int, default=10000)
    ap.add_argument("--lr-solver", type=str, default="saga")

    ap.add_argument("--ridge-alpha", type=float, default=1.0)
    ap.add_argument("--linsvc-C", type=float, default=0.5)
    ap.add_argument("--linsvc-max-iter", type=int, default=20000)

    # permutation testing
    ap.add_argument("--perm-n", type=int, default=0, help="Number of label permutations per model (0 = off)")
    ap.add_argument("--perm-models", type=str, default="knn,ridge,svc",
                    help="Comma-separated subset of {lr,knn,ridge,svc} to permute")
    ap.add_argument("--perm-pca-components", type=int, default=30,
                    help="PCA components used during permutations (smaller = faster)")
    ap.add_argument("--perm-max-per-site", type=int, default=25,
                    help="Per-site cap used during permutations (smaller = faster)")

    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)

    df_raw, feats = load_df(args.raw)
    df_cb, feats2 = load_df(args.combat)
    if feats != feats2:
        raise ValueError("Feature sets differ between RAW and COMBAT files.")
    print(f"[info] features: {len(feats)}")

    compute_all("raw", df_raw, feats, out, args)
    compute_all("combat", df_cb, feats, out, args)

    if args.covbat:
        df_cov, feats3 = load_df(args.covbat)
        if feats != feats3:
            raise ValueError("Feature sets differ between RAW and COVBAT files.")
        compute_all("covbat", df_cov, feats, out, args)

if __name__ == "__main__":
    main()
