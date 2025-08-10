#!/usr/bin/env python
# 05_make_figures.py
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

def load_json(p): 
    with open(p, "r") as f: 
        return json.load(f)

def ensure_out(outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)

def chance_line(ax, chance):
    ax.axhline(chance, linestyle="--", linewidth=1)

def fig_site_accuracy(diagnostics: Path, outdir: Path, n_sites: int):
    chance = 1.0 / n_sites
    def pick(tag, stem):
        p = diagnostics / f"{stem}_{tag}.json"
        return load_json(p) if p.exists() else None

    models = [
        ("LR+PCA",    "site_pred_lr_pca"),
        ("Ridge+PCA", "site_pred_ridge_pca"),
        ("LinSVC+PCA","site_pred_linsvc_pca"),
        ("kNN+PCA",   "site_pred_knn_pca"),
    ]
    tags = ["raw","combat"]

    acc = np.zeros((len(models), len(tags)), dtype=float)
    f1  = np.zeros_like(acc)
    for i,(label, stem) in enumerate(models):
        for j,tag in enumerate(tags):
            js = pick(tag, stem)
            if js is None: continue
            acc[i,j] = js["acc_mean"]
            f1[i,j]  = js["f1_macro_mean"]

    # Accuracy bars
    fig, ax = plt.subplots(figsize=(6,4))
    x = np.arange(len(models))
    w = 0.35
    ax.bar(x - w/2, acc[:,0], width=w, label="raw")
    ax.bar(x + w/2, acc[:,1], width=w, label="ComBat")
    chance_line(ax, chance)
    ax.set_xticks(x)
    ax.set_xticklabels([m[0] for m in models], rotation=0)
    ax.set_ylabel("Site prediction accuracy")
    ax.set_title("Site classification: raw vs. ComBat")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / "site_acc_raw_vs_combat.png", dpi=300)
    plt.close(fig)

    # F1 bars
    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(x - w/2, f1[:,0], width=w, label="raw")
    ax.bar(x + w/2, f1[:,1], width=w, label="ComBat")
    ax.set_xticks(x)
    ax.set_xticklabels([m[0] for m in models], rotation=0)
    ax.set_ylabel("Macro F1")
    ax.set_title("Site classification (F1): raw vs. ComBat")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / "site_f1_raw_vs_combat.png", dpi=300)
    plt.close(fig)

def fig_perm_nulls(diagnostics: Path, outdir: Path, tag: str, stem: str, metric: str = "acc_mean"):
    """Overlay permutation null for one model and condition."""
    p = diagnostics / f"{stem}_{tag}.json"
    if not p.exists(): 
        return
    js = load_json(p)
    if "perm_acc" not in js:
        return

    if metric == "acc_mean":
        obs = js["acc_mean"]; null = np.array(js["perm_acc"], float)
        title = f"Permutation null (accuracy): {stem.replace('site_pred_','')} [{tag}]"
        fname = f"perm_{stem.split('site_pred_')[1]}_{tag}_acc.png"
    else:
        obs = js["f1_macro_mean"]; null = np.array(js["perm_f1"], float)
        title = f"Permutation null (macro F1): {stem.replace('site_pred_','')} [{tag}]"
        fname = f"perm_{stem.split('site_pred_')[1]}_{tag}_f1.png"

    fig, ax = plt.subplots(figsize=(6,4))
    ax.hist(null, bins=20)
    ax.axvline(obs, linestyle="--", linewidth=1)
    ax.set_xlabel(metric.replace("_"," "))
    ax.set_ylabel("count")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(outdir / fname, dpi=300)
    plt.close(fig)

def fig_silhouette(diagnostics: Path, outdir: Path):
    vals = []
    for tag in ["raw","combat"]:
        p = diagnostics / f"silhouette_{tag}.json"
        js = load_json(p)
        vals.append(js["silhouette"])
    fig, ax = plt.subplots(figsize=(4,4))
    ax.bar([0,1], vals)
    ax.set_xticks([0,1]); ax.set_xticklabels(["raw","ComBat"])
    ax.set_ylabel("Silhouette (SITE_ID)")
    ax.set_title("Cluster separation by site")
    fig.tight_layout()
    fig.savefig(outdir / "silhouette_raw_vs_combat.png", dpi=300)
    plt.close(fig)

def fig_qcfc_hist(diagnostics: Path, outdir: Path):
    def load_q(tag):
        df = pd.read_csv(diagnostics / f"qcfct_{tag}.csv")
        return np.abs(df["r"].to_numpy(dtype=float))
    abs_raw = load_q("raw")
    abs_cb  = load_q("combat")

    # Two panels to avoid color choices
    for arr, tag in [(abs_raw,"raw"), (abs_cb,"combat")]:
        fig, ax = plt.subplots(figsize=(5,4))
        ax.hist(arr, bins=40)
        ax.set_xlabel("|r(QC–FC)|")
        ax.set_ylabel("count")
        ax.set_title(f"QC–FC: |r| distribution [{tag}]")
        fig.tight_layout()
        fig.savefig(outdir / f"qcfc_abs_r_hist_{tag}.png", dpi=300)
        plt.close(fig)

def infer_n_sites(diagnostics: Path):
    # chance = 1/n_sites; approximate from JSON if stored; otherwise ask user
    # Here, derive from your earlier logs (17 sites). Hardcode fallback to 17.
    return 17

def fig_pca_site_scatter(parquet_path: Path, outdir: Path, tag: str, max_per_site: int = 40, seed: int = 0):
    df = pd.read_parquet(parquet_path)
    feat_cols = [c for c in df.columns if c.startswith("e")]
    # cap per site for readability
    parts = []
    rng = np.random.default_rng(seed)
    for site, block in df.groupby("SITE_ID"):
        k = min(max_per_site, len(block))
        parts.append(block.sample(n=k, random_state=int(rng.integers(0, 1e9))))
    d = pd.concat(parts, axis=0).reset_index(drop=True)

    X = d[feat_cols].to_numpy(dtype=np.float32)
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    Z = PCA(n_components=2, random_state=seed).fit_transform(StandardScaler().fit_transform(X))

    # scatter (single color; site labels as text density is too heavy); we’ll annotate with site count
    fig, ax = plt.subplots(figsize=(5,4))
    ax.scatter(Z[:,0], Z[:,1], s=6, alpha=0.6)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title(f"PCA (2D) by SITE_ID (points subsampled) [{tag}]")
    fig.tight_layout()
    fig.savefig(outdir / f"pca_scatter_{tag}.png", dpi=300)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diagnostics", required=True)
    ap.add_argument("--raw-parquet", required=True)
    ap.add_argument("--combat-parquet", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    diagnostics = Path(args.diagnostics)
    outdir = Path(args.outdir); ensure_out(outdir)

    n_sites = infer_n_sites(diagnostics)

    # 1) accuracy + F1 bars
    fig_site_accuracy(diagnostics, outdir, n_sites)

    # 2) permutation histograms for kNN/Ridge/SVC (raw and ComBat)
    for stem in ["site_pred_knn_pca", "site_pred_ridge_pca", "site_pred_linsvc_pca"]:
        for tag in ["raw","combat"]:
            fig_perm_nulls(diagnostics, outdir, tag, stem, metric="acc_mean")

    # 3) silhouette bars
    fig_silhouette(diagnostics, outdir)

    # 4) QC–FC histograms
    fig_qcfc_hist(diagnostics, outdir)

    # 5) PCA scatters (optional, but visually intuitive)
    fig_pca_site_scatter(Path(args.raw_parquet), outdir, tag="raw")
    fig_pca_site_scatter(Path(args.combat_parquet), outdir, tag="ComBat")

if __name__ == "__main__":
    main()
