#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
07_export_report_assets.py

Collect paper-ready assets from the pipeline:
- demographics from merged parquet
- LOSO results (raw/combat) if JSON present
- copy selected figures into a report folder
- emit LaTeX table(s) + figure include snippets
- write a short markdown summary

Usage (typical):
  python 07_export_report_assets.py \
    --merged ./derivatives/abide_cc200_merged_fd02.parquet \
    --loso-dir ./derivatives/asd_loso \
    --fig-dir  ./derivatives/figures \
    --outdir   ./derivatives/report_assets

This script is read-only w.r.t. your analysis artifacts; it only *reads*
from derivatives and *writes* into `--outdir`.
"""

from __future__ import annotations
import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd


# -------------------------------
# Helpers
# -------------------------------

def _mk_outdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _safe_read_json(p: Path) -> Optional[Dict[str, Any]]:
    if not p.exists():
        return None
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _fmt(x: Optional[float], nd: int = 3) -> str:
    return ("-" if x is None else f"{x:.{nd}f}")

def _to_tex_safe(s: str) -> str:
    # minimal escaping for captions
    return (s.replace("_", r"\_")
             .replace("%", r"\%")
             .replace("&", r"\&"))

# -------------------------------
# Demographics
# -------------------------------

def compute_demographics(merged_parquet: Path) -> Dict[str, Any]:
    df = pd.read_parquet(merged_parquet)
    # Common column names in your pipeline: SITE_ID, AGE_AT_SCAN, SEX (0/1 or M/F)
    # Be defensive about SEX coding:
    sex_series = df["SEX"]
    if sex_series.dtype.kind in "iuf":  # numeric
        n_m = int((sex_series == 1).sum())  # if coded 1=male (ABIDE style)
        n_f = int((sex_series == 2).sum())  # if coded 2=female
        # if it's 0/1, try a heuristic: assume 1=male, 0=female
        if n_m + n_f == 0 and sex_series.dropna().isin([0, 1]).all():
            n_m = int((sex_series == 1).sum())
            n_f = int((sex_series == 0).sum())
    else:
        # string-like (e.g., 'M'/'F')
        n_m = int(sex_series.astype(str).str.upper().eq("M").sum())
        n_f = int(sex_series.astype(str).str.upper().eq("F").sum())

    age = df["AGE_AT_SCAN"].astype(float)
    info = {
        "N": int(len(df)),
        "n_sites": int(df["SITE_ID"].nunique()),
        "age_mean": float(age.mean()),
        "age_std": float(age.std(ddof=1)),
        "age_min": float(age.min()),
        "age_max": float(age.max()),
        "n_male": n_m,
        "n_female": n_f,
    }

    # Per-site table
    per_site = (
        df.groupby("SITE_ID")
          .agg(N=("SITE_ID", "size"),
               age_mean=("AGE_AT_SCAN", "mean"),
               age_sd=("AGE_AT_SCAN", "std"),
               n_male=("SEX", lambda x: int((pd.Series(x) == 1).sum()) if pd.Series(x).dtype.kind in "iuf" else int(pd.Series(x).astype(str).str.upper().eq("M").sum())),
               n_female=("SEX", lambda x: int((pd.Series(x) == 2).sum()) if pd.Series(x).dtype.kind in "iuf" else int(pd.Series(x).astype(str).str.upper().eq("F").sum()))
          )
          .reset_index()
          .sort_values("SITE_ID")
    )

    return {"overall": info, "by_site": per_site}


def write_demographics(outdir: Path, demog: Dict[str, Any]) -> None:
    # CSV(s)
    pd.DataFrame([demog["overall"]]).to_csv(outdir / "demographics_overall.csv", index=False)
    demog["by_site"].to_csv(outdir / "demographics_by_site.csv", index=False)

    # LaTeX table (by site; compact)
    df = demog["by_site"].copy()
    # Round a little for display
    for c in ("age_mean", "age_sd"):
        if c in df.columns:
            df[c] = df[c].astype(float).round(2)

    cols = ["SITE_ID", "N", "age_mean", "age_sd", "n_male", "n_female"]
    df = df[cols]

    latex = []
    latex.append(r"\begin{table}[t]")
    latex.append(r"\centering")
    latex.append(r"\setlength{\tabcolsep}{6pt}")
    latex.append(r"\begin{tabular}{lrrrrr}")
    latex.append(r"\toprule")
    latex.append(r"Site & $N$ & Age mean & Age SD & Male & Female \\")
    latex.append(r"\midrule")
    for _, r in df.iterrows():
        latex.append(f"{_to_tex_safe(str(r['SITE_ID']))} & {int(r['N'])} & {r['age_mean']:.2f} & {r['age_sd']:.2f} & {int(r['n_male'])} & {int(r['n_female'])} \\\\")
    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\caption{ABIDE demographics by site after FD$\leq$0.2 filter.}")
    latex.append(r"\label{tab:demographics_by_site}")
    latex.append(r"\end{table}")

    (outdir / "table_demographics_by_site.tex").write_text("\n".join(latex), encoding="utf-8")

# -------------------------------
# LOSO results -> LaTeX table
# -------------------------------

def load_loso(loso_dir: Path) -> Dict[str, Any]:
    raw = _safe_read_json(loso_dir / "loso_raw.json")
    combat = _safe_read_json(loso_dir / "loso_combat.json")
    return {"raw": raw, "combat": combat}

def write_loso_table(outdir: Path, loso: Dict[str, Any]) -> None:
    raw = loso.get("raw") or {}
    com = loso.get("combat") or {}

    # Expecting keys like: acc, f1, p_acc, p_f1
    # Be defensive:
    def g(d: Dict[str, Any], k: str) -> Optional[float]:
        v = d.get(k)
        try:
            return float(v) if v is not None else None
        except Exception:
            return None

    rows = [
        ("Raw",    g(raw, "acc"), g(raw, "f1"), g(raw, "p_acc"), g(raw, "p_f1")),
        ("ComBat", g(com, "acc"), g(com, "f1"), g(com, "p_acc"), g(com, "p_f1")),
    ]

    latex = []
    latex.append(r"\begin{table}[t]")
    latex.append(r"\centering")
    latex.append(r"\setlength{\tabcolsep}{6pt}")
    latex.append(r"\begin{tabular}{lrrrr}")
    latex.append(r"\toprule")
    latex.append(r"Condition & Acc & F1 & $p_{\mathrm{acc}}$ & $p_{\mathrm{F1}}$ \\")
    latex.append(r"\midrule")
    for name, acc, f1, pacc, pf1 in rows:
        latex.append(f"{name} & {_fmt(acc)} & {_fmt(f1)} & {_fmt(pacc, 5)} & {_fmt(pf1, 5)} \\\\")
    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\caption{Leave-one-site-out (LOSO) ASD vs. control classification. $p$-values from label permutations ($n$ as configured).}")
    latex.append(r"\label{tab:loso}")
    latex.append(r"\end{table}")

    (outdir / "table_loso.tex").write_text("\n".join(latex), encoding="utf-8")

# -------------------------------
# Figures: copy & include snippets
# -------------------------------

DEFAULT_FIGS = [
    "site_acc_raw_vs_combat.png",
    "site_f1_raw_vs_combat.png",
    "silhouette_raw_vs_combat.png",
    "qcfc_abs_r_hist_raw.png",
    "qcfc_abs_r_hist_combat.png",
    "perm_knn_pca_raw_acc.png",
    "perm_knn_pca_combat_acc.png",
    "pca_scatter_raw.png",
    "pca_scatter_ComBat.png",
]

CAPTIONS = {
    "site_acc_raw_vs_combat.png": "Site-prediction accuracy drops sharply after ComBat harmonization.",
    "site_f1_raw_vs_combat.png":  "Macro-F1 for site prediction before/after harmonization.",
    "silhouette_raw_vs_combat.png": "Silhouette by site (lower is better; negative values indicate weak clustering by site).",
    "qcfc_abs_r_hist_raw.png": "QC–FC absolute correlations (raw).",
    "qcfc_abs_r_hist_combat.png": "QC–FC absolute correlations (ComBat).",
    "perm_knn_pca_raw_acc.png": "Permutation distribution of kNN+PCA site-accuracy (raw).",
    "perm_knn_pca_combat_acc.png": "Permutation distribution of kNN+PCA site-accuracy (ComBat).",
    "pca_scatter_raw.png": "PCA scatter (raw edges), colored by site.",
    "pca_scatter_ComBat.png": "PCA scatter (ComBat edges), colored by site.",
}

def copy_figures(fig_dir: Path, outdir: Path, names: List[str]) -> List[Path]:
    out_fig = outdir / "figures"
    _mk_outdir(out_fig)
    copied = []
    for n in names:
        src = fig_dir / n
        if src.exists():
            dst = out_fig / n
            shutil.copy2(src, dst)
            copied.append(dst)
    return copied

def write_fig_includes(outdir: Path, copied: List[Path]) -> None:
    # One figure per file, width=\linewidth
    lines = []
    for p in copied:
        name = p.name
        cap = _to_tex_safe(CAPTIONS.get(name, name))
        lines.append(r"\begin{figure}[t]")
        lines.append(r"\centering")
        lines.append(fr"\includegraphics[width=\linewidth]{{figures/{_to_tex_safe(name)}}}")
        lines.append(fr"\caption{{{cap}}}")
        label = name.replace(".png", "").replace(".", "_").replace("-", "_")
        lines.append(fr"\label{{fig:{label}}}")
        lines.append(r"\end{figure}")
        lines.append("")  # blank line
    (outdir / "fig_includes.tex").write_text("\n".join(lines), encoding="utf-8")

# -------------------------------
# Markdown summary
# -------------------------------

def write_summary_md(outdir: Path, demog: Dict[str, Any], loso: Dict[str, Any]) -> None:
    o = demog["overall"]
    raw = loso.get("raw") or {}
    com = loso.get("combat") or {}
    md = []
    md.append("# Report Assets Summary")
    md.append("")
    md.append("## Demographics (post-QC)")
    md.append(f"- N = **{o['N']}** subjects across **{o['n_sites']}** sites")
    md.append(f"- Age: mean **{o['age_mean']:.2f}**, SD **{o['age_std']:.2f}**, range **{o['age_min']:.2f}–{o['age_max']:.2f}**")
    md.append(f"- Sex: **{o['n_male']} male**, **{o['n_female']} female**")
    md.append("")
    if raw:
        md.append("## LOSO classification")
        md.append(f"- RAW:    acc **{_fmt(raw.get('acc'))}**, F1 **{_fmt(raw.get('f1'))}**")
    if com:
        md.append(f"- ComBat: acc **{_fmt(com.get('acc'))}**, F1 **{_fmt(com.get('f1'))}**")
        if com.get("p_acc") is not None or com.get("p_f1") is not None:
            md.append(f"  (p_acc **{_fmt(com.get('p_acc'),5)}**, p_f1 **{_fmt(com.get('p_f1'),5)}**)")
    md.append("")
    md.append("See LaTeX snippets: `table_demographics_by_site.tex`, `table_loso.tex`, and figure includes in `fig_includes.tex`.")
    (outdir / "summary.md").write_text("\n".join(md), encoding="utf-8")

# -------------------------------
# Main
# -------------------------------

def main():
    ap = argparse.ArgumentParser(description="Export paper-ready assets (tables, figs, snippets).")
    ap.add_argument("--merged", type=Path, required=True, help="Merged parquet after QC (e.g., derivatives/abide_cc200_merged_fd02.parquet)")
    ap.add_argument("--loso-dir", type=Path, default=Path("./derivatives/asd_loso"), help="Directory containing loso_raw.json / loso_combat.json")
    ap.add_argument("--fig-dir",  type=Path, default=Path("./derivatives/figures"), help="Directory of generated PNG figures")
    ap.add_argument("--outdir",   type=Path, default=Path("./derivatives/report_assets"), help="Output directory for report assets")
    ap.add_argument("--figs", nargs="*", default=DEFAULT_FIGS, help="Specific figure filenames to copy/include")
    args = ap.parse_args()

    _mk_outdir(args.outdir)

    # Demographics
    print("[info] computing demographics…")
    demog = compute_demographics(args.merged)
    write_demographics(args.outdir, demog)

    # LOSO results
    print("[info] reading LOSO results…")
    loso = load_loso(args.loso_dir)
    if (loso.get("raw") is not None) or (loso.get("combat") is not None):
        write_loso_table(args.outdir, loso)
    else:
        print("[warn] no LOSO JSON found; skipping table_loso.tex")

    # Figures
    print("[info] copying figures…")
    copied = copy_figures(args.fig_dir, args.outdir, args.figs)
    if copied:
        write_fig_includes(args.outdir, copied)
    else:
        print("[warn] no figures copied (check --fig-dir and names)")

    # Markdown summary
    write_summary_md(args.outdir, demog, loso)

    print("\n[done] Assets written to:", str(args.outdir))
    print("      - demographics_overall.csv, demographics_by_site.csv")
    print("      - table_demographics_by_site.tex, table_loso.tex (if LOSO present)")
    print("      - figures/ (copied PNGs) + fig_includes.tex")
    print("      - summary.md")


if __name__ == "__main__":
    main()
