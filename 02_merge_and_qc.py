#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Merge ABIDE connectome vectors with phenotypes; perform motion QC; report site retention.

Inputs
------
--connectomes : Parquet file from the build step (subjects x edges), index=FILE_ID
--phenotypes  : ABIDE phenotypes CSV or ZIP containing Phenotypic_V1_0b_preprocessed1.csv
--fd-thresh   : participant-level mean FD cutoff in mm (default 0.2)
--out         : output Parquet (post-QC). A pre-QC copy and a site_retention.csv are also written.

Example
-------
python 02_merge_and_qc.py \
  --connectomes ./derivatives/connectomes_cc200.parquet \
  --phenotypes ./Phenotypic_V1_0b_preprocessed1.zip \
  --fd-thresh 0.2 \
  --out ./derivatives/abide_cc200_merged_fd02.parquet
"""

import argparse, zipfile
from pathlib import Path
import pandas as pd
import numpy as np

def load_pheno(pheno_path: str) -> pd.DataFrame:
    p = Path(pheno_path)
    if p.suffix.lower() == ".zip":
        with zipfile.ZipFile(p) as zf, zf.open("Phenotypic_V1_0b_preprocessed1.csv") as f:
            pheno = pd.read_csv(f)
    else:
        pheno = pd.read_csv(p)

    # Standardize key fields
    pheno["FILE_ID"] = pheno["FILE_ID"].astype(str).str.strip()
    pheno["SITE_ID"] = pheno["SITE_ID"].astype(str).str.strip()
    pheno["SEX"] = pheno["SEX"].astype(str).str.strip()
    pheno["AGE_AT_SCAN"] = pd.to_numeric(pheno["AGE_AT_SCAN"], errors="coerce")
    pheno["func_mean_fd"] = pd.to_numeric(pheno["func_mean_fd"], errors="coerce")

    pheno = pheno.set_index("FILE_ID", drop=False)
    return pheno

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--connectomes", required=True)
    ap.add_argument("--phenotypes", required=True)
    ap.add_argument("--fd-thresh", type=float, default=0.2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    conn = pd.read_parquet(args.connectomes)
    if conn.index.name != "FILE_ID":
        # If index not saved, try to coerce
        conn.index.name = "FILE_ID"

    pheno = load_pheno(args.phenotypes)

    # Inner join: keep only subjects seen in both
    merged_pre = pheno.join(conn, how="inner")
    print(f"[info] Pre-QC merge: {merged_pre.shape[0]} subjects × {merged_pre.shape[1]} columns")

    # Site retention before QC
    pre_site = merged_pre.groupby("SITE_ID", dropna=False).size().rename("n_pre").to_frame()

    # Motion QC
    keep_mask = merged_pre["func_mean_fd"].notna() & (merged_pre["func_mean_fd"] <= args.fd_thresh)
    merged = merged_pre.loc[keep_mask].copy()
    print(f"[info] Post-QC (FD≤{args.fd_thresh:.2f} mm): {merged.shape[0]} subjects")

    post_site = merged.groupby("SITE_ID", dropna=False).size().rename("n_post").to_frame()
    site_tbl = pre_site.join(post_site, how="outer").fillna(0).astype(int)
    site_tbl["retention_%"] = (100.0 * site_tbl["n_post"] / site_tbl["n_pre"].clip(lower=1)).round(1)
    site_tbl = site_tbl.sort_values("n_pre", ascending=False)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Save both versions
    merged_pre.to_parquet(out.with_name(out.stem + "_preQC.parquet"), compression="zstd")
    merged.to_parquet(out, compression="zstd")
    site_tbl.to_csv(out.with_name("site_retention.csv"))

    # Quick summary
    fd_stats = merged["func_mean_fd"].describe()[["count","mean","std","min","25%","50%","75%","max"]]
    print("\n[info] Site retention (subjects):")
    print(site_tbl)
    print("\n[info] func_mean_fd summary (post-QC):")
    print(fd_stats.to_string())

if __name__ == "__main__":
    main()
