#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build Fisher-z connectomes from AFNI .1D ROI time-series with two strategies:
  - fixed:     keep all ROIs; neutralize invalid ROIs per subject → constant length
  - prevalence:keep ROIs valid in ≥ prevalence threshold across subjects; optional subject QC

Outputs:
  - Parquet: subjects × edges connectome matrix
  - CSV QC:  per-subject QC table (TRs, invalid-ROI counts/fractions)
  - (prevalence) NPY mask with kept ROI indices

Usage (CC200, fixed-length):
  python build_connectomes.py --roi-root ./rois_cc200 --atlas cc200 \
         --strategy fixed --out ./derivatives/connectomes_cc200.parquet

Usage (CC200, prevalence 95%, drop subjects with >25% bad kept-ROIs):
  python build_connectomes.py --roi-root ./rois_cc200 --atlas cc200 \
         --strategy prevalence --roi-prevalence 0.95 --max-bad-roi-frac 0.25 \
         --out ./derivatives/connectomes_cc200.parquet --mask-out ./derivatives/roi_mask_cc200.npy
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# ----------------------------- I/O helpers ----------------------------------
def file_id_from_path(p: Path) -> str:
    """Map '*_rois_cc200.1D' or '*_rois_ho.1D' → FILE_ID used in ABIDE phenotypes."""
    stem = p.stem
    for suf in ("_rois_cc200", "_rois_ho"):
        if stem.endswith(suf):
            return stem[:-len(suf)]
    return stem

def read_1d(path: Path) -> np.ndarray:
    """
    Robust reader for AFNI .1D (ASCII numbers in rows/cols; '#' comments).
    Tries genfromtxt (handles NaNs), falls back to loadtxt.
    Returns a 2D array [T, N].
    """
    try:
        arr = np.genfromtxt(path, comments="#")
    except Exception:
        arr = np.loadtxt(path)
    if arr.ndim == 1:
        arr = arr[:, None]
    return arr

# ------------------------- correlation / vectorization -----------------------
def fisher_z_from_timeseries(
    ts: np.ndarray,
    neutralize_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute Fisher-z correlations from time-series (T x N).
    If 'neutralize_mask' is provided, rows/cols for those ROIs are set to r=0
    before the transform (keeps dimensionality constant).
    Always zeros the diagonal and clips r ∈ (-1, 1) to avoid ±∞ in atanh.
    Returns upper-triangle vector (float32).
    """
    # Pearson correlation; ts assumed finite except possibly some columns
    r = np.corrcoef(ts.T)
    if neutralize_mask is not None and neutralize_mask.any():
        r[neutralize_mask, :] = 0.0
        r[:, neutralize_mask] = 0.0
    np.fill_diagonal(r, 0.0)  # avoid atanh(1)
    r = np.clip(r, -0.999999, 0.999999)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.arctanh(r)
    iu = np.triu_indices_from(z, k=1)
    return z[iu].astype("float32")

# ------------------------------ strategies -----------------------------------
def build_fixed_length(
    files: list[Path],
    min_trs: int,
    std_eps: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build fixed-length connectomes for all subjects:
      - drop no ROI; detect invalid ROIs per subject via std <= std_eps or non-finite
      - neutralize their correlations (r=0) so vector length is constant
    Returns (connectomes_df, qc_df).
    """
    rows, ids, qc = [], [], []
    base_N = None
    for p in files:
        ts = read_1d(p)
        if ts.ndim != 2 or ts.shape[0] < min_trs:
            continue
        N = ts.shape[1]
        if base_N is None:
            base_N = N
        elif N != base_N:
            print(f"[warn] {p.name}: ROI count {N} differs from base {base_N}", file=sys.stderr)

        std = ts.std(axis=0)
        bad = ~np.isfinite(std) | (std <= std_eps) | ~np.isfinite(ts).all(axis=0)
        vec = fisher_z_from_timeseries(ts, neutralize_mask=bad)

        rows.append(vec)
        ids.append(file_id_from_path(p))
        qc.append({
            "FILE_ID": ids[-1],
            "n_trs": int(ts.shape[0]),
            "n_rois": int(N),
            "n_bad_rois": int(bad.sum()),
            "bad_roi_frac": float(bad.mean()),
            "strategy": "fixed"
        })

    if not rows:
        raise RuntimeError("No valid subjects produced connectomes.")
    X = np.vstack(rows)
    cols = [f"e{i}" for i in range(X.shape[1])]
    conn = pd.DataFrame(X, index=ids, columns=cols)
    conn.index.name = "FILE_ID"
    qcdf = pd.DataFrame(qc).set_index("FILE_ID")
    return conn, qcdf


def build_prevalence_masked(
    files: list[Path],
    min_trs: int,
    std_eps: float,
    roi_prevalence: float,
    max_bad_roi_frac: float | None,
    mask_out: Path | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Two-pass build:
      Pass 1: compute valid-ROI mask per subject; get prevalence across subjects;
              optionally drop subjects with too many invalid kept-ROIs
      Pass 2: compute connectomes only among kept ROIs for retained subjects
    Returns (connectomes_df, qc_df). Saves mask if 'mask_out' given.
    """
    valid_masks, ts_meta = [], []
    usable_files = []
    base_N = None

    # Pass 1: gather validity masks
    for p in files:
        ts = read_1d(p)
        if ts.ndim != 2 or ts.shape[0] < min_trs:
            continue
        N = ts.shape[1]
        if base_N is None:
            base_N = N
        elif N != base_N:
            print(f"[warn] {p.name}: ROI count {N} differs from base {base_N}", file=sys.stderr)
        std = ts.std(axis=0)
        ok = np.isfinite(std) & (std > std_eps) & np.isfinite(ts).all(axis=0)
        valid_masks.append(ok)
        usable_files.append(p)
        ts_meta.append((file_id_from_path(p), ts.shape[0], N))

    if not valid_masks:
        raise RuntimeError("No usable subjects in pass 1.")
    valid_arr = np.vstack(valid_masks)  # S × N
    roi_prev = valid_arr.mean(axis=0)
    keep_idx = np.where(roi_prev >= roi_prevalence)[0]
    print(f"[info] Prevalence≥{roi_prevalence:.2f}: kept {keep_idx.size} / {valid_arr.shape[1]} ROIs")

    # Optional subject QC (drop subjects with too many invalid among kept ROIs)
    keep_subj_mask = np.ones(valid_arr.shape[0], dtype=bool)
    if max_bad_roi_frac is not None:
        bad_frac = 1.0 - valid_arr[:, keep_idx].mean(axis=1)
        keep_subj_mask = bad_frac <= max_bad_roi_frac
        print(f"[info] Retained subjects after subject-QC (≤{max_bad_roi_frac:.2f} bad kept-ROIs): "
              f"{int(keep_subj_mask.sum())}/{valid_arr.shape[0]}")

        # Recompute prevalence on retained subjects for stability
        roi_prev2 = valid_arr[keep_subj_mask].mean(axis=0)
        keep_idx = np.where(roi_prev2 >= roi_prevalence)[0]
        print(f"[info] Final kept ROIs: {keep_idx.size}")

    if keep_idx.size < 3:
        raise RuntimeError("Too few ROIs after prevalence masking; adjust thresholds.")

    if mask_out is not None:
        mask_out.parent.mkdir(parents=True, exist_ok=True)
        np.save(mask_out, keep_idx)

    # Pass 2: build on kept ROIs only, for retained subjects
    rows, ids, qc = [], [], []
    kept_files = [f for k, f in enumerate(usable_files) if keep_subj_mask[k]]
    for p in kept_files:
        ts = read_1d(p)
        if ts.ndim != 2 or ts.shape[0] < min_trs:
            continue
        ts_sub = ts[:, keep_idx]
        # safe Fisher-z (no neutralization needed; columns were screened)
        vec = fisher_z_from_timeseries(ts_sub, neutralize_mask=None)
        rows.append(vec)
        fid = file_id_from_path(p)
        ids.append(fid)

        # QC bookkeeping
        n_trs, N_total = ts.shape[0], ts.shape[1]
        # fraction invalid among the final kept set for this subject
        std = ts.std(axis=0)
        ok_all = np.isfinite(std) & (std > std_eps) & np.isfinite(ts).all(axis=0)
        bad_kept_frac = 1.0 - ok_all[keep_idx].mean()
        qc.append({
            "FILE_ID": fid,
            "n_trs": int(n_trs),
            "n_rois_total": int(N_total),
            "n_rois_kept": int(keep_idx.size),
            "bad_kept_roi_frac": float(bad_kept_frac),
            "strategy": "prevalence",
            "roi_prevalence": float(roi_prevalence)
        })

    if not rows:
        raise RuntimeError("No valid subjects produced connectomes in pass 2.")
    X = np.vstack(rows)
    cols = [f"e{i}" for i in range(X.shape[1])]
    conn = pd.DataFrame(X, index=ids, columns=cols)
    conn.index.name = "FILE_ID"
    qcdf = pd.DataFrame(qc).set_index("FILE_ID")
    return conn, qcdf

# ---------------------------------- main -------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roi-root", required=True,
                    help="Directory containing AFNI ROI time-series files (e.g., *_rois_cc200.1D)")
    ap.add_argument("--atlas", choices=["cc200", "ho"], default="cc200",
                    help="Atlas token used in filenames (*_rois_<atlas>.1D)")
    ap.add_argument("--min-trs", type=int, default=120,
                    help="Skip subjects with fewer than this many time points (default: 120)")
    ap.add_argument("--std-eps", type=float, default=1e-8,
                    help="Minimum std to consider an ROI valid (default: 1e-8)")

    ap.add_argument("--strategy", choices=["fixed", "prevalence"], default="fixed",
                    help="Vectorization strategy")
    ap.add_argument("--roi-prevalence", type=float, default=0.95,
                    help="(prevalence) Keep ROI if valid in ≥ this fraction of subjects")
    ap.add_argument("--max-bad-roi-frac", type=float, default=0.25,
                    help="(prevalence) Drop subject if > this fraction of kept ROIs are invalid")
    ap.add_argument("--mask-out", type=Path, default=None,
                    help="(prevalence) Path to save kept ROI indices as .npy")

    ap.add_argument("--out", required=True,
                    help="Output Parquet path for connectomes matrix (subjects × edges)")
    ap.add_argument("--qc-csv", default=None,
                    help="Path to write QC CSV (optional; default: alongside --out)")

    args = ap.parse_args()

    roi_root = Path(args.roi_root)
    pattern = f"*_rois_{args.atlas}.1D"
    files = sorted(roi_root.glob(pattern))
    if not files:
        print(f"[error] No files under {roi_root} matching {pattern}", file=sys.stderr)
        sys.exit(1)

    if args.strategy == "fixed":
        conn, qc = build_fixed_length(files, args.min_trs, args.std_eps)
    else:
        conn, qc = build_prevalence_masked(
            files=files,
            min_trs=args.min_trs,
            std_eps=args.std_eps,
            roi_prevalence=args.roi_prevalence,
            max_bad_roi_frac=args.max_bad_roi_frac,
            mask_out=args.mask_out,
        )

    # Write outputs
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    conn.to_parquet(out_path, compression="zstd")
    if args.qc_csv is None:
        qc_csv = out_path.with_suffix("").as_posix() + "_qc.csv"
    else:
        qc_csv = args.qc_csv
    Path(qc_csv).parent.mkdir(parents=True, exist_ok=True)
    qc.to_csv(qc_csv)

    # Console report
    print(f"[done] Wrote connectomes {conn.shape} → {out_path}")
    print(f"[done] Wrote per-subject QC → {qc_csv}")
    if args.strategy == "fixed":
        # Expect N*(N-1)/2 edges; infer N from vector length
        L = conn.shape[1]
        # Solve L = N*(N-1)/2 for N
        N = int((1 + np.sqrt(1 + 8*L)) / 2)
        print(f"[info] Fixed-length vector size {L} implies {N} ROIs.")
    else:
        print(f"[info] Prevalence-masked vector length = {conn.shape[1]} edges.")

if __name__ == "__main__":
    main()
