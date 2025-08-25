
from __future__ import annotations
import os, argparse, numpy as np, pandas as pd
from dx_from_1d import build_features_from_1d

def parse_args():
    ap = argparse.ArgumentParser(description="Build connectome feature matrix from ROI time series (*.1D).")
    ap.add_argument("--roi-dir", required=True, help="Directory with *.1D files (T x R).")
    ap.add_argument("--pattern", default="*.1D", help="Glob pattern (default: *.1D).")
    ap.add_argument("--no-fisher", action="store_true", help="Disable Fisher z transform (default: enabled).")
    ap.add_argument("--outdir", required=True, help="Output directory for X_raw.npy and sub_ids.npy")
    ap.add_argument("--csv", action="store_true", help="Also write CSV versions (X_raw.csv, sub_ids.csv).")
    return ap.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    X, sub_ids = build_features_from_1d(args.roi_dir, pattern=args.pattern, fisher=(not args.no_fisher))
    np.save(os.path.join(args.outdir, "X_raw.npy"), X)
    np.save(os.path.join(args.outdir, "sub_ids.npy"), sub_ids)
    if args.csv:
        import pandas as pd
        pd.DataFrame(X).to_csv(os.path.join(args.outdir, "X_raw.csv"), index=False)
        pd.Series(sub_ids, name="SUB_ID").to_csv(os.path.join(args.outdir, "sub_ids.csv"), index=False)
    print("Wrote", X.shape[0], "subjects with", X.shape[1], "edges to", args.outdir)

if __name__ == "__main__":
    main()
