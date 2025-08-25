
from __future__ import annotations
import argparse, os, json
import numpy as np, pandas as pd
from dx_config import DXConfig
from dx_data import load_phenotype, load_features, align_by_subject, qc_filter
from dx_harmonize import combat_harmonize_preserve_dx
from dx_glm import mass_univariate_glm
from dx_qcfc import qcfc_by_group
from dx_classify import classify_stratified, classify_loso
from dx_report import save_volcano, save_histogram, write_table, cohort_table

def parse_args():
    ap = argparse.ArgumentParser(description="DX-group case–control analysis pipeline")
    ap.add_argument("--phenopath", required=True, help="Path to Phenotypic_V1_0b_preprocessed1.csv")
    # Option A: precomputed features
    ap.add_argument("--xpath", help="Path to features .npy (n_subjects x n_edges)")
    ap.add_argument("--subids", help="Path to subject IDs .npy (n_subjects,)")
    # Option B: build from ROI time series
    ap.add_argument("--roi-dir", help="Directory with ROI time series (*.1D). If given, --xpath/--subids are ignored.")
    ap.add_argument("--pattern", default="*.1D", help="Glob pattern for ROI files (default: *.1D)")
    ap.add_argument("--no-fisher", action="store_true", help="Disable Fisher z on connectivity (default: enabled)")
    # General
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--glm-site-fe", action="store_true", help="Add site fixed effects if ComBat is not used")
    ap.add_argument("--no-combat", action="store_true", help="Skip ComBat harmonization explicitly")
    return ap.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    cfg = DXConfig()

    # 1) IO & alignment
    ph = load_phenotype(args.phenopath, cfg)
    if args.roi_dir:
        from dx_from_1d import build_features_from_1d
        X, sub_ids = build_features_from_1d(args.roi_dir, pattern=args.pattern, fisher=(not args.no_fisher))
    else:
        if not (args.xpath and args.subids):
            raise SystemExit("Either provide --roi-dir or both --xpath and --subids.")
        X, sub_ids = load_features(args.xpath, args.subids)

    ph = align_by_subject(ph, sub_ids, cfg)

    # 2) QC
    mask = qc_filter(ph, cfg)
    X = X[mask]; ph = ph.loc[mask].reset_index(drop=True)

    # Sanitize features: remove NaN/Inf early to avoid failures in ComBat/PCA/GLM
    import numpy as _np
    X = _np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Drop single-group sites (required for identifiable ComBat with DX preserved)
    counts = ph.groupby([cfg.site_col, "DX_BIN"]).size().unstack(fill_value=0)
    keep_sites = counts[(counts.get(0,0) > 0) & (counts.get(1,0) > 0)].index.astype(str)
    mask_sites = ph[cfg.site_col].astype(str).isin(keep_sites)
    X = X[mask_sites.values]
    ph = ph.loc[mask_sites].reset_index(drop=True)


    # 3) Harmonization (preserving DX)
    if args.no_combat:
        Xc = X.copy()
    else:
        Xc = combat_harmonize_preserve_dx(X, ph, cfg)

    # 4) Mass-univariate GLM
    glm = mass_univariate_glm(Xc, ph, cfg, site_fe=args.glm_site_fe)
    np.save(os.path.join(args.outdir, "beta_dx.npy"), glm["beta_dx"])
    np.save(os.path.join(args.outdir, "qvals.npy"), glm["qvals"])
    np.save(os.path.join(args.outdir, "g_vals.npy"), glm["g"])

    np.save(os.path.join(args.outdir, "pvals.npy"), glm["pvals"])
    import pandas as pd
    top = pd.DataFrame({
        "beta_dx": glm["beta_dx"],
        "pval":    glm["pvals"],
        "qval":    glm["qvals"],
        "g":       glm["g"],
    }).sort_values("pval").head(50)
    top.to_csv(os.path.join(args.outdir, "top_edges.csv"), index=False)


    # 5) QC–FC summaries
    qcfc = qcfc_by_group(Xc, ph, cfg)
    with open(os.path.join(args.outdir, "qcfc.json"), "w") as f:
        json.dump(qcfc, f, indent=2)

    # 6) Classification
    y = ph["DX_BIN"].values
    site = ph[cfg.site_col].astype(str).values
    metrics_cv   = classify_stratified(Xc, y, cfg)
    metrics_loso = classify_loso(Xc, y, site, cfg)
    with open(os.path.join(args.outdir, "classify_cv.json"), "w") as f:
        json.dump(metrics_cv, f, indent=2)
    with open(os.path.join(args.outdir, "classify_loso.json"), "w") as f:
        json.dump(metrics_loso, f, indent=2)

    # 7) Reporting (tables + figs)
    write_table(cohort_table(ph, cfg), os.path.join(args.outdir, "cohort_by_site.csv"))
    save_volcano(glm["beta_dx"], glm["qvals"], os.path.join(args.outdir, "volcano_dx.png"))
    save_histogram(glm["g"], os.path.join(args.outdir, "effectsize_hist.png"), "Hedges g (ASD - Control)")

    # Console summary
    print("[DX] Subjects after QC:", len(ph))
    print("[DX] Significant edges (FDR q<0.05):", int(glm["sig_mask"].sum()))
    print("[DX] Stratified CV:", metrics_cv)
    print("[DX] LOSO by site:", metrics_loso)

if __name__ == "__main__":
    main()
