# DX-Group Case–Control Analysis (ABIDE-style)
This package extends your site-effects pipeline to a **case–control analysis** using `DX_GROUP`
from `Phenotypic_V1_0b_preprocessed1.csv`. It preserves clinical effects during batch harmonization
and reproduces the following components:

1. Load & align phenotypes with connectome features
2. QC filtering (motion, completeness)
3. ComBat harmonization with `DX_GROUP` protected
4. Mass-univariate GLM (edge ~ DX + covariates) with FDR
5. QC–FC within-group summaries
6. Classification (stratified CV and LOSO-by-site)
7. Reporting helpers and figure generation

> **DX mapping (ABIDE convention):** `DX_GROUP=1` → ASD/patient; `DX_GROUP=2` → Control/healthy.

## Quick start
```
python -m pip install -r requirements.txt
python dx_main.py --phenopath /path/Phenotypic_V1_0b_preprocessed1.csv                   --xpath X_raw.npy --subids sub_ids.npy                   --outdir ./dx_out
```

- `X_raw.npy`: shape `(n_subjects, n_edges)` features (e.g., upper triangle Fisher-z correlations).
- `sub_ids.npy`: subject IDs (must match the phenotype file's column `SUB_ID` by default).
- The script writes CSV summaries and figures under `--outdir`.

## Files
- `dx_config.py` — configuration and CLI defaults.
- `dx_data.py` — IO, alignment, QC.
- `dx_harmonize.py` — ComBat with disease preserved (fallback if library is absent).
- `dx_glm.py` — mass-univariate GLM, FDR, Hedges g.
- `dx_qcfc.py` — QC–FC correlation summaries by group.
- `dx_classify.py` — classifiers (stratified CV and LOSO-by-site).
- `dx_report.py` — simple plots/tables.
- `dx_main.py` — orchestration CLI.

## Notes
- If `neurocombat-sklearn` is not available, the pipeline continues **without harmonization**,
  but you can still add site fixed effects in GLM by passing `--glm-site-fe`.
- Figures use only matplotlib with default colors and single-plot per figure, to keep environments light-weight.

## Using ROI time series (*.1D) instead of .npy

If you have ABIDE-style AFNI `.1D` ROI time series (T×R) per subject:

### Option A — one-shot run (no intermediate .npy)
```
python dx_main.py   --phenopath /path/Phenotypic_V1_0b_preprocessed1.csv   --roi-dir /path/to/roi_timeseries_dir   --outdir ./dx_out
```

### Option B — prebuild features to .npy for reuse
```
python dx_build_from_1D.py --roi-dir /path/to/roi_timeseries_dir --outdir ./dx_feats
python dx_main.py   --phenopath /path/Phenotypic_V1_0b_preprocessed1.csv   --xpath ./dx_feats/X_raw.npy   --subids ./dx_feats/sub_ids.npy   --outdir ./dx_out
```
