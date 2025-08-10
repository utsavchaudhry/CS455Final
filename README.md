# Site Differences in Neuroimaging (ABIDE)

> **Reproducible pipeline to quantify and mitigate scanner/site effects in multi‑site rsfMRI (ABIDE, CC200)**

&#x20;&#x20;

This repository builds functional connectomes from ABIDE CC200 ROI time‑series, merges with phenotypes, harmonizes edges via **ComBat**, and evaluates **residual site effects** via classification, silhouette, and QC–FC. It also exports figures suitable for Overleaf/ISBI.

---

## Contents

- [What you get](#what-you-get)
- [Quickstart](#quickstart)
- [Data layout](#data-layout)
- [End‑to‑end commands](#end-to-end-commands)
- [Parameters](#parameters)
- [Outputs](#outputs)
- [Troubleshooting](#troubleshooting)
- [Reproducibility notes](#reproducibility-notes)
- [Cite](#cite)

---

## What you get

**Key claims (from our successful run):**

- ComBat reduced median **F(site)** from **4.519** (raw) → **0.266** (≈ **4.25×** reduction).
- Site‑prediction accuracy collapsed from \~0.80 (LR+PCA, raw) to \~0.03 (ComBat).
- QC–FC |r| shifted: **0.194 → 0.169**; FDR<0.05 edges: **92.1% → 89.3%**.

See `derivatives/diagnostics` and `derivatives/figures` for plots.

---

## Quickstart

```bash
# 1) Create & activate a virtual environment (example: Windows PowerShell)
python -m venv venv
venv\Scripts\Activate.ps1

# 2) Install deps
pip install -r requirements.txt

# 3) Put data in place (see layout below), then run the orchestrator
#    Windows PowerShell
./run_abide_pipeline.ps1
```

> **Batch run:** `run_abide_pipeline.ps1` executes Steps 1–5 with the exact arguments used in our analysis and drops all artifacts under `./derivatives`.

---

## Data layout

```
<ROOT>/
  ├─ rois_cc200/                         # CC200 ROI time‑series, .1D per subject
  │   └─ <SITE>_<...>_<SUB_ID>_rois_cc200.1D
  ├─ Phenotypic_V1_0b_preprocessed1.csv  # ABIDE phenotypes
  ├─ build_connectomes.py
  ├─ 02_merge_and_qc.py
  ├─ 03_harmonize_and_check.py
  ├─ 04_check_residual_site_effects.py
  ├─ 05_make_figures.py
  ├─ run_abide_pipeline.ps1               # one‑touch orchestration
  └─ derivatives/                         # created by the pipeline
```

**Notes**

- ROI files must share a **SUB\_ID/FILE\_ID** convention consistent with ABIDE phenotypes (already handled in the scripts).
- We use **FD ≤ 0.20 mm** as strict motion QC by default.

---

## End‑to‑end commands

You can run each step manually (cross‑platform) or rely on the PowerShell orchestrator.

### 1) Build connectomes

```bash
python build_connectomes.py \
  --roi-root ./rois_cc200 \
  --atlas cc200 \
  --strategy fixed \
  --out ./derivatives/connectomes_cc200.parquet
```

**Output:** `derivatives/connectomes_cc200.parquet` (≈934×19,900) and `connectomes_cc200_qc.csv`.

### 2) Merge with phenotypes + motion QC

```bash
python 02_merge_and_qc.py \
  --connectomes ./derivatives/connectomes_cc200.parquet \
  --phenotypes ./Phenotypic_V1_0b_preprocessed1.csv \
  --fd-thresh 0.2 \
  --out ./derivatives/abide_cc200_merged_fd02.parquet
```

**Output:** `derivatives/abide_cc200_merged_fd02.parquet` (n≈753 after FD filter across 17 sites).

### 3) Harmonize with ComBat

```bash
python 03_harmonize_and_check.py \
  --merged ./derivatives/abide_cc200_merged_fd02.parquet \
  --out-prefix ./derivatives/abide_cc200 \
  --covars AGE_AT_SCAN SEX
```

**Output:** `derivatives/abide_cc200_combat.parquet` + console summary of F(site) reduction.

### 4) Residual site effects (fast sanity)

```bash
python 04_check_residual_site_effects.py \
  --raw ./derivatives/abide_cc200_merged_fd02.parquet \
  --combat ./derivatives/abide_cc200_combat.parquet \
  --outdir ./derivatives/diagnostics \
  --sil-n-components 50 --sil-max-n 600 --seed 0 \
  --lr-pca-components 50 --lr-C 0.5 --lr-max-iter 10000 --lr-solver saga \
  --ridge-pca-components 50 --ridge-alpha 1.0 \
  --linsvc-pca-components 50 --linsvc-C 0.5 --linsvc-max-iter 20000 \
  --knn-pca-components 50 --knn-k 5 --max-per-site 40
```

### 4b) Residual site effects (permutation p‑values)

```bash
python 04_check_residual_site_effects.py \
  --raw ./derivatives/abide_cc200_merged_fd02.parquet \
  --combat ./derivatives/abide_cc200_combat.parquet \
  --outdir ./derivatives/diagnostics \
  --sil-n-components 50 --sil-max-n 600 --seed 0 \
  --lr-pca-components 50 --lr-C 0.5 --lr-max-iter 10000 --lr-solver saga \
  --ridge-pca-components 50 --ridge-alpha 1.0 \
  --linsvc-pca-components 50 --linsvc-C 0.5 --linsvc-max-iter 20000 \
  --knn-pca-components 50 --knn-k 5 --max-per-site 40 \
  --perm-n 200 --perm-models knn,ridge,svc \
  --perm-pca-components 30 --perm-max-per-site 25
```

### 5) Figures for the paper

```bash
python 05_make_figures.py \
  --raw ./derivatives/abide_cc200_merged_fd02.parquet \
  --combat ./derivatives/abide_cc200_combat.parquet \
  --outdir ./derivatives/figures
```

---

## Parameters

| Script                              | Key flags        | Default                 | Notes                                                         |
| ----------------------------------- | ---------------- | ----------------------- | ------------------------------------------------------------- |
| `build_connectomes.py`              | `--strategy`     | `fixed`                 | Fisher‑z upper‑triangle features (p=19900 for CC200).         |
|                                     | `--atlas`        | `cc200`                 | Assumes 200 ROI time‑series per subject.                      |
| `02_merge_and_qc.py`                | `--fd-thresh`    | `0.2`                   | Strict motion threshold (mm).                                 |
| `03_harmonize_and_check.py`         | `--covars`       | `AGE_AT_SCAN SEX`       | ComBat covariates; batch=`SITE_ID`.                           |
| `04_check_residual_site_effects.py` | `--max-per-site` | `40`                    | Caps class imbalance.                                         |
|                                     | `--sil-*`        | `50, 600`               | PCA components / max N for silhouette.                        |
|                                     | `--perm-n`       | `200`                   | Number of permutations. Reduce for speed.                     |
|                                     | `--perm-models`  | `knn,ridge,svc`         | Models to permute (LR is often trivially significant in raw). |
| `05_make_figures.py`                | `--outdir`       | `./derivatives/figures` | PNGs for Overleaf/ISBI.                                       |

---

## Outputs

```
./derivatives/
  ├─ connectomes_cc200.parquet
  ├─ connectomes_cc200_qc.csv
  ├─ abide_cc200_merged_fd02.parquet
  ├─ abide_cc200_combat.parquet
  ├─ diagnostics/
  │   ├─ metrics_*.json / .csv
  │   └─ plots_*.png (e.g., silhouette, QC–FC, site‑pred confusion)
  └─ figures/
      ├─ site_acc_raw_vs_combat.png
      ├─ site_f1_raw_vs_combat.png
      ├─ silhouette_raw_vs_combat.png
      ├─ qcfc_abs_r_hist_raw.png
      ├─ qcfc_abs_r_hist_combat.png
      ├─ perm_knn_pca_raw_acc.png
      ├─ perm_knn_pca_combat_acc.png
      ├─ pca_scatter_raw.png
      └─ pca_scatter_ComBat.png
```

### Figures (inline previews)

> If you're browsing on GitHub, the previews below load from `derivatives/figures/` produced by Step 5.

---

## Troubleshooting

- **Parquet engine not found** → `pip install pyarrow fastparquet` (both listed in `requirements.txt`).
- **neuroHarmonize import errors** (e.g., `statsmodels`, `nibabel`) → ensure these are installed; script gracefully falls back to **neuroCombat**.
- **Slow permutations** → decrease `--perm-n` and/or limit with `--perm-max-per-site`.
- **Convergence warnings (LR)** → we use `saga` + `max_iter=10000` and PCA=50; warnings are expected but results are stable.
- **Shape mismatch** → ensure ROI time‑series truly have 200 columns; the pipeline enforces vector length = 19,900 edges for CC200.

---

## Reproducibility notes

- Seeded: `--seed 0` for stochastic steps.
- Site balancing: `--max-per-site` to avoid dominance.
- Exact CLI arguments above reproduce our figures and numbers.
- Works cross‑platform (paths/shell differ). The PowerShell orchestrator captures the Windows run used in our study.

---

## Cite

If you use this pipeline, please consider citing:

- Abraham et al., *NeuroImage* 2016 — multi‑site reproducibility in autism.
- Yu et al., *HBM* 2018 — ComBat for functional connectivity.
- Dansereau et al., *NeuroImage* 2017 — multi‑site power/prediction.
- Chen et al., *HBM* 2021 — covariance harmonization.
- Yamashita et al., *PLOS Biology* 2019 — measurement vs sampling bias.
- Carmon et al., *NeuroImage* 2020 — structural covariance reliability.
- Xu et al., *Frontiers in Neuroscience* 2023 — dual‑projection ICA.
- An et al., *MedIA* 2025 — deep harmonization.
- Gardner et al., *HBM* 2025 — ComBatLS.

> For full BibTeX, see the Overleaf manuscript or add these entries to your repo’s `refs.bib`.

---

### License

This project is released under the **MIT License**. See `LICENSE`.

### Acknowledgements

Data were provided by the **ABIDE** initiative and the **Preprocessed Connectomes Project**. Thanks to all contributing sites and investigators.

