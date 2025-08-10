#!/usr/bin/env bash
# ABIDE site-differences pipeline — end-to-end runner (Linux/macOS bash)
# Usage: bash run_abide_pipeline.sh
set -euo pipefail

# -----------------------------
# Configuration (edit if needed)
# -----------------------------
ROI_ROOT="./rois_cc200"
PHENO="./Phenotypic_V1_0b_preprocessed1.csv"
DERIV="./derivatives"
ATLAS="cc200"
FD_THRESH="0.2"
SEED=0

# 04_* classification/silhouette settings (match your final runs)
SIL_NCOMP=50
SIL_MAXN=600
LR_PCA=50; LR_C=0.5; LR_MAXITER=10000; LR_SOLVER="saga"
RIDGE_PCA=50; RIDGE_ALPHA=1.0
LSVC_PCA=50; LSVC_C=0.5; LSVC_MAXITER=20000
KNN_PCA=50; KNN_K=5
MAX_PER_SITE=40

# Permutations (final run)
PERM_N=200
PERM_MODELS="knn,ridge,svc"
PERM_PCA=30
PERM_MAX_PER_SITE=25

# -----------------------------
# Helpers
# -----------------------------
ts() { date +"%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(ts)] $*"; }
ensure_dir() { mkdir -p "$1"; }

# -----------------------------
# 0) Virtualenv + dependencies
# -----------------------------
if [ ! -d "venv" ]; then
  log "Creating Python venv ..."
  python3 -m venv venv
fi
# shellcheck disable=SC1091
source venv/bin/activate
python -m pip install --upgrade pip
# Core deps (pinning not required for this repro; adjust if your team prefers)
pip install numpy pandas pyarrow fastparquet scikit-learn matplotlib scipy statsmodels nibabel neuroHarmonize neuroCombat

# -----------------------------
# 1) Build connectomes
# -----------------------------
ensure_dir "$DERIV"
OUT_CONNECTOMES="$DERIV/connectomes_cc200.parquet"
log "Step 1/5: Building connectomes → $OUT_CONNECTOMES"
python build_connectomes.py \
  --roi-root "$ROI_ROOT" \
  --atlas "$ATLAS" \
  --strategy fixed \
  --out "$OUT_CONNECTOMES"

# -----------------------------
# 2) Merge + motion QC
# -----------------------------
MERGED="$DERIV/abide_cc200_merged_fd02.parquet"
log "Step 2/5: Merging with phenotypes (FD ≤ $FD_THRESH) → $MERGED"
python 02_merge_and_qc.py \
  --connectomes "$OUT_CONNECTOMES" \
  --phenotypes "$PHENO" \
  --fd-thresh "$FD_THRESH" \
  --out "$MERGED"

# -----------------------------
# 3) Harmonize with ComBat
# -----------------------------
OUT_PREFIX="$DERIV/abide_cc200"
COMBAT="$DERIV/abide_cc200_combat.parquet"
log "Step 3/5: Harmonizing with ComBat → $COMBAT"
python 03_harmonize_and_check.py \
  --merged "$MERGED" \
  --out-prefix "$OUT_PREFIX" \
  --covars AGE_AT_SCAN SEX

# -----------------------------
# 4) Residual site effects
# -----------------------------
DIAG="$DERIV/diagnostics"
ensure_dir "$DIAG"

log "Step 4/5a: Residual site effects (fast sanity, no permutations)"
python 04_check_residual_site_effects.py \
  --raw "$MERGED" \
  --combat "$COMBAT" \
  --outdir "$DIAG" \
  --sil-n-components "$SIL_NCOMP" --sil-max-n "$SIL_MAXN" \
  --lr-pca-components "$LR_PCA" --lr-C "$LR_C" --lr-max-iter "$LR_MAXITER" --lr-solver "$LR_SOLVER" \
  --ridge-pca-components "$RIDGE_PCA" --ridge-alpha "$RIDGE_ALPHA" \
  --linsvc-pca-components "$LSVC_PCA" --linsvc-C "$LSVC_C" --linsvc-max-iter "$LSVC_MAXITER" \
  --knn-pca-components "$KNN_PCA" --knn-k "$KNN_K" \
  --max-per-site "$MAX_PER_SITE" \
  --seed "$SEED"

log "Step 4/5b: Residual site effects with permutation p-values (final report)"
python 04_check_residual_site_effects.py \
  --raw "$MERGED" \
  --combat "$COMBAT" \
  --outdir "$DIAG" \
  --sil-n-components "$SIL_NCOMP" --sil-max-n "$SIL_MAXN" \
  --lr-pca-components "$LR_PCA" --lr-C "$LR_C" --lr-max-iter "$LR_MAXITER" --lr-solver "$LR_SOLVER" \
  --ridge-pca-components "$RIDGE_PCA" --ridge-alpha "$RIDGE_ALPHA" \
  --linsvc-pca-components "$LSVC_PCA" --linsvc-C "$LSVC_C" --linsvc-max-iter "$LSVC_MAXITER" \
  --knn-pca-components "$KNN_PCA" --knn-k "$KNN_K" \
  --max-per-site "$MAX_PER_SITE" \
  --perm-n "$PERM_N" --perm-models "$PERM_MODELS" \
  --perm-pca-components "$PERM_PCA" --perm-max-per-site "$PERM_MAX_PER_SITE" \
  --seed "$SEED"

# -----------------------------
# 5) Paper-ready figures
# -----------------------------
FIGDIR="$DERIV/figures"
ensure_dir "$FIGDIR"
log "Step 5/5: Generating figures → $FIGDIR"
python 05_make_figures.py \
  --diagnostics "$DIAG" \
  --raw-parquet "$MERGED" \
  --combat-parquet "$COMBAT" \
  --outdir "$FIGDIR"

log "Done."
log "Key outputs:"
log "  Connectomes      : $OUT_CONNECTOMES"
log "  Merged (FD≤$FD_THRESH): $MERGED"
log "  ComBat           : $COMBAT"
log "  Diagnostics JSON/CSV: $DIAG"
log "  Figures (PNGs)   : $FIGDIR"
