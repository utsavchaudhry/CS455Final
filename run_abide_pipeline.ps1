<#  ABIDE site-differences pipeline — end-to-end runner (PowerShell)
    Usage:  .\run_abide_pipeline.ps1
#>
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Log($msg) { Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $msg" }
function Ensure-Dir($p) { if (-not (Test-Path $p)) { New-Item -ItemType Directory -Path $p | Out-Null } }

# -----------------------------
# Configuration (edit if needed)
# -----------------------------
$ROI_ROOT = ".\rois_cc200"
$PHENO    = ".\Phenotypic_V1_0b_preprocessed1.csv"
$DERIV    = ".\derivatives"
$ATLAS    = "cc200"
$FD_THRESH = 0.2
$SEED = 0

# 04_* classification/silhouette settings (your final runs)
$SIL_NCOMP = 50
$SIL_MAXN  = 600
$LR_PCA = 50;   $LR_C = 0.5;   $LR_MAXITER = 10000; $LR_SOLVER = "saga"
$RIDGE_PCA = 50; $RIDGE_ALPHA = 1.0
$LSVC_PCA = 50;  $LSVC_C = 0.5; $LSVC_MAXITER = 20000
$KNN_PCA = 50;   $KNN_K = 5
$MAX_PER_SITE = 40

# Permutations (final report)
$PERM_N = 200
$PERM_MODELS = "knn,ridge,svc"
$PERM_PCA = 30
$PERM_MAX_PER_SITE = 25

# -----------------------------
# 0) Virtualenv + dependencies
# -----------------------------
if (-not (Test-Path ".\venv\Scripts\Activate.ps1")) {
  Log "Creating Python venv ..."
  python -m venv venv
}
. .\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install numpy pandas pyarrow fastparquet scikit-learn matplotlib scipy statsmodels nibabel neuroHarmonize neuroCombat

# -----------------------------
# 1) Build connectomes
# -----------------------------
Ensure-Dir $DERIV
$OUT_CONNECTOMES = Join-Path $DERIV "connectomes_cc200.parquet"
Log "Step 1/5: Building connectomes → $OUT_CONNECTOMES"
python build_connectomes.py `
  --roi-root $ROI_ROOT `
  --atlas $ATLAS `
  --strategy fixed `
  --out $OUT_CONNECTOMES

# -----------------------------
# 2) Merge + motion QC
# -----------------------------
$MERGED = Join-Path $DERIV "abide_cc200_merged_fd02.parquet"
Log "Step 2/5: Merging with phenotypes (FD ≤ $FD_THRESH) → $MERGED"
python 02_merge_and_qc.py `
  --connectomes $OUT_CONNECTOMES `
  --phenotypes $PHENO `
  --fd-thresh $FD_THRESH `
  --out $MERGED

# -----------------------------
# 3) Harmonize with ComBat
# -----------------------------
$OUT_PREFIX = Join-Path $DERIV "abide_cc200"
$COMBAT = Join-Path $DERIV "abide_cc200_combat.parquet"
Log "Step 3/5: Harmonizing with ComBat → $COMBAT"
python 03_harmonize_and_check.py `
  --merged $MERGED `
  --out-prefix $OUT_PREFIX `
  --covars AGE_AT_SCAN SEX

# -----------------------------
# 4) Residual site effects
# -----------------------------
$DIAG = Join-Path $DERIV "diagnostics"
Ensure-Dir $DIAG

Log "Step 4/5a: Residual site effects (fast sanity, no permutations)"
python 04_check_residual_site_effects.py `
  --raw $MERGED `
  --combat $COMBAT `
  --outdir $DIAG `
  --sil-n-components $SIL_NCOMP --sil-max-n $SIL_MAXN `
  --lr-pca-components $LR_PCA --lr-C $LR_C --lr-max-iter $LR_MAXITER --lr-solver $LR_SOLVER `
  --ridge-pca-components $RIDGE_PCA --ridge-alpha $RIDGE_ALPHA `
  --linsvc-pca-components $LSVC_PCA --linsvc-C $LSVC_C --linsvc-max-iter $LSVC_MAXITER `
  --knn-pca-components $KNN_PCA --knn-k $KNN_K `
  --max-per-site $MAX_PER_SITE `
  --seed $SEED

Log "Step 4/5b: Residual site effects with permutation p-values (final report)"
python 04_check_residual_site_effects.py `
  --raw $MERGED `
  --combat $COMBAT `
  --outdir $DIAG `
  --sil-n-components $SIL_NCOMP --sil-max-n $SIL_MAXN `
  --lr-pca-components $LR_PCA --lr-C $LR_C --lr-max-iter $LR_MAXITER --lr-solver $LR_SOLVER `
  --ridge-pca-components $RIDGE_PCA --ridge-alpha $RIDGE_ALPHA `
  --linsvc-pca-components $LSVC_PCA --linsvc-C $LSVC_C --linsvc-max-iter $LSVC_MAXITER `
  --knn-pca-components $KNN_PCA --knn-k $KNN_K `
  --max-per-site $MAX_PER_SITE `
  --perm-n $PERM_N --perm-models $PERM_MODELS `
  --perm-pca-components $PERM_PCA --perm-max-per-site $PERM_MAX_PER_SITE `
  --seed $SEED

# -----------------------------
# 5) Paper-ready figures
# -----------------------------
$FIGDIR = Join-Path $DERIV "figures"
Ensure-Dir $FIGDIR
Log "Step 5/5: Generating figures → $FIGDIR"
python 05_make_figures.py `
  --diagnostics $DIAG `
  --raw-parquet $MERGED `
  --combat-parquet $COMBAT `
  --outdir $FIGDIR

Log "Done."
Log "Key outputs:"
Log "  Connectomes         : $OUT_CONNECTOMES"
Log "  Merged (FD<=$FD_THRESH): $MERGED"
Log "  ComBat              : $COMBAT"
Log "  Diagnostics         : $DIAG"
Log "  Figures (PNGs)      : $FIGDIR"
