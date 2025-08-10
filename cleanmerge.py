import pandas as pd, numpy as np
from pathlib import Path

# ----------------- Load phenotypes -----------------
pheno = pd.read_csv("Phenotypic_V1_0b_preprocessed1.csv")
pheno["FILE_ID"] = pheno["FILE_ID"].astype(str).str.strip()
pheno["func_mean_fd"] = pd.to_numeric(pheno["func_mean_fd"], errors="coerce")
pheno_full = pheno.set_index("FILE_ID", drop=False).copy()   # keep an unfiltered copy

# ----------------- Helper functions ----------------
def file_id_from_path(p: Path) -> str:
    stem = p.stem
    for suf in ("_rois_cc200", "_rois_ho"):
        if stem.endswith(suf):
            return stem[:-len(suf)]
    return stem

def fisher_z_connectivity(ts: np.ndarray) -> np.ndarray:
    std = ts.std(axis=0)
    ok = np.isfinite(std) & (std > 0)
    ts = ts[:, ok]
    r = np.corrcoef(ts.T)
    np.fill_diagonal(r, 0.0)
    r = np.clip(r, -0.999999, 0.999999)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.arctanh(r)
    iu = np.triu_indices_from(z, 1)
    return z[iu].astype("float32")

# ----------------- Load one ROI file ----------------
roi_file = Path("rois_cc200/CMU_b_0050650_rois_cc200.1D")
fid = file_id_from_path(roi_file)

# -- Diagnostics: is this subject present pre/post QC?
in_pheno = fid in pheno_full.index
fd_val = pheno_full.loc[fid, "func_mean_fd"] if in_pheno else np.nan
print(f"ROI file maps to FILE_ID={fid}")
print(f"Present in phenotypes (pre-QC)? {in_pheno}")
if in_pheno:
    print(f"func_mean_fd for {fid}: {fd_val}")

# If the subject failed FD≤0.2, that explains the empty merge you saw.
passed_qc = (pd.notna(fd_val) and fd_val <= 0.2)

# ----------------- Compute connectivity vector ----------------
ts = np.loadtxt(roi_file)            # rows=time, cols=ROIs
vec = fisher_z_connectivity(ts)
conn_cols = [f"e{i}" for i in range(vec.size)]
conn_df = pd.DataFrame([vec], index=[fid], columns=conn_cols)

# ----------------- Merge, then (optionally) filter by FD ------------
merged = pheno_full.join(conn_df, how="inner")
print("Merged (pre-QC) shape:", merged.shape)    # should be (1, 19900 + pheno cols) if key matches

# Now apply motion QC AFTER the merge (or skip if you prefer a softer threshold)
merged_qc = merged.loc[merged["func_mean_fd"] <= 0.2]
print("Merged (post-QC, FD<=0.2) shape:", merged_qc.shape)
