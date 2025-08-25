from dataclasses import dataclass

@dataclass
class DXConfig:
    # Column names in the phenotype CSV
    sub_id_col: str = "SUB_ID"
    site_col: str = "SITE_ID"
    dx_col: str = "DX_GROUP"           # 1=ASD patients, 2=Controls (ABIDE convention)
    age_col: str = "AGE_AT_SCAN"
    sex_col: str = "SEX"               # 1 = male, 2 = female (ABIDE convention)
    fd_col: str = "func_mean_fd"       # mean framewise displacement per subject

    # QC thresholds
    max_fd: float = 0.20               # mm

    # GLM/FDR
    fdr_q: float = 0.05

    # PCA/Classify
    n_pca: int = 50
