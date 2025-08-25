from __future__ import annotations
import numpy as np, pandas as pd
from typing import Dict, Tuple
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from dx_config import DXConfig

def _design_matrix(ph: pd.DataFrame, cfg: DXConfig, site_fe: bool=False) -> pd.DataFrame:
    Xcov = pd.DataFrame({
        "Intercept": 1.0,
        "DX": ph["DX_BIN"].values,
        "Age": ph[cfg.age_col].values,
        "SexF": ph["SexF"].values,
        "FD": ph[cfg.fd_col].values
    })
    if site_fe:
        fe = pd.get_dummies(ph[cfg.site_col].astype(str), prefix="SITE", drop_first=True)
        Xcov = pd.concat([Xcov, fe], axis=1)
    return Xcov

def hedges_g(vec: np.ndarray, grp: np.ndarray) -> float:
    a, b = vec[grp==1], vec[grp==0]
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    sa2, sb2 = a.var(ddof=1), b.var(ddof=1)
    sp = np.sqrt(((na-1)*sa2 + (nb-1)*sb2) / max(na+nb-2, 1))
    if sp == 0: return 0.0
    g = (a.mean()-b.mean())/sp
    J = 1 - 3/(4*(na+nb)-9)
    return float(J*g)

def mass_univariate_glm(X: np.ndarray, ph: pd.DataFrame, cfg: DXConfig, site_fe: bool=False) -> Dict[str, np.ndarray]:
    Xcov = _design_matrix(ph, cfg, site_fe)
    Xcov_np = Xcov.values
    beta_dx = np.zeros(X.shape[1], dtype=float)
    pvals = np.ones(X.shape[1], dtype=float)
    for j in range(X.shape[1]):
        y = X[:, j]
        try:
            mod = sm.OLS(y, Xcov_np).fit()
            beta_dx[j] = mod.params[Xcov.columns.get_loc("DX")]
            pvals[j]   = mod.pvalues[Xcov.columns.get_loc("DX")]
        except Exception:
            beta_dx[j] = np.nan
            pvals[j]   = 1.0
    rej, qvals, *_ = multipletests(pvals, alpha=cfg.fdr_q, method="fdr_bh")
    g_vals = np.apply_along_axis(lambda v: hedges_g(v, ph["DX_BIN"].values), 0, X)
    return {"beta_dx": beta_dx, "pvals": pvals, "qvals": qvals, "sig_mask": rej, "g": g_vals}
