from __future__ import annotations
import numpy as np, pandas as pd
from typing import Dict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from dx_config import DXConfig

def _clf_pipeline(cfg: DXConfig) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("pca", PCA(n_components=cfg.n_pca, random_state=0)),
        ("lr", LogisticRegression(max_iter=3000, solver="saga", C=0.5, class_weight="balanced"))
    ])

def classify_stratified(X: np.ndarray, y: np.ndarray, cfg: DXConfig, n_splits: int=5) -> Dict[str, float]:
    pipe = _clf_pipeline(cfg)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1)
    ys = []; pr = []; pd_ = []
    for tr, te in skf.split(X, y):
        pipe.fit(X[tr], y[tr])
        pd_.extend(pipe.predict(X[te]))
        pr.extend(pipe.predict_proba(X[te])[:,1])
        ys.extend(y[te])
    ys, pr, pd_ = np.array(ys), np.array(pr), np.array(pd_)
    return {"AUC": float(roc_auc_score(ys, pr)),
            "ACC": float(accuracy_score(ys, pd_)),
            "F1":  float(f1_score(ys, pd_))}

def classify_loso(X: np.ndarray, y: np.ndarray, site: np.ndarray, cfg: DXConfig) -> Dict[str, float]:
    pipe = _clf_pipeline(cfg)
    gkf = GroupKFold(n_splits=len(np.unique(site)))
    aucs=[]; accs=[]; f1s=[]
    for tr, te in gkf.split(X, y, groups=site.astype(str)):
        pipe.fit(X[tr], y[tr])
        p = pipe.predict(X[te])
        pr = pipe.predict_proba(X[te])[:,1]
        aucs.append(roc_auc_score(y[te], pr))
        accs.append(accuracy_score(y[te], p))
        f1s.append(f1_score(y[te], p))
    return {"AUC": float(np.mean(aucs)),
            "ACC": float(np.mean(accs)),
            "F1":  float(np.mean(f1s))}
