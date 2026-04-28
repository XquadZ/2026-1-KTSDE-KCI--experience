from __future__ import annotations

from typing import Dict, Optional

import pandas as pd
from scipy.stats import pearsonr, spearmanr


def safe_spearman(x: pd.Series, y: pd.Series) -> Dict[str, float]:
    tmp = pd.concat([x, y], axis=1).dropna()
    if len(tmp) < 2:
        return {"rho": float("nan"), "p_value": float("nan"), "n": len(tmp)}
    rho, p = spearmanr(tmp.iloc[:, 0], tmp.iloc[:, 1])
    return {"rho": float(rho), "p_value": float(p), "n": len(tmp)}


def safe_pearson(x: pd.Series, y: pd.Series) -> Dict[str, float]:
    tmp = pd.concat([x, y], axis=1).dropna()
    if len(tmp) < 2:
        return {"r": float("nan"), "p_value": float("nan"), "n": len(tmp)}
    r, p = pearsonr(tmp.iloc[:, 0], tmp.iloc[:, 1])
    return {"r": float(r), "p_value": float(p), "n": len(tmp)}


def ranking_table(df: pd.DataFrame, score_col: str = "S", combo_col: str = "Combo") -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values(score_col, ascending=False).reset_index(drop=True)
    out.insert(0, "Rank", out.index + 1)
    cols = [c for c in ["Rank", combo_col, score_col, "GainAP", "GainAR"] if c in out.columns]
    return out[cols + [c for c in out.columns if c not in cols]]


def correlation_report(
    df: pd.DataFrame,
    score_col: str = "S",
    gain_ap_col: Optional[str] = "GainAP",
    gain_ar_col: Optional[str] = "GainAR",
) -> pd.DataFrame:
    rows = []
    if gain_ap_col and gain_ap_col in df.columns:
        s1 = safe_spearman(df[score_col], df[gain_ap_col])
        p1 = safe_pearson(df[score_col], df[gain_ap_col])
        rows.append(
            {
                "target": gain_ap_col,
                "spearman_rho": s1["rho"],
                "spearman_p": s1["p_value"],
                "pearson_r": p1["r"],
                "pearson_p": p1["p_value"],
                "n": s1["n"],
            }
        )
    if gain_ar_col and gain_ar_col in df.columns:
        s2 = safe_spearman(df[score_col], df[gain_ar_col])
        p2 = safe_pearson(df[score_col], df[gain_ar_col])
        rows.append(
            {
                "target": gain_ar_col,
                "spearman_rho": s2["rho"],
                "spearman_p": s2["p_value"],
                "pearson_r": p2["r"],
                "pearson_p": p2["p_value"],
                "n": s2["n"],
            }
        )
    return pd.DataFrame(rows)
