from __future__ import annotations

import pandas as pd


def minmax(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    lo, hi = s.min(), s.max()
    if hi - lo == 0:
        return pd.Series([0.0] * len(s), index=s.index)
    return (s - lo) / (hi - lo)


def compute_score_s(gain_miss: float, dis: float, ufp: float, comp: float, tfc: float) -> float:
    """
    논문 식(9):
    S = Gain_miss + Dis + Comp - UFP - TFC
    (정규화 이후 값 기준)
    """
    return float(gain_miss) + float(dis) + float(comp) - float(ufp) - float(tfc)


def add_score_columns(
    df: pd.DataFrame,
    gain_col: str = "Gain_miss",
    dis_col: str = "Dis",
    ufp_col: str = "UFP",
    comp_col: str = "Comp",
    tfc_col: str = "TFC",
) -> pd.DataFrame:
    out = df.copy()
    out["n_Gain_miss"] = minmax(out[gain_col])
    out["n_Dis"] = minmax(out[dis_col])
    out["n_UFP"] = minmax(out[ufp_col])
    out["n_Comp"] = minmax(out[comp_col])
    out["n_TFC"] = minmax(out[tfc_col])
    out["S"] = (
        out["n_Gain_miss"]
        + out["n_Dis"]
        + out["n_Comp"]
        - out["n_UFP"]
        - out["n_TFC"]
    )
    return out
