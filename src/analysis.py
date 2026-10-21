"""EDA: churn rate by segment + top contributing features (numbers, no charts)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DATASETS


def _churn_rate(g: pd.DataFrame) -> pd.DataFrame:
    g = g.groupby("segment", dropna=False)
    out = pd.DataFrame({
        "count": g.size(),
        "churners": g["churn"].sum(),
        "churn_rate": g["churn"].mean(),
    }).reset_index().rename(columns={"segment": "group"})
    return out.sort_values("churn_rate", ascending=False)


def _readable_group(segment: str, value) -> str:
    """Relabel the raw SeniorCitizen 0/1 to No/Yes (matches what clean() does)."""
    if segment == "SeniorCitizen":
        return {0: "No", 1: "Yes", "0": "No", "1": "Yes"}.get(value, str(value))
    return str(value)


def churn_by_segment(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Churn rate per category of each segment column.

    Rows carry BOTH the column name (`segment`) and the category value
    (`group`) so a reader can tell that "Yes" came from `Partner` and not
    from `Dependents` -- without that, identical-looking groups are
    indistinguishable in the Excel and in the chart.
    """
    spec = DATASETS[dataset]
    rows = []
    for col in spec["segments"]:
        sub = (df.groupby(col)["churn"]
                 .agg(churn_rate="mean", count="count", churners="sum")
                 .reset_index()
                 .rename(columns={col: "group"}))
        sub["group"] = [_readable_group(col, v) for v in sub["group"]]
        sub.insert(0, "segment", col)
        rows.append(sub)
    out = pd.concat(rows, ignore_index=True)
    return out.sort_values(["churn_rate", "count"], ascending=[False, True])


def top_features(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Correlation of numeric features with the churn target."""
    num = df.select_dtypes(include=[np.number])
    corr = num.corr()["churn"].drop("churn").abs().sort_values(ascending=False)
    return pd.DataFrame({"feature": corr.index, "abs_corr": corr.values}).head(top_n)


def churn_by_tenure_band(df: pd.DataFrame, dataset: str, bins: list[float] | None = None) -> pd.DataFrame:
    """Churn rate as tenure increases (retention curve)."""
    bins = bins or [0, 12, 24, 36, 48, 60, 72]
    df = df.copy()
    tenure_col = DATASETS[dataset]["tenure"]
    df[tenure_col] = pd.to_numeric(df[tenure_col], errors="coerce")
    df = df.dropna(subset=[tenure_col])
    df["tenure_band"] = pd.cut(df[tenure_col], bins=bins, right=False)
    out = (df.groupby("tenure_band", observed=True)["churn"]
             .agg(["mean", "count", "sum"])
             .rename(columns={"mean": "churn_rate", "count": "count", "sum": "churners"})
             .reset_index())
    return out
