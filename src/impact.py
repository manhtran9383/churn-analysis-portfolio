"""
Business impact: turns model output into money.

Two views:
  * **Revenue at risk** — monthly revenue lost if the top-`cap` churn-risk
    customers leave (baseline = everyone equally likely to churn).
  * **Targeting lift** — if the company can only serve `cap` at-risk
    customers, does prioritising the ranked list beat serving everyone?
"""
from __future__ import annotations

import pandas as pd

from .config import DATASETS
from .clean import clean


def risk_score(df: pd.DataFrame, dataset: str, model) -> pd.DataFrame:
    """Attach a churn-probability score to each customer."""
    spec = DATASETS[dataset]
    df = clean(df, dataset)
    prepared = _infer_features(df, dataset)
    X = prepared["X"][prepared["usable"]]
    proba = model.predict_proba(X)[:, 1]
    out = X.copy()
    out["churn_risk"] = proba
    out["churn"] = prepared["y"].values
    return out


def targeting_lift(df: pd.DataFrame, model, cap: int) -> pd.DataFrame:
    """Churn-rate of the top-`cap` riskers vs the rest of the base."""
    df = df.sort_values("churn_risk", ascending=False).reset_index(drop=True)
    n = len(df)
    rows = []
    for size in [0.02, 0.05, 0.10, 0.20, 0.5, 1.0]:
        m = min(int(size * n), n)
        mask = pd.Series(True, index=df.index)
        if m < n:
            mask.iloc[m:] = False
        seg = df[mask]
        rows.append({
            "list_size": m,
            "churners": int(seg["churn"].sum()),
            "customers": int(len(seg)),
            "churn_rate_pct": round(float(seg["churn"].mean()) * 100, 2),
            "avg_risk": round(float(seg["churn_risk"].mean()), 4),
        })
    out = pd.DataFrame(rows)
    base_rate = float(df["churn"].mean()) * 100
    out["lift_vs_all_pp"] = (out["churn_rate_pct"] - base_rate).round(2)
    return out


def revenue_at_risk(df: pd.DataFrame, dataset: str, model, cap: int) -> dict:
    """Estimated monthly revenue lost if the top-`cap` riskers leave."""
    spec = DATASETS[dataset]
    prepared = _infer_features(df, dataset)
    X = prepared["X"][prepared["usable"]]
    y = prepared["y"]
    proba = model.predict_proba(X)[:, 1]
    out = X.copy()
    out["churn_risk"] = proba
    out["churn"] = y.values

    if spec.get("monthly_revenue"):
        rev = pd.to_numeric(out[spec["monthly_revenue"]], errors="coerce")
        rev = rev.fillna(rev.median())
        out["monthly_revenue"] = rev
        top = out.sort_values("churn_risk", ascending=False).head(cap)
        loss = float(top["monthly_revenue"].sum())
        return {
            "cap": cap,
            "customers_at_risk": int(min(cap, len(out))),
            "revenue_at_risk": round(loss, 2),
            "estimated_monthly_profit_lost": round(loss * 0.35, 2),
            "avg_risk_of_top_cap": round(float(
                out.sort_values("churn_risk", ascending=False).head(cap)["churn_risk"].mean()), 4),
        }
    return {
        "cap": cap,
        "customers_at_risk": int(min(cap, len(out))),
        "revenue_at_risk": None,
        "estimated_monthly_profit_lost": None,
        "avg_risk_of_top_cap": round(float(
            out.sort_values("churn_risk", ascending=False).head(cap)["churn_risk"].mean()), 4),
    }


def _infer_features(df: pd.DataFrame, dataset: str) -> dict:
    from .model import _infer_features as _f
    return _f(df, dataset)
