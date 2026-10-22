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


def targeting_lift(df: pd.DataFrame, cap: int = None) -> pd.DataFrame:
    """Churn-rate of the top-`cap` riskers vs the rest of the base."""
    df = df.sort_values("churn_risk", ascending=False).reset_index(drop=True)
    n = len(df)
    rows = []
    for size in [0.02, 0.05, 0.10, 0.20, 0.5, 1.0]:
        m = min(int(size * n), n)
        if cap is not None:
            m = min(m, cap)
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
    # Always include the requested cap as a row so consumers can always find
    # "top-500" (or whatever cap was asked for), even if the test set is smaller.
    if cap is not None and not any(r["list_size"] == cap for r in rows) and cap > 0:
        mask = pd.Series(True, index=df.index)
        if cap < len(df):
            mask.iloc[cap:] = False
        seg = df[mask]
        rows.append({
            "list_size": cap,
            "churners": int(seg["churn"].sum()),
            "customers": int(len(seg)),
            "churn_rate_pct": round(float(seg["churn"].mean()) * 100, 2),
            "avg_risk": round(float(seg["churn_risk"].mean()), 4),
            "lift_vs_all_pp": round(
                float(seg["churn"].mean() * 100) - base_rate, 2),
        })
    out = pd.DataFrame(rows)
    out["lift_vs_all_pp"] = (out["churn_rate_pct"] - base_rate).round(2)
    return out


def revenue_at_risk(df: pd.DataFrame, dataset: str, model, cap: int = 500) -> dict:
    """Estimated monthly revenue lost if the top-`cap` riskers leave.

    Expects `df` to already have `churn_risk` and `churn` columns from `risk_score`.
    """
    from .config import load_config
    cfg = load_config()
    margin = cfg.get("impact", {}).get("margin_pct", 0.35)
    spec = DATASETS[dataset]
    if spec.get("monthly_revenue"):
        rev = pd.to_numeric(df[spec["monthly_revenue"]], errors="coerce")
        rev = rev.fillna(rev.median())
        df = df.copy()
        df["monthly_revenue"] = rev
        top = df.sort_values("churn_risk", ascending=False).head(cap)
        loss = float(top["monthly_revenue"].sum())
        return {
            "cap": cap,
            "customers_at_risk": int(min(cap, len(df))),
            "revenue_at_risk": round(loss, 2),
            "estimated_monthly_profit_lost": round(loss * margin, 2),
            "avg_risk_of_top_cap": round(float(
                df.sort_values("churn_risk", ascending=False).head(cap)["churn_risk"].mean()), 4),
        }
    return {
        "cap": cap,
        "customers_at_risk": int(min(cap, len(df))),
        "revenue_at_risk": None,
        "estimated_monthly_profit_lost": None,
        "avg_risk_of_top_cap": round(float(
            df.sort_values("churn_risk", ascending=False).head(cap)["churn_risk"].mean()), 4),
    }


def _infer_features(df: pd.DataFrame, dataset: str) -> dict:
    from .model import _infer_features as _f
    return _f(df, dataset)
