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


def targeting_lift(df: pd.DataFrame, cap: int | None = None) -> pd.DataFrame:
    """Churn-rate of the top-`cap` riskers vs the rest of the base.

    Keeps the natural sizes of the base (test set) exactly as produced by the
    pipeline: 2 %, 5 %, 10 %, 20 %, 50 % and 100 % (baseline). No clamping and no
    padded row, so the CSV is always 1:1 with the underlying data.

    The optional ``cap`` appears only once, and only when it is a real slice of
    the base (cap <= n) and has not already a natural size there, so telco and
    bank each emit a single, clean 500 row. No duplicate rows and no fabricated
    "top-500" row when the test set is smaller than cap.
    """
    df = df.sort_values("churn_risk", ascending=False).reset_index(drop=True)
    n = len(df)
    rows = []
    for size in [0.02, 0.05, 0.10, 0.20, 0.5]:
        m = int(size * n)
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
    # baseline = whole base, 100 % of customers
    seg = df
    rows.append({
        "list_size": n,
        "churners": int(seg["churn"].sum()),
        "customers": int(len(seg)),
        "churn_rate_pct": round(float(seg["churn"].mean()) * 100, 2),
        "avg_risk": round(float(seg["churn_risk"].mean()), 4),
    })
    out = pd.DataFrame(rows)
    base_rate = float(df["churn"].mean()) * 100
    out["lift_vs_all_pp"] = (out["churn_rate_pct"] - base_rate).round(2)

    # optional real cap slice: only when cap <= n (a genuine slice) and not
    # already present as a natural size.
    if cap is not None and cap > 0 and cap <= n and not any(r["list_size"] == cap for r in rows):
        mask = pd.Series(True, index=df.index)
        if cap < n:
            mask.iloc[cap:] = False
        seg = df[mask]
        rows.append({
            "list_size": cap,
            "churners": int(seg["churn"].sum()),
            "customers": int(len(seg)),
            "churn_rate_pct": round(float(seg["churn"].mean()) * 100, 2),
            "avg_risk": round(float(seg["churn_risk"].mean()), 4),
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
