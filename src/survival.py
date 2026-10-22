"""
Survival analysis — time-to-churn.

The classification approach answers *"will this customer churn?"*. Survival
analysis answers the question the business actually asks: *"how long do they
stay, and what makes them leave sooner?"*

Setup
-----
The dataset is a snapshot, so this is a standard **right-censoring** problem:

* ``churn = 1``  -> the event was observed: the customer left at ``tenure``
* ``churn = 0``  -> censored: the customer was still active at ``tenure``

That distinction matters. Censored customers have systematically *longer* tenure
than churners (they survived), so any method that ignores censoring is biased.
Kaplan-Meier and Cox handle it correctly.

This module works from the **raw** frame on purpose: ``clean()`` turns
categoricals into integer codes, which is right for a classifier but destroys
the labels needed to read a survival curve by segment. Here we keep the labels
and one-hot encode internally for the Cox model.

What is in this module
----------------------
* ``km_overall``      - Kaplan-Meier survival curve for the whole base
* ``km_by_group``     - one curve per category of a segment (Contract, ...)
* ``cox_model``       - Cox proportional hazards -> hazard ratios + p-values
* ``logrank``         - statistical test: do two groups survive differently?
* ``survival_summary``- median survival + survival at key horizons
* ``conditional_survival`` - P(stay to t1 | already stayed to t0)
* ``risk_score_at``   - per-customer survival probability at a horizon

Honest caveats applied
----------------------
* ``TotalCharges`` is **excluded** from the Cox covariates: it is approximately
  ``MonthlyCharges x tenure``, i.e. a function of the time variable itself, not
  a baseline covariate. Including it would be a methodological error.
* Median survival is reported as ``None`` when the curve never reaches 50%
  within the observation window — that is a finding, not a bug.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test

from .config import DATASETS

# Covariates for the Cox model, per dataset. TotalCharges is deliberately
# absent for telco (it is a function of the time variable -- see module docstring).
COX_COVARIATES = {
    "telco": ["MonthlyCharges", "Contract", "InternetService", "PaymentMethod",
              "gender", "SeniorCitizen", "Partner", "Dependents",
              "PaperlessBilling", "OnlineSecurity", "TechSupport"],
    "bank": ["CreditScore", "Age", "Balance", "NumOfProducts",
             "IsActiveMember", "HasCrCard", "EstimatedSalary",
             "Geography", "Gender"],
}

# Horizons (in the dataset's time unit) at which survival is reported.
HORIZONS = {"telco": [6, 12, 24, 36, 48, 60, 72], "bank": [1, 2, 3, 5, 7, 10]}


# ---------------------------------------------------------------------------
# preparation
# ---------------------------------------------------------------------------
def _event_series(df: pd.DataFrame, dataset: str) -> pd.Series:
    """Map the dataset's target column to a 0/1 event indicator."""
    spec = DATASETS[dataset]
    raw = df[spec["target"]]
    pos = str(spec["positive"]).strip().lower()
    as_text = raw.astype(str).str.strip().str.lower()
    return (as_text == pos).astype(int)


def _survival_frame(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Return a frame with ``time`` / ``event`` plus the raw covariates."""
    spec = DATASETS[dataset]
    out = df.copy()
    out = out.rename(columns={spec["tenure"]: "time"})
    out["time"] = pd.to_numeric(out["time"], errors="coerce")
    out["event"] = _event_series(df, dataset)
    out = out.dropna(subset=["time", "event"])
    return out


def _cox_frame(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Design matrix for the Cox model: numeric + one-hot categoricals.

    ``time`` is coerced to numeric **before** the categorical check. Without
    that, a text ``tenure`` (raw frame) is mistaken for a category and one-hot
    encoded into one dummy per month, which silently destroys the duration.
    """
    spec = DATASETS[dataset]
    d = _survival_frame(df, dataset)
    wanted = ["time", "event"] + COX_COVARIATES[dataset]
    cols = [c for c in wanted if c in d.columns]
    d = d[cols].copy()

    # duration + event must be numeric before the categorical pass
    for col in ("time", "event"):
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce")

    for col in d.columns:
        if col in ("time", "event"):
            continue
        nums = pd.to_numeric(d[col], errors="coerce")
        if nums.notna().all():
            d[col] = nums

    cat = [c for c in d.columns if not pd.api.types.is_numeric_dtype(d[c])]
    if cat:
        d = pd.get_dummies(d, columns=cat, drop_first=True, dtype=float)
    return d.dropna()


# ---------------------------------------------------------------------------
# estimators
# ---------------------------------------------------------------------------
def km_overall(df: pd.DataFrame, dataset: str) -> KaplanMeierFitter:
    """Kaplan-Meier estimate of the survival function for the whole base."""
    d = _survival_frame(df, dataset)
    kmf = KaplanMeierFitter()
    kmf.fit(d["time"], d["event"], label="all customers")
    return kmf


def km_by_group(df: pd.DataFrame, dataset: str, group_col: str) -> dict:
    """One Kaplan-Meier curve per category of ``group_col`` (labels kept)."""
    d = _survival_frame(df, dataset)
    out = {}
    for value, sub in d.groupby(group_col):
        kmf = KaplanMeierFitter()
        kmf.fit(sub["time"], sub["event"], label=str(value))
        kmf.n_obs = int(len(sub))
        out[str(value)] = kmf
    return out


def cox_model(df: pd.DataFrame, dataset: str) -> CoxPHFitter:
    """Cox proportional-hazards model on the dataset's covariates."""
    d = _cox_frame(df, dataset)
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(d, duration_col="time", event_col="event")
    return cph


def logrank(df: pd.DataFrame, dataset: str, group_col: str) -> dict:
    """Log-rank test across groups."""
    d = _survival_frame(df, dataset)
    groups = {str(v): g for v, g in d.groupby(group_col)}
    names = list(groups)
    results = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = groups[names[i]], groups[names[j]]
            res = logrank_test(a["time"], b["time"],
                               event_observed_A=a["event"],
                               event_observed_B=b["event"])
            results[f"{names[i]} vs {names[j]}"] = {
                "test_statistic": round(float(res.test_statistic), 4),
                "p_value": float(res.p_value),
            }
    return results


# ---------------------------------------------------------------------------
# summaries
# ---------------------------------------------------------------------------
def survival_summary(kmf: KaplanMeierFitter, horizons: list[int]) -> dict:
    """Median survival + survival probability at each horizon."""
    median = kmf.median_survival_time_
    median = None if median == np.inf else float(median)
    at = {f"survival_at_{h}": round(float(kmf.predict(h)), 4) for h in horizons}
    return {"median_survival_time": median, **at}


def conditional_survival(kmf: KaplanMeierFitter, t0: int, t1: int) -> float:
    """P(survive to t1 | already survived to t0)."""
    s0 = kmf.predict(t0)
    s1 = kmf.predict(t1)
    return round(float(s1 / s0), 4) if s0 > 0 else None


def cox_table(cph: CoxPHFitter) -> pd.DataFrame:
    """Hazard ratios with confidence intervals and p-values."""
    summ = cph.summary
    out = pd.DataFrame({
        "covariate": summ.index,
        "coef": summ["coef"].values,
        "hazard_ratio": summ["exp(coef)"].values,
        "hr_lower_95": summ["exp(coef) lower 95%"].values,
        "hr_upper_95": summ["exp(coef) upper 95%"].values,
        "p_value": summ["p"].values,
    })
    return out.sort_values("hazard_ratio", ascending=False)


def risk_score_at(cph: CoxPHFitter, df: pd.DataFrame, dataset: str,
                  horizon: int) -> pd.Series:
    """Per-customer probability of surviving to ``horizon`` (1 = safe)."""
    d = _cox_frame(df, dataset)
    sf = cph.predict_survival_function(d, times=[horizon])
    return pd.Series(sf.iloc[0].values, index=d.index, name=f"surv_prob_{horizon}")
