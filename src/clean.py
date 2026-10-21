"""
Cleaning for the churn project.

Two classic traps are handled explicitly:

1. **TotalCharges is stored as text** and contains 11 empty cells. Those cells
   belong to `tenure = 0` customers (they have been subscribed for exactly one
   billing cycle). The correct value is `MonthlyCharges * 1`, i.e. the amount
   charged in their first month. We fill it with `MonthlyCharges` rather than
   `0` so the monotonic rule `TotalCharges ≈ MonthlyCharges × tenure` holds.
2. **SeniorCitizen is 0/1 but is really Yes/No** — it is re-labelled so the
   report is legible.
"""
from __future__ import annotations

import logging

import pandas as pd

from .config import DATASETS

import numpy as np

logger = logging.getLogger(__name__)

NUMERIC_COLUMNS = [
    "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
    "HasCrCard", "IsActiveMember", "EstimatedSalary",
]


def clean(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Return a cleaned copy; the input is not modified."""
    spec = DATASETS[dataset]
    out = df.copy()

    # --- 1. numeric columns that landed as text --------------------------
    for col in NUMERIC_COLUMNS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    # --- 2. the TotalCharges text / blank trap ---------------------------
    if "TotalCharges" in out.columns and spec.get("total_revenue"):
        out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce")
        blank = out["TotalCharges"].isna()
        if blank.any():
            logger.warning(
                "Filled %d blank TotalCharges cells with the first month's "
                "charge (MonthlyCharges) for tenure=0 customers.", int(blank.sum())
            )
            out.loc[blank, "TotalCharges"] = pd.to_numeric(
                out.loc[blank, "MonthlyCharges"], errors="coerce"
            )

    # --- 3. SeniorCitizen 0/1 -> Yes/No -----------------------------------
    if "SeniorCitizen" in out.columns:
        _map = {"0": "No", "1": "Yes", 0: "No", 1: "Yes"}
        out["SeniorCitizen"] = out["SeniorCitizen"].map(_map)

    # --- 4. drop the raw ID column (kept for the record) -------------------
    if spec.get("id") and spec["id"] in out.columns:
        out = out.drop(columns=[spec["id"]])

    # --- 5. yes/no / categorical string columns -> 0/1 / codes -------------
    object_cols = [c for c in out.columns
                   if out[c].dtype == "object"
                   or isinstance(out[c].dtype, pd.StringDtype)]
    for col in object_cols:
        if col in (spec["target"], "churn"):
            continue
        vals = [v for v in out[col].unique() if isinstance(v, str)]
        if not vals:
            continue
        yes_no = {v.lower() for v in vals}
        if yes_no <= {"yes", "no", "1", "0", "true", "false", ""}:
            # 2-class categorical -> 0/1
            lab = {"yes": 1, "no": 0, "true": 1, "false": 0, "1": 1, "0": 0}
            out[col] = out[col].str.lower().map(lab).astype("Int64")
        else:
            # try numeric first: tenure / MonthlyCharges / TotalCharges may be
            # stored as text but are actually continuous variables
            nums = pd.to_numeric(out[col], errors="coerce")
            if len(nums) == len(out) and nums.notna().all():
                out[col] = nums
            elif len(vals) <= 4:
                out[col] = pd.Categorical(out[col]).codes
            else:
                # high-cardinality non-numeric (e.g. surname, RowNumber) -> drop
                out = out.drop(columns=[col])

    return out
