"""
Stage - Power BI export (churn project).

Writes a Power BI-ready data pack into `powerbi/`:

  powerbi/
    data/dim_customer.csv          grain = one row per customer
                                   (churn label, churn_risk, survival prob)
    data/dim_tenure.csv            tenure dimension (month -> band) + churn rate
    data/kpi_churn_by_segment.csv  churn rate by segment category
    data/kpi_model_comparison.csv  ROC / PR / precision / recall per model
    data/kpi_survival_summary.csv  Kaplan-Meier summary by segment
    data/kpi_cox_hazards.csv       hazard ratios + confidence intervals
    data/kpi_business_impact.csv   revenue at risk for the top-`cap` riskers
    data/kpi_targeting_lift.csv    churn rate by size of the contacted list
    data/connection.txt            exact connection options

Power BI connects either by importing these CSVs (portable, no driver needed)
or by importing the Excel workbook. Both paths are documented in
`powerbi/README.md`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import churn_by_segment, churn_by_tenure_band
from .clean import clean
from .config import DATASETS, Paths
from .impact import revenue_at_risk, risk_score, targeting_lift
from .load_data import load_raw
from .model import fit_models
from .survival import (
    HORIZONS,
    cox_model,
    cox_table,
    km_by_group,
    km_overall,
    risk_score_at,
    survival_summary,
)

# The churn "time" axis is tenure, not calendar time. A calendar dim_date would
# be empty here, so the date-table analogue is a tenure dimension.
TENURE_BINS = [0, 12, 24, 36, 48, 60, 72]
TENURE_LABELS = ["0-12", "12-24", "24-36", "36-48", "48-60", "60-72"]


def _write(df: pd.DataFrame, path: Path) -> int:
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return len(df)


def _dim_customer(raw: pd.DataFrame, dataset: str, models, cph) -> pd.DataFrame:
    """One row per customer: raw labels + model risk + survival probability.

    Built from the **raw** frame on purpose: ``clean()`` turns Contract,
    PaymentMethod, ... into integer codes, which is right for a model but
    illegible in a Power BI slicer. The labels stay readable here.
    """
    spec = DATASETS[dataset]
    risked = risk_score(raw, dataset, models["logistic"])
    horizon = HORIZONS[dataset][1]
    surv = risk_score_at(cph, raw, dataset, horizon)

    dim = raw.copy()
    # numeric coercion (the raw frame keeps tenure / charges as text)
    for c in (spec["tenure"], spec.get("monthly_revenue"), spec.get("total_revenue")):
        if c and c in dim.columns:
            dim[c] = pd.to_numeric(dim[c], errors="coerce")
    if spec.get("total_revenue") and spec.get("monthly_revenue"):
        tr, mr = spec["total_revenue"], spec["monthly_revenue"]
        dim[tr] = dim[tr].fillna(dim[mr])  # the TotalCharges blank trap

    dim["churn"] = risked["churn"].values
    dim["churn_label"] = np.where(dim["churn"] == 1, "Churned", "Active")
    dim["churn_risk_score"] = risked["churn_risk"].values
    dim["surv_prob_at_horizon"] = surv.reindex(dim.index).values
    dim["tenure_band"] = pd.cut(
        dim[spec["tenure"]], bins=TENURE_BINS, labels=TENURE_LABELS, right=False
    ).astype(str)
    return dim


def _dim_tenure(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Tenure dimension: month -> band, with churn rate and survival prob."""
    spec = DATASETS[dataset]
    t = pd.to_numeric(df[spec["tenure"]], errors="coerce")
    band = pd.cut(t, bins=TENURE_BINS, labels=TENURE_LABELS, right=False)
    per_month = (
        pd.DataFrame({"tenure_month": t, "tenure_band": band, "churn": df["churn"]})
        .dropna(subset=["tenure_month"])
        .groupby(["tenure_month", "tenure_band"], observed=True)["churn"]
        .agg(customers="size", churners="sum")
        .reset_index()
    )
    per_month["churn_rate_pct"] = (
        per_month["churners"] / per_month["customers"] * 100
    ).round(2)
    # survival probability from the overall Kaplan-Meier curve
    kmf = km_overall(df, dataset)
    per_month["survival_prob"] = per_month["tenure_month"].apply(
        lambda m: round(float(kmf.predict(m)), 4)
    )
    return per_month


def _write_connection(paths: Paths) -> str:
    lines = [
        "# Power BI connection - Customer Churn Analysis",
        "#",
        "# Option A - import the CSVs in powerbi/data/ (recommended, portable)",
        "Folder: " + (paths.root / "powerbi" / "data").as_posix(),
        "",
        "# Option B - import the Excel workbook",
        "File:   " + (paths.root / "reports" / "Churn_Analysis.xlsx").as_posix(),
        "",
        "# No driver / no database needed for either option.",
        "# See powerbi/README.md for the full build instructions.",
    ]
    path = paths.root / "powerbi" / "data" / "connection.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def run(paths: Paths | None = None, dataset: str = "telco") -> dict:
    """Write the Power BI data pack. Returns {table: rows}."""
    paths = paths or Paths()
    out = paths.root / "powerbi" / "data"
    out.mkdir(parents=True, exist_ok=True)

    spec = DATASETS[dataset]
    raw = load_raw(paths, dataset)
    df = clean(raw, dataset)
    models, results = fit_models(df, dataset, seed=42)
    # survival works from the RAW frame (see report.make_figures for why)
    cph = cox_model(raw, dataset)

    counts: dict = {"dataset": dataset, "customers": len(df)}

    counts["dim_customer"] = _write(
        _dim_customer(raw, dataset, models, cph), out / "dim_customer.csv"
    )
    counts["dim_tenure"] = _write(
        _dim_tenure(df, dataset), out / "dim_tenure.csv"
    )
    counts["kpi_churn_by_segment"] = _write(
        churn_by_segment(raw, dataset), out / "kpi_churn_by_segment.csv"
    )
    counts["kpi_tenure_bands"] = _write(
        churn_by_tenure_band(df, dataset, bins=TENURE_BINS),
        out / "kpi_tenure_bands.csv",
    )

    # model comparison (flatten the logistic importance list for CSV)
    mc = results.drop(columns=["feature_importance_df"], errors="ignore").copy()
    mc["feature_importance"] = mc["feature_importance"].apply(
        lambda v: " | ".join(v) if isinstance(v, list) else v
    )
    counts["kpi_model_comparison"] = _write(mc, out / "kpi_model_comparison.csv")

    # survival
    # survival: raw frame keeps the segment labels readable
    kmf_all = km_overall(raw, dataset)
    seg_col = "Contract" if dataset == "telco" else "Geography"
    curves = km_by_group(raw, dataset, seg_col)
    surv_rows = [{"group": "All customers", **survival_summary(kmf_all, HORIZONS[dataset])}]
    surv_rows += [
        {"group": f"{seg_col} = {lab}", **survival_summary(k, HORIZONS[dataset])}
        for lab, k in curves.items()
    ]
    counts["kpi_survival_summary"] = _write(
        pd.DataFrame(surv_rows), out / "kpi_survival_summary.csv"
    )

    cox = cox_table(cph)
    cox.insert(0, "concordance", round(float(cph.concordance_index_), 4))
    counts["kpi_cox_hazards"] = _write(cox, out / "kpi_cox_hazards.csv")

    # business impact
    counts["kpi_business_impact"] = _write(
        pd.DataFrame([revenue_at_risk(df, dataset, models["logistic"], cap=500)]),
        out / "kpi_business_impact.csv",
    )
    counts["kpi_targeting_lift"] = _write(
        targeting_lift(risk_score(df, dataset, models["logistic"]),
                       models["logistic"], cap=500),
        out / "kpi_targeting_lift.csv",
    )

    counts["connection"] = 1
    _write_connection(paths)
    return counts


if __name__ == "__main__":
    for name, count in run().items():
        print(f"{name}: {count}")
