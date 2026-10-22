"""
Reporting: PNG figures + multi-sheet Excel summary.

Outputs:
  * reports/figures/*.png  - the charts
  * reports/Churn_Analysis.xlsx - sheets for segments, features,
    tenure, model comparison, targeting, business impact
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

from typing import Optional
from .config import DATASETS, Paths
from .analysis import (churn_by_segment, top_features,
                       churn_by_tenure_band)

FIGS = {
    "segments": "Churn_by_segment.png",
    "tenure": "Retention_curve_by_tenure.png",
    "features": "Top_numeric_features.png",
    "models": "Model_comparison.png",
    "importance": "Logistic_feature_importance.png",
    "survival": "Survival_curve.png",
}


def _style_ax(ax):
    ax.set_facecolor("#fafafa")
    for s in ax.spines.values():
        s.set_color("#dddddd")
    ax.tick_params(colors="#444444")


def churn_segment_chart(df_seg: pd.DataFrame, dataset: str, top_n: int = 8, paths: Paths = None):
    spec = DATASETS[dataset]
    labs = [f"{c} = {v}\n({r*100:.1f}%)" for c, v, r in
            zip(df_seg["segment"].head(top_n), df_seg["group"].head(top_n),
                df_seg["churn_rate"].head(top_n))]
    rates = df_seg["churn_rate"].head(top_n)
    colors = ["#d9534f" if r > 0.3 else "#f0ad4e" if r > 0.15 else "#5cb85c"
              for r in rates]
    fig, ax = plt.subplots(figsize=(9, max(3, 0.45 * len(rates))))
    ax.barh(labs, rates, color=colors)
    ax.set_xlabel("Churn rate (%)", color="#444444")
    ax.set_title(f"Churn rate by segment ({spec['label']})", color="#1a1a1a", fontsize=13)
    _style_ax(ax)
    ax.set_xlim(0, min(1.0, max(rates) * 1.3))
    fig.tight_layout()
    if paths:
        fig.savefig(paths.figures / FIGS["segments"], dpi=150)
    else:
        fig.savefig(Path("reports/figures/Churn_by_segment.png"), dpi=150)
    plt.close(fig)


def retention_chart(df_ten: pd.DataFrame, dataset: str, paths: Paths = None,
                    min_count: int = 3):
    """Retention curve by tenure band.

    Bands with fewer than ``min_count`` customers are excluded from the line
    (they are too noisy, e.g. HR 40-45 has n=1) and annotated. The Excel
    workbook keeps the full set.
    """
    spec = DATASETS[dataset]
    time_unit = spec.get("time_unit", "months")
    counts = df_ten["count"]
    keep = counts >= min_count
    rates = df_ten["churn_rate"] * 100

    fig, ax = plt.subplots(figsize=(8, 4.5))
    if keep.any():
        ax.plot(
            df_ten.loc[keep, "tenure_band"].astype(str),
            rates[keep.values], marker="o", color="#d9534f",
        )
        ax.fill_between(
            df_ten.loc[keep, "tenure_band"].astype(str), 0,
            rates[keep.values], color="#d9534f", alpha=0.12,
        )
    # annotate excluded noisy bands
    excl = ~keep
    if excl.any():
        labels = [
            f"{b} (n={int(c)})"
            for b, c in zip(df_ten.loc[excl, "tenure_band"],
                            df_ten.loc[excl, "count"])
        ]
        ax.annotate(", ".join(labels), xy=(0.02, 0.02), xycoords="axes fraction",
                    fontsize=8, color="#888", style="italic")
    ax.set_xlabel(f"Tenure band ({time_unit})", color="#444444")
    ax.set_ylabel("Churn rate (%)", color="#444444")
    ax.set_title("Retention curve - churn falls sharply with tenure (>=3/mo shown)",
                 color="#1a1a1a")
    _style_ax(ax)
    fig.tight_layout()
    if paths:
        fig.savefig(paths.figures / FIGS["tenure"], dpi=150)
    else:
        fig.savefig(Path("reports/figures/Retention_curve_by_tenure.png"), dpi=150)
    plt.close(fig)


from typing import Optional


def features_chart(df_feat: pd.DataFrame, paths: Optional[Paths] = None):
    feat = df_feat.sort_values("abs_corr")
    labels = [f for f in feat["feature"]]
    vals = feat["abs_corr"].values
    fig, ax = plt.subplots(figsize=(8, max(4, 0.4 * len(feat))))
    ax.barh(labels, vals, color="#5bc0de")
    ax.set_xlabel("|correlation| with churn", color="#444444")
    ax.set_title("Top numeric features vs churn", color="#1a1a1a")
    _style_ax(ax)
    fig.tight_layout()
    if paths:
        fig.savefig(paths.figures / FIGS["features"], dpi=150)
    else:
        fig.savefig(Path("reports/figures/Top_numeric_features.png"), dpi=150)
    plt.close(fig)


def model_chart(results: pd.DataFrame, paths: Optional[Paths] = None):
    rows = results[results["model"].isin(["logistic", "gradient_boosting"])]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (metric, name) in zip(axes, [
        ("roc_auc", "ROC AUC"), ("pr_auc", "PR AUC")]):
        vals = rows.set_index("model")[metric]
        ax.bar(vals.index, vals.values, color=["#4472C4", "#ED7D31"])
        ax.set_title(name, color="#1a1a1a")
        ax.set_ylim(0.5, 1.0)
        _style_ax(ax)
        ax.tick_params(axis="x", labelsize=10)
    fig.suptitle("Churn model comparison", color="#1a1a1a", fontsize=14)
    fig.tight_layout()
    if paths:
        fig.savefig(paths.figures / FIGS["models"], dpi=150)
    else:
        fig.savefig(Path("reports/figures/Model_comparison.png"), dpi=150)
    plt.close(fig)


def importance_chart(results: pd.DataFrame, paths: Optional[Paths] = None):
    fi = results[results["model"] == "logistic"]["feature_importance_df"].iloc[0]
    feat = fi.sort_values("importance")
    fig, ax = plt.subplots(figsize=(8, max(4, 0.4 * len(feat))))
    ax.barh(feat["feature"], feat["importance"], color="#A5A5A5")
    ax.set_xlabel("|logistic coefficient|", color="#444444")
    ax.set_title("Logistic feature importance", color="#1a1a1a")
    _style_ax(ax)
    fig.tight_layout()
    if paths:
        fig.savefig(paths.figures / FIGS["importance"], dpi=150)
    else:
        fig.savefig(Path("reports/figures/Logistic_feature_importance.png"), dpi=150)
    plt.close(fig)


def survival_chart(dataset: str, df, paths, group_col: str | None = None):
    """Kaplan-Meier survival curve + return the KM summary rows for Excel."""
    from .survival import HORIZONS, km_by_group, km_overall, survival_summary

    spec = DATASETS[dataset]
    if group_col is None:
        group_col = spec.get("survival_segment", "Department")

    kmf_all = km_overall(df, dataset)
    curves = km_by_group(df, dataset, group_col)

    fig, ax = plt.subplots(figsize=(8, 5))
    kmf_all.plot_survival_function(ax=ax, label="All customers", color="#4472C4")
    for label, k in curves.items():
        k.plot_survival_function(ax=ax, label=f"{group_col} = {label}", alpha=0.7)
    ax.set_xlabel(f"Tenure ({spec.get('time_unit', 'months')})", color="#444444")
    ax.set_ylabel("Survival probability", color="#444444")
    ax.set_title(f"Kaplan-Meier survival curve - {spec['label']}",
                 color="#1a1a1a")
    ax.legend(loc="upper right", fontsize=9)
    _style_ax(ax)
    fig.tight_layout()
    fig.savefig(paths.figures / "Survival_curve.png", dpi=150)
    plt.close(fig)

    rows = []
    for label, k in curves.items():
        label = "All" if label == "all" else f"{group_col} = {label}"
        rows.append({"group": label, **survival_summary(k, HORIZONS[dataset])})
    return rows


def build_excel(dataset: str, segment_df: pd.DataFrame, feature_df: pd.DataFrame,
               tenure_df: pd.DataFrame, models: pd.DataFrame,
               targeting: pd.DataFrame, impact: dict, paths,
               surv_rows: list[dict] | None = None,
               surv_cox: pd.DataFrame | None = None) -> str:
    outp = paths.report_file("Churn_Analysis.xlsx")
    # flatten the logistic feature_importance_df (a DataFrame) into a string list
    # so openpyxl writes it as clean cells instead of a nested table; drop the
    # DataFrame copy since the chart is already rendered
    for row in models.itertuples():
        fi = getattr(row, "feature_importance_df", None)
        if fi is not None and hasattr(fi, "columns"):
            models.at[row.Index, "feature_importance"] = "\n".join(
                f"{r.feature}: {r.importance:.4f}" for r in fi.itertuples()
            )
            models.at[row.Index, "feature_importance_df"] = None

    with pd.ExcelWriter(outp, engine="openpyxl") as writer:
        segment_df.to_excel(writer, sheet_name="churn_by_segment", index=False)
        feature_df.to_excel(writer, sheet_name="top_features", index=False)
        tenure_df.to_excel(writer, sheet_name="tenure_bands", index=False)
        models.to_excel(writer, sheet_name="model_comparison", index=False)
        targeting.to_excel(writer, sheet_name="targeting_lift", index=False)
        impact.to_excel(writer, sheet_name="business_impact", index=False)
        if surv_rows is not None:
            pd.DataFrame(surv_rows).to_excel(writer,
                                              sheet_name="survival_summary",
                                              index=False)
        if surv_cox is not None:
            surv_cox.to_excel(writer, sheet_name="surv_cox", index=False)
    return str(outp)


def make_figures(dataset: str, paths, top_n: int = 8, results: pd.DataFrame | None = None):
    """Generate every figure, one per output slot."""
    from .load_data import load_raw
    from .analysis import churn_by_segment, churn_by_tenure_band, top_features
    from .clean import clean
    from .report import (churn_segment_chart, retention_chart, features_chart,
                         model_chart, importance_chart)

    df = load_raw(paths, dataset)
    ch = clean(df, dataset)
    # raw, not cleaned: clean() replaces Contract / PaymentMethod with integer
    # codes, which is right for a model but illegible in a report.
    seg = churn_by_segment(df, dataset)
    ten = churn_by_tenure_band(ch, dataset)
    feat = top_features(ch, top_n)
    churn_segment_chart(seg, dataset, top_n, paths)
    retention_chart(ten, dataset, paths)
    features_chart(feat, paths)
    if results is not None:
        model_chart(results, paths)
        importance_chart(results, paths)

    # survival analysis -------------------------------------------------
    # NOTE: survival functions take the RAW frame, not the cleaned one.
    # clean() turns Contract / PaymentMethod / ... into integer codes, which
    # would make the Cox model treat them as ordinal and would render the KM
    # segment labels as 0/1/2. Raw keeps the labels and lets _cox_frame()
    # one-hot encode the categoricals properly.
    from .survival import cox_model, cox_table
    cph = cox_model(df, dataset)
    surv_rows = survival_chart(dataset, df, paths)
    surv_cox = cox_table(cph)
    conc = cph.concordance_index_
    print(f"Survival (Cox PH): concordance = {conc:.3f}")

    return seg, ten, feat, surv_rows, surv_cox, conc
