"""
Entry point: Customer Churn Analysis pipeline.

End-to-end run:
  1. load + profile each dataset
  2. clean (TotalCharges trap, label encoding)
  3. EDA (segment churn, retention by tenure, top features)
  4. model (logistic + gradient boosting, identical split)
  5. business impact (revenue at risk, targeting lift)
  6. survival analysis (Kaplan-Meier, Cox PH, right-censoring)
  7. figures + Excel report

Stages:
  python -m src.run_churn_analysis --dataset telco            # full run
  python -m src.run_churn_analysis --dataset telco --stage powerbi   # data pack CSVs
  python -m src.run_churn_analysis --dataset telco --stage preview   # HTML preview

Usage:
  python -m src.run_churn_analysis           # telco deep-dive (default)
  python -m src.run_churn_analysis --dataset telco
  python -m src.run_churn_analysis --dataset bank
"""
from __future__ import annotations

import logging

import pandas as pd

from .config import DATASETS, load_config, Paths
from .clean import clean
from .load_data import load_raw, profile, save_profile

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("churn-analysis")


def _load_and_clean(dataset: str, paths) -> pd.DataFrame:
    log.info("Loading %s …", DATASETS[dataset]["label"])
    df = load_raw(paths, dataset)
    prof = profile(df, DATASETS[dataset])
    prof_path = save_profile(paths, dataset, prof)
    log.info("profile → %s", prof_path)
    log.info("rows=%d cols=%d churn_rate=%.2f%%",
             prof["rows"], prof["columns"], prof["churn_rate"])
    return clean(df, dataset), prof


def run(dataset: str = "telco"):
    cfg = load_config()
    paths = Paths(cfg)
    log.info("Dataset: %s", DATASETS[dataset]["label"])

    # 1. load + clean -------------------------------------------------
    df, prof = _load_and_clean(dataset, paths)
    from .load_data import load_raw
    df_raw = load_raw(paths, dataset)

    # 2. EDA ---------------------------------------------------------
    from .analysis import churn_by_segment, top_features, churn_by_tenure_band
    seg = churn_by_segment(df_raw, dataset)
    feat = top_features(df, cfg["reporting"]["top_n"])
    ten = churn_by_tenure_band(df, dataset, bins=[0, 12, 24, 36, 48, 60, 72])

    # 3. model -------------------------------------------------------
    from .model import fit_models
    models, results = fit_models(df, dataset, seed=cfg["model"]["random_state"])
    log.info("models saved → %s", paths.reports)

    # 4. business impact ----------------------------------------------
    from .impact import risk_score, targeting_lift, revenue_at_risk
    risked = risk_score(df, dataset, models["logistic"])
    tl = targeting_lift(risked, models["logistic"], cap=500)
    impact = pd.DataFrame([revenue_at_risk(risked, dataset, models["logistic"], cap=500)])

    # 5. figures ------------------------------------------------------
    from .report import make_figures, build_excel
    seg, ten, feat, surv_rows, surv_cox, conc = make_figures(dataset, paths,
                                                                top_n=cfg["reporting"]["top_n"],
                                                                results=results)
    excel_path = build_excel(dataset, seg, feat, ten, results, tl, impact,
                             paths, surv_rows=surv_rows, surv_cox=surv_cox)
    log.info("Excel → %s", excel_path)

    # 6. summary ------------------------------------------------------
    top_seg = seg.head(3)
    best = results.loc[results["roc_auc"].idxmax()]
    print(f"\n{'='*60}")
    print(f" CHURN ANALYSIS - {DATASETS[dataset]['label']}")
    print(f"{'='*60}")
    print(f"Churn rate: {prof['churn_rate']}%")
    print(f"Top segment driver: {top_seg.iloc[0]['group']} - "
          f"{top_seg.iloc[0]['churn_rate']*100:.1f}% churn")
    print(f"Best model: {best['model']} -> ROC-AUC {best['roc_auc']:.3f}, "
          f"PR-AUC {best['pr_auc']:.3f}")
    imp = impact.iloc[0]
    ra = imp["revenue_at_risk"]
    ra_str = f"{ra:,.0f}" if pd.notna(ra) else "n/a (no monthly_revenue)"
    print(f"Top-500 riskers: {int(imp['customers_at_risk'])} customers; "
          f"revenue at risk: {ra_str}/month")
    print(f"\nSurvival (Cox PH): concordance = {conc:.3f}")
    print(f"\nOutputs: {paths.figures} | {excel_path}")
    return {"dataset": dataset, "results": results,
            "targeting": tl, "impact": impact, "paths": paths}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Customer Churn Analysis")
    p.add_argument("--dataset", choices=sorted(DATASETS), default="telco")
    p.add_argument(
        "--stage", choices=["all", "powerbi", "preview"], default="all",
        help="all = full analysis; powerbi = data pack CSVs; preview = HTML preview",
    )
    args = p.parse_args()

    if args.stage == "powerbi":
        from .powerbi_export import run as run_powerbi
        for name, count in run_powerbi(dataset=args.dataset).items():
            print(f"{name}: {count}")
    elif args.stage == "preview":
        from .dashboard_preview import run as run_preview
        for kind, path in run_preview(dataset=args.dataset).items():
            print(f"{kind}: {path}")
    else:
        run(args.dataset)
