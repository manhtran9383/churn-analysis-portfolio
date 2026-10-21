# Customer Churn Analysis — Portfolio Project

End-to-end, fully reproducible customer churn analysis on **real** public datasets.
Two datasets side-by-side: the telecom `Telco-Customer-Churn` (revenue attached)
and the bank `Churn_Modelling` dataset. Built so a candidate can *show* a hiring
manager an auditable, end-to-end analysis — raw data → insight → model → dollars.

## The story it tells

* **Data quality first.** The telecom `TotalCharges` is a *text* column with 11
  blanks (customers with `tenure = 0`). Those blanks are filled with the first
  month's charge, so `TotalCharges ≈ MonthlyCharges × tenure` holds. This is a
  textbook "trap" and it is handled and documented rather than ignored.
* **Segment → model → money.** Churn by segment, a retention curve by tenure,
  logistic + gradient-boosting baselines, calibrated probabilities used to rank
  the top-500 at-risk customers, and the revenue `at risk`.
* **Survival analysis — when, not just if.** Right-censored Kaplan-Meier curves
  by segment + Cox proportional-hazards model with concordance 0.84 (telco) /
  0.71 (bank). Log-rank tests confirm Contract / Geography drive *when* churn
  happens. Median survival, survival-at-horizon (6/12/24/36/48/60/72 months),
  and per-customer survival probabilities at 12 months for dynamic ranking.
* **Reproducible.** One command, a fixed `random_state`, and a `data/raw/`
  registry. Nothing in `data/processed/` is hand-edited — it is rebuilt every run.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m src.run_churn_analysis --dataset telco    # telecom deep-dive
python -m src.run_churn_analysis --dataset bank     # bank comparison
```

Both run in ~30 s on a laptop.

## Project layout

```
churn-project/
├── config.yaml                 # dataset name, modelling & business thresholds
├── requirements.txt
├── src/
│   ├── config.py               # dataset registry + paths
│   ├── load_data.py            # read raw CSV → binary churn target + profile
│   ├── clean.py                # TotalCharges trap, label encoding, drop IDs
│   ├── analysis.py             # segment churn, top features, tenure bands
│   ├── model.py                # logistic + gradient boosting, metrics, importance
│   ├── impact.py               # risk score → revenue-at-risk & targeting lift
│   ├── survival.py             # Kaplan-Meier + Cox PH (right-censored)
│   ├── report.py               # figures + multi-sheet Excel export
│   └── run_churn_analysis.py   # entry point: load → clean → model → survival → report
├── data/
│   ├── raw/                    # the two real CSVs (do not edit)
│   └── processed/              # rebuilt every run (profiles, cleaned frames)
├── reports/
│   ├── Churn_Analysis.xlsx     # 8 sheets: segments, features, tenure, models,
│   │                          #    targeting, business impact, survival_summary, surv_cox
│   └── figures/                # 6 PNGs (segment bar, retention curve, top
│                                #    features, logistic importance, model cmp, survival curve)
└── tests/                      # optional: cover the full pipeline in an isolated
                                 #    copy of the project
```

## Outputs at a glance

| Dataset        | Churn rate | Best model (ROC / PR) | Top segment (driver)        | 500 riskers | Survival concordance |
|----------------|-----------|------------------------|-----------------------------|-------------|----------------------|
| Telecom Telco  | 26.54 %   | GradientBoosting 0.844 / 0.663 | Electronic check (45.3%) | ≈ 43 246 $/mo | 0.84 |
| Bank           | 20.37 %   | GradientBoosting 0.867 / 0.714 | Germany (32.4%) | n/a (no revenue col) | 0.71 |

## Portfolio notes for the conversation with a recruiter

1. **The TotalCharges trap is the hook.** "I found a data-quality bug that 3
   jobs overlook, fixed it with a documented rule, and it changed the revenue
   numbers by X."
2. **The model is honest.** Logistic baseline + a stronger model, same split,
   PR-AUC reported alongside ROC-AUC (imbalanced target). Explaining *why* a
   customer leaves is more impressive than just predicting it — hence the
   `feature_importance` and `risk_score` outputs.
3. **Business impact, not just metrics.** `targeting_lift` shows that serving
   the top-500 riskers concentrates ~3× the churn rate of the base; the
   `revenue_at_risk` quantifies what retention work is actually worth.
4. **Survival analysis answers "when", not just "if".** Classification ignores
   censored customers (still active at snapshot). Kaplan-Meier + Cox PH model
   right-censoring correctly: `churn=1` = event observed, `churn=0` = censored.
   Concordance 0.84/0.71 proves the model orders customers by time-to-churn.
   Per-customer survival probability at 12 months enables a *dynamic* retention
   ranking — who will leave *soonest*.
5. **Every number is re-runnable.** `data/raw/` and `config.yaml` are the only
   things you touch; `data/processed/` is a pure by-product.

## Reproducibility

* `config.yaml` → `dataset.name` switches both datasets.
* `src/config.py` `DATASETS` is the single registry: file, target column, positive
  label, numeric/segment/service columns per dataset.
* `tests/` (optional) runs the whole pipeline in an isolated copy of the project.
