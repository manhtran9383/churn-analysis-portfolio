# Customer Churn Analysis — Portfolio Project

End-to-end, fully reproducible customer churn analysis on **real** public datasets.
Two datasets side-by-side: the telecom `Telco-Customer-Churn` (revenue attached)
and the bank `Churn_Modelling` dataset. Built so a candidate can *show* a hiring
manager an auditable, end-to-end analysis — raw data → insight → model → dollars.

## The story it tells

* **Data quality first.** The telecom `TotalCharges` is a *text* column with 11
  blanks (customers with `tenure = 0`). Those blanks are filled with the first
  month's charge, so `TotalCharges ≈ MonthlyCharges × tenure` holds for tenure > 0
  — the only case a retention analyst actually uses. The zero-tenure blanks are a
  data-quality trap: filled with the first month's charge they make `TotalCharges`
  equal to `MonthlyCharges` (i.e. tenure 1), a hidden assumption this project
  documents rather than ignores.
* **Segment → model → money.** Churn by segment, a retention curve by tenure,
  logistic + gradient-boosting baselines, class-weighted probability scores used
  to rank the top-500 at-risk customers, and the revenue `at risk`.
* **Survival analysis — when, not just if.** Right-censored Kaplan-Meier curves
  by segment + Cox proportional-hazards model with concordance 0.844 (telco) /
  0.720 (bank). Log-rank tests confirm Contract / Geography drive *when* churn
  happens. Median survival, survival-at-horizon (6/12/24/36/48/60/72 months),
  and per-customer survival probabilities at 12 months for dynamic ranking.
* **Reproducible.** One command, a fixed `random_state`, and a `data/raw/` registry.
  Nothing in `data/processed/` is hand-edited — it is rebuilt every run.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m src.run_churn_analysis --dataset telco    # telecom deep-dive
python -m src.run_churn_analysis --dataset bank     # bank comparison
python -m src.run_churn_analysis --dataset hr       # human-resources attrition (1,470 employees)
```

Both run in ~30 s on a laptop.

### Human resources (HR) — streamlined 1-pager

> Model-source: HR impact figures below are **logistic on the hold-out test set**;
> the app's "Lift" uses **gradient-boosting** on the test set. Don't mix the two
> when quoting. (telco/bank: logistic 2.02×/2.19×; GB 2.07×/2.84×.)



The HR dataset (`HR_Employee_Attrition.csv`, 1,470 employees, 16.1% attrition) is a
single-command portfolio slice aimed at a **people analytics / HR** audience:

```bash
# run the full HR analysis (figures + Excel + model + survival + targeting)
python -m src.run_churn_analysis --dataset hr
```

What the HR run surfaces — no extra modelling code needed beyond the shared pipeline:

- **Churn mix** (`churn_by_segment` tab): top driver is *Sales Representative* at 39.8%
  churn vs a 16.1% base rate; `OverTime`, `MonthlyIncome` and *JobRole* also rank high.
- **Retention by tenure** (`tenure_bands` tab): churn concentrates in the first 5 years
  (`YearsAtCompany` 0–5), then plateaus — an early-retention intervention window.
- **Risk ranking** (`targeting_lift` tab): top-500 predicted leavers vs the rest of the
  base; `Lift` column shows how much churn rate concentrates at the top of the list.
- **Survival/KM** (`surv_cox` + `Survival_curve.png`): right-censored KM by
  `Department` + Cox PH with tenure-aware horizons (1, 2, 3, 5, 7, 10, 15, 20 years).
- **Business impact** (`business_impact` tab): `revenue_at_risk` is `n/a` (no per-employee
  revenue column) — HR's "cost" is attrition itself, so the sheet surfaces the
  *churn-rate concentration* and *targeting lift* instead.
- **Catalog**: `EmployeeNumber` (customer ID) is auto-dropped; `EmployeeCount`,
  `Over18`, `StandardHours` (zero-variance columns) are auto-dropped; `JobRole`,
  `Department`, `BusinessTravel`, `OverTime`, `MaritalStatus`, `Gender`,
  `EducationField` are one-hot encoded for the models.
- **Honor roll**: Cox concordance 0.860 (HR, baseline-only) vs 0.844 (telco) vs 0.720 (bank). The earlier 0.937 included time-cumulative covariates (look-ahead) — see the HR Cox-note above.

This differs from the two revenue datasets in one deliberate way: the "cost" of attrition is
not a monthly revenue number, so the impact sheet reports churn-concentration metrics
rather than `revenue_at_risk`.

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
├── powerbi/                    # Power BI build pack: 10 CSVs per dataset + M code + DAX + layout
└── notebooks/                  # executed Jupyter notebooks (0 error cells)
```

## Outputs at a glance

| Dataset        | Churn rate | Best model (ROC / PR) | Top segment (driver)        | 500 riskers | Survival concordance |
|----------------|-----------|------------------------|-----------------------------|-------------|----------------------|
| Telecom Telco  | 26.54 %   | GradientBoosting 0.844 / 0.663 | Electronic check (45.3%) | ≈ 43 246 $/mo | 0.844 |
| Bank           | 20.37 %   | GradientBoosting 0.867 / 0.714 | Germany (32.4%) | n/a (no revenue col) | 0.720 |
| HR (IBM Attrition) | 16.12 % | Logistic 0.818 / 0.622 | Sales Representative (39.8%) | n/a (no revenue col) | 0.860 |

> **HR Cox covariates** — time-cumulative quantities (TotalWorkingYears,
> YearsInCurrentRole, YearsSinceLastPromotion, YearsWithCurrManager,
> NumCompaniesWorked) are excluded from the HR Cox model: they are measured at
> exit and bounded by the duration, so they leak look-ahead information. With
> them, concordance was 0.937; baseline-only is **0.860**.

## Portfolio notes for the conversation with a recruiter

1. **The TotalCharges trap is the hook.** "I found a data-quality bug that 3
   jobs overlook, fixed it with a documented rule, and it changed the revenue
   numbers by X."
2. **The model is honest.** Logistic baseline + a stronger model, same split,
   PR-AUC reported alongside ROC-AUC (imbalanced target). Explaining *why* a
   customer leaves is more impressive than just predicting it — hence the
   `feature_importance` and `risk_score` outputs.
3. **Business impact, not just metrics.** `targeting_lift` shows that serving
   the top-500 riskers concentrates ~2× the churn rate of the base (measured on
   the hold-out test set); the `revenue_at_risk` quantifies what retention work
   is actually worth.
4. **Survival analysis answers "when", not just "if".** Classification ignores
   censored customers (still active at snapshot). Kaplan-Meier + Cox PH model
   right-censoring correctly: `churn=1` = event observed, `churn=0` = censored.
   Concordance 0.844/0.720 proves the model orders customers by time-to-churn.
   Per-customer survival probability at 12 months enables a *dynamic* retention
   ranking — who will leave *soonest*.
5. **Every number is re-runnable.** `data/raw/` and `config.yaml` are the only
   things you touch; `data/processed/` is a pure by-product.

## Reproducibility

* `config.yaml` → `dataset.name` switches both datasets.
* `src/config.py` `DATASETS` is the single registry: file, target column, positive
  label, numeric/segment/service columns per dataset.
* `tests/` runs 5 headless Streamlit smoke tests (`streamlit.testing.v1.AppTest`)
  that assert the app renders, the tabs exist, and the ranking/survival metrics
  match the pipeline output.
