# 02 · Data Model

Star-schema layout for the churn data pack. Every table is written by:

```bash
python -m src.run_churn_analysis --dataset telco --stage powerbi
```

## Tables

| Table | Grain | Rows (telco) | Role |
|---|---|---|---|
| `dim_customer` | one row per customer | 7,043 | **fact** — carries the measures |
| `dim_tenure` | one row per tenure month | 72 | **dimension** (the "date table" analogue) |
| `kpi_churn_by_segment` | segment × category | 20 | small aggregate |
| `kpi_tenure_bands` | tenure band | 6 | small aggregate |
| `kpi_model_comparison` | one row per model | 2 | small aggregate |
| `kpi_survival_summary` | segment curve | 4 | small aggregate |
| `kpi_cox_hazards` | one row per covariate | 17 | small aggregate |
| `kpi_business_impact` | one row | 1 | KPI |
| `kpi_targeting_lift` | one row per list size | 6 | small aggregate |

## Relationships

```
dim_customer[tenure]  ──many:1──►  dim_tenure[tenure_month]
```

* **Cross-filter direction:** `Single` (dimension → fact). Tenure is a property
  of a customer, not a shared conformed dimension — there is one customer base.
* **Cardinality:** `Many` on `dim_customer`, `One` on `dim_tenure`.

`dim_customer[tenure_band]` is intentionally **not** related to `dim_tenure`
(second relationship on the same dimension is redundant and slows the model).
Use the `tenure` relationship and let `dim_tenure[tenure_band]` supply the label.

## Why a tenure dimension and not `dim_date`

The churn project's time axis is **tenure in months**, not calendar time — the
source data is a snapshot with no transaction dates. A calendar `dim_date` would
be empty, so the date-table requirement is satisfied by `dim_tenure`, which
provides:

* `tenure_month` — the key (0 … 72)
* `tenure_band` — human label (`0-12`, `12-24`, … `60-72`)
* `customers`, `churners`, `churn_rate_pct` — pre-aggregated for tooltips
* `survival_prob` — Kaplan-Meier estimate at that month

Time-intelligence style measures (`SAMEPERIODLASTYEAR`, …) do **not** apply.
Use the tenure key directly, or the band for grouping.

## Column notes

| Column | Type | Notes |
|---|---|---|
| `dim_customer[customer_id]` | text | natural key; **do not** summarise |
| `dim_customer[churn]` | whole number | 0/1 event indicator — use for counting |
| `dim_customer[Churn]` | text | raw `Yes`/`No` label, nicer in slicers |
| `dim_customer[churn_label]` | text | `Churned` / `Active` |
| `dim_customer[churn_risk_score]` | decimal | logistic probability, 0 … 1 |
| `dim_customer[surv_prob_at_horizon]` | decimal | Cox survival probability at 12 months |
| `dim_customer[tenure]` | whole number | **join key** to `dim_tenure` |
| `kpi_cox_hazards[p_value]` | decimal | filter `p_value < 0.05` for significant effects |

## Modelling rules

1. **Never** summarise `churn_risk_score` with `SUM` — use `AVERAGE`.
2. `churn` is a 0/1 flag; count churners with `CALCULATE(COUNTROWS(...), churn = 1)`,
   not with `SUM(churn)` inside a visual that also aggregates rows.
3. The KPI tables are **disconnected** (no relationship). Pull their values with
   `MAX(<column>)` inside a measure — see `03_dax_measures.md`.
