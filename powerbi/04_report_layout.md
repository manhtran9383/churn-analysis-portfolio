# 04 · Report Layout

Exact page-by-page spec for the Power BI report. Build in Power BI Desktop after
loading the 9 tables via `01_power_query.m` and wiring the single relationship
from `02_data_model.md`.

## Page 1 — Executive Summary

| Visual | Position | DAX / Field | Notes |
|---|---|---|---|
| **Card** (Total Customers) | Top-left | `[Total Customers]` | Format `#,##0` |
| **Card** (Churn Rate) | Top-center | `[Churn Rate]` | Format `0.0%` |
| **Card** (Avg Churn Risk) | Top-right | `[Avg Churn Risk]` | Format `0.00%` |
| **Card** (Revenue at Risk) | 2nd row left | `[Revenue at Risk]` | Currency `$#,##0` |
| **Card** (Monthly Profit Lost) | 2nd row center | `[Monthly Profit Lost]` | Currency `$#,##0` |
| **Card** (Cox Concordance) | 2nd row right | `[Cox Concordance]` | Format `0.000` |
| **Clustered bar** | Full width, 50% height | Axis: `kpi_churn_by_segment[group]`<br>Value: `kpi_churn_by_segment[churn_rate]`<br>Legend: `kpi_churn_by_segment[segment]` | Sort axis by churn rate desc; data label on |
| **Line chart** (Retention curve) | Full width, 50% height | Axis: `dim_tenure[tenure_month]`<br>Value: `dim_tenure[survival_prob]` | X-axis continuous; title "Kaplan-Meier retention curve" |

**Slicers (vertical strip, right):**
- `dim_customer[Contract]` (dropdown)
- `dim_customer[InternetService]` (dropdown)
- `dim_customer[PaymentMethod]` (dropdown)
- `dim_customer[SeniorCitizen]` (checkbox: Yes/No)

**What-if parameter:** `Selected K` (whole number, 50–1000, step 50, default 500)
→ drives Page 2.

---

## Page 2 — Targeting & Lift

| Visual | Position | DAX / Field | Notes |
|---|---|---|---|
| **Card** (Churn Rate Top K) | Top-left | `[Churn Rate Top K]` | Format `0.0%` |
| **Card** (Targeting Lift) | Top-center | `[Targeting Lift]` | Format `0.00`× |
| **Card** (Revenue at Risk Top K) | Top-right | `[Revenue at Risk Top K]` | Currency |
| **Clustered column** | Left 55% | Axis: `kpi_targeting_lift[k]`<br>Value: `kpi_targeting_lift[churn_rate_pct]`<br>Secondary axis: `kpi_targeting_lift[lift_vs_all_pp]` | Dual axis; title "Churn rate & lift by list size K" |
| **Table** (Top 500 at-risk customers) | Right 45% | `dim_customer` fields: `customerID`, `Contract`, `tenure`, `MonthlyCharges`, `churn_risk_score`, `surv_prob_at_horizon` | Filter: TOPN 500 by `churn_risk_score` DESC; conditional formatting on risk (red gradient) |
| **Slicer** | Right of column | `Selected K` what-if slider | |

---

## Page 3 — Survival Analysis

| Visual | Position | DAX / Field | Notes |
|---|---|---|---|
| **Line chart** (KM by segment) | Full width, 45% height | Axis: `dim_tenure[tenure_month]`<br>Value: `dim_tenure[survival_prob]`<br>Legend: `dim_customer[Contract]` | Use *Analyze → Forecast* off; add constant line at 0.5 |
| **Table** (Cox hazards) | Bottom-left 55% | `kpi_cox_hazards`: `covariate`, `hazard_ratio`, `ci_lower`, `ci_upper`, `p_value` | Sort by `hazard_ratio` desc; conditional format HR > 1 red, < 1 green; p-value < 0.05 bold |
| **Card** (Median Survival) | Bottom-right | `[Median Survival]` | Show "∞" if blank; format `0` months |
| **Card** (Conditional Survival) | Below median | `[Conditional Survival]` | Format `0.0%` |
| **Slicer** | Right of line chart | `dim_customer[Contract]` | Single select |

---

## Page 4 — Model & Features

| Visual | Position | DAX / Field | Notes |
|---|---|---|---|
| **Clustered bar** (Model comparison) | Top-left | Axis: `kpi_model_comparison[model]`<br>Value: `kpi_model_comparison[roc_auc]` | Add `pr_auc` as tooltip; title "ROC-AUC by model" |
| **Bar chart** (Feature importance) | Top-right | Axis: `kpi_model_comparison`… wait, features are in `kpi_cox_hazards` for survival. For classification, import `Logistic_feature_importance.png` as an image or create a separate feature table. | If no feature table: use the PNG as image visual; otherwise build a `kpi_feature_importance` table in the pack. |
| **Table** (Top features) | Bottom | Same fields as bar | |

> **Tip:** add a `kpi_feature_importance` CSV to the pack if you want
> interactive feature sorting. For now the report ships the PNG and the
> HTML preview (`dashboard_preview.html`) already shows it.

---

## Page 5 — Data Quality & Methodology (appendix)

| Visual | Position | Content |
|---|---|---|
| **Multi-row card** | Full width | `Total Customers`, `Features used`, `Train / test split`, `Random seed (42)`, `Right-censoring handling`, `Cox PH assumption check (Schoenfeld residuals)` |
| **Text box** | Below | Link to GitHub repo + notebook + Streamlit demo + methodology write-up |

---

## Theme & formatting

| Item | Value |
|---|---|
| **Colours** | Dark blue `#1F4E78` (primary), Accent blue `#2E75B6`, Churn red `#C0392B`, Survival green `#27AE60`, Neutral `#6B6B6B` |
| **Font** | Segoe UI, 11 pt base |
| **Page size** | 16:9 (1280 × 720) |
| **Background** | `#F5F6F8` |
| **Visual headers** | Off (clean look) |
| **Tooltips** | On for every visual; include all measures |
| **Drill-through** | On `dim_customer[customer_id]` → "Customer detail" page (optional) |

---

## Publishing checklist

- [ ] All 9 queries loaded, types applied, `dim_customer → dim_tenure` related
- [ ] All 18 measures created (copy from `03_dax_measures.md`)
- [ ] 5 pages built to spec above
- [ ] What-if parameter `Selected K` created and bound to slider
- [ ] Conditional formatting applied (HR, risk, p-value)
- [ ] Theme colours set; page background set
- [ ] Publish to Power BI Service → pin tiles to a dashboard
- [ ] Schedule daily refresh if source CSVs are in OneDrive / SharePoint