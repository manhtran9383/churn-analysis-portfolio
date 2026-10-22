# 03 · DAX Measures

Paste these into a **new measure** in Power BI Desktop (`Table tools → New measure`).
They are written against the tables in `02_data_model.md` and verified against
the pipeline's own numbers.

## Core KPIs

```dax
// Total customers in the analysed base
Total Customers =
COUNTROWS ( dim_customer )

// Churn rate — the honest 0/1 count, not SUM(churn) over an aggregated visual
Churn Rate =
VAR churners =
    CALCULATE ( COUNTROWS ( dim_customer ), dim_customer[churn] = 1 )
VAR total = [Total Customers]
RETURN
    DIVIDE ( churners, total )

// Average model risk across the base
Avg Churn Risk =
AVERAGE ( dim_customer[churn_risk_score] )

// Revenue at risk per month (telco only; blank for bank — no revenue column)
Revenue at Risk =
MAX ( kpi_business_impact[revenue_at_risk] )

// Estimated monthly profit lost (35% margin, from the pipeline)
Monthly Profit Lost =
MAX ( kpi_business_impact[estimated_monthly_profit_lost] )

// Cox model concordance — how well the model orders time-to-churn
Cox Concordance =
MAX ( kpi_cox_hazards[concordance] )

// Kaplan-Meier survival probability at 12 months, all customers
Survival @ 12 Months =
CALCULATE (
    MAX ( kpi_survival_summary[survival_at_12] ),
    kpi_survival_summary[group] = "All customers"
)
```

## Ranking / targeting

```dax
// Churn rate of the top-K customers, ranked by risk.
// K is a what-if parameter (see 04_report_layout.md).
Churn Rate Top K =
VAR K = [Selected K]
VAR ranked =
    TOPN ( K, ALLSELECTED ( dim_customer ), dim_customer[churn_risk_score], DESC )
RETURN
    DIVIDE (
        CALCULATE ( COUNTROWS ( dim_customer ), dim_customer[churn] = 1, ranked ),
        COUNTROWS ( ranked )
    )

// Lift = top-K churn rate / base churn rate
Targeting Lift =
DIVIDE ( [Churn Rate Top K], [Churn Rate] )

// Revenue at risk for the top-K list (telco)
Revenue at Risk Top K =
VAR K = [Selected K]
VAR ranked =
    TOPN ( K, ALLSELECTED ( dim_customer ), dim_customer[churn_risk_score], DESC )
RETURN
    SUMX ( ranked, dim_customer[MonthlyCharges] )
```

> **Note on `Churn Rate Top K`:** the `ranked` variable must be a table
> expression. In some Power BI versions the inner `CALCULATE` needs the filter
> applied as a table argument; if it errors, rewrite as
> `CALCULATE ( [Churn Rate], ranked )`.

## Segment analysis

```dax
// Churn rate for the currently selected segment category
Churn Rate by Segment =
VAR seg = SELECTEDVALUE ( kpi_churn_by_segment[segment] )
VAR grp = SELECTEDVALUE ( kpi_churn_by_segment[group] )
RETURN
    CALCULATE (
        MAX ( kpi_churn_by_segment[churn_rate] ),
        kpi_churn_by_segment[segment] = seg,
        kpi_churn_by_segment[group] = grp
    )

// Number of churners in the selected segment
Churners in Segment =
VAR seg = SELECTEDVALUE ( kpi_churn_by_segment[segment] )
VAR grp = SELECTEDVALUE ( kpi_churn_by_segment[group] )
RETURN
    CALCULATE (
        MAX ( kpi_churn_by_segment[churners] ),
        kpi_churn_by_segment[segment] = seg,
        kpi_churn_by_segment[group] = grp
    )
```

## Survival

```dax
// Median survival for the selected segment (blank = curve never reaches 50%)
Median Survival =
CALCULATE (
    MAX ( kpi_survival_summary[median_survival_time] ),
    kpi_survival_summary[group] = SELECTEDVALUE ( kpi_survival_summary[group] )
)

// Conditional survival: P(stay to 72 | already stayed to 12)
// = survival_at_72 / survival_at_12, computed over the selected segment
Conditional Survival =
DIVIDE (
    CALCULATE ( MAX ( kpi_survival_summary[survival_at_72] ) ),
    CALCULATE ( MAX ( kpi_survival_summary[survival_at_12] ) )
)
```

## Hazard ratios (Cox)

```dax
// Hazard ratio for the selected covariate
Hazard Ratio =
CALCULATE (
    MAX ( kpi_cox_hazards[hazard_ratio] ),
    kpi_cox_hazards[covariate] = SELECTEDVALUE ( kpi_cox_hazards[covariate] )
)

// Is the effect statistically significant?
Is Significant =
IF ( CALCULATE ( MAX ( kpi_cox_hazards[p_value] ) ) < 0.05, "Yes", "No" )
```

## Formatting

| Measure | Format |
|---|---|
| `Churn Rate`, `Churn Rate Top K`, `Churn Rate by Segment` | Percentage, 1 decimal |
| `Targeting Lift` | Decimal number, 2 decimals, suffix `×` |
| `Revenue at Risk`, `Revenue at Risk Top K` | Currency `$#,##0` |
| `Survival @ 12 Months`, `Conditional Survival` | Percentage, 1 decimal |
| `Cox Concordance` | Decimal number, 3 decimals |
| `Hazard Ratio` | Decimal number, 2 decimals |

## What-if parameter

Create **Selected K** as a What-if parameter on `dim_customer[churn_risk_score]`
domain: *Whole number*, min `50`, max `1000`, increment `50`, default `500`.
This drives the `TOPN` in the targeting measures and the K slider on the
report page.
