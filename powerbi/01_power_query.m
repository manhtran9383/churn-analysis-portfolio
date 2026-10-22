// =====================================================================
// 01_power_query.m - paste-ready Power Query for the churn data pack
// =====================================================================
//
// HOW TO USE
//   1. Copy the whole block for a table into Power BI Desktop
//      (Transform data -> Transform -> Advanced Editor).
//   2. Edit the file path in Csv.Document(...) to your machine.
//   3. Repeat for each table, then apply 02_data_model.md.
//
// Column types below were generated from the actual CSV headers of the
// TELCO data pack. The M code is a template: the table names, the M shape
// and every join/filter are dataset-agnostic, but the column TYPES in each
// TransformColumnTypes(...) block must be edited to the real CSV headers of
// the target dataset, which you can read from
//   powerbi/data/<dataset>/dim_customer.csv
// (telco = customerID / Contract / InternetService / PaymentMethod /
//  MonthlyCharges / TotalCharges ;  bank = CustomerId / Geography /
//  Tenure / EstimatedSalary ;  hr  = EmployeeNumber / Department /
//  OverTime / MonthlyIncome).
//
// Build for any dataset with:
//   python -m src.run_churn_analysis --dataset <dataset> --stage powerbi
// =====================================================================

// ---- dim_customer --------------------------------------------------------
let
    Source = Csv.Document(File.Contents("C:\\INSERT\\PATH\\dim_customer.csv"),
        [Delimiter=",", Encoding=65001, QuoteStyle=QuoteStyle.Csv]),
    Promoted = Table.PromoteHeaders(Source, [PromoteAllScalars=true]),
    Typed = Table.TransformColumnTypes(Promoted, {{"customerID", type text}, {"gender", type text}, {"SeniorCitizen", Int64.Type}, {"Partner", type text}, {"Dependents", type text}, {"tenure", Int64.Type}, {"PhoneService", type text}, {"MultipleLines", type text}, {"InternetService", type text}, {"OnlineSecurity", type text}, {"OnlineBackup", type text}, {"DeviceProtection", type text}, {"TechSupport", type text}, {"StreamingTV", type text}, {"StreamingMovies", type text}, {"Contract", type text}, {"PaperlessBilling", type text}, {"PaymentMethod", type text}, {"MonthlyCharges", type number}, {"TotalCharges", type number}, {"Churn", type text}, {"churn", Int64.Type}, {"churn_label", type text}, {"churn_risk_score", type number}, {"surv_prob_at_horizon", type number}, {"tenure_band", type text}})
in
    Typed

// ---- dim_tenure --------------------------------------------------------
let
    Source = Csv.Document(File.Contents("C:\\INSERT\\PATH\\dim_tenure.csv"),
        [Delimiter=",", Encoding=65001, QuoteStyle=QuoteStyle.Csv]),
    Promoted = Table.PromoteHeaders(Source, [PromoteAllScalars=true]),
    Typed = Table.TransformColumnTypes(Promoted, {{"tenure_month", Int64.Type}, {"tenure_band", type text}, {"customers", Int64.Type}, {"churners", Int64.Type}, {"churn_rate_pct", type number}, {"survival_prob", type number}})
in
    Typed

// ---- kpi_churn_by_segment --------------------------------------------------------
let
    Source = Csv.Document(File.Contents("C:\\INSERT\\PATH\\kpi_churn_by_segment.csv"),
        [Delimiter=",", Encoding=65001, QuoteStyle=QuoteStyle.Csv]),
    Promoted = Table.PromoteHeaders(Source, [PromoteAllScalars=true]),
    Typed = Table.TransformColumnTypes(Promoted, {{"segment", type text}, {"group", type text}, {"churn_rate", type number}, {"count", Int64.Type}, {"churners", Int64.Type}})
in
    Typed

// ---- kpi_tenure_bands --------------------------------------------------------
let
    Source = Csv.Document(File.Contents("C:\\INSERT\\PATH\\kpi_tenure_bands.csv"),
        [Delimiter=",", Encoding=65001, QuoteStyle=QuoteStyle.Csv]),
    Promoted = Table.PromoteHeaders(Source, [PromoteAllScalars=true]),
    Typed = Table.TransformColumnTypes(Promoted, {{"tenure_band", type text}, {"churn_rate", type number}, {"count", Int64.Type}, {"churners", Int64.Type}})
in
    Typed

// ---- kpi_model_comparison --------------------------------------------------------
let
    Source = Csv.Document(File.Contents("C:\\INSERT\\PATH\\kpi_model_comparison.csv"),
        [Delimiter=",", Encoding=65001, QuoteStyle=QuoteStyle.Csv]),
    Promoted = Table.PromoteHeaders(Source, [PromoteAllScalars=true]),
    Typed = Table.TransformColumnTypes(Promoted, {{"model", type text}, {"roc_auc", type number}, {"pr_auc", type number}, {"recall_1", type number}, {"precision_1", type number}, {"f1_1", type number}, {"support_1", Int64.Type}, {"tn", Int64.Type}, {"fp", Int64.Type}, {"fn", Int64.Type}, {"tp", Int64.Type}, {"feature_importance", type text}})
in
    Typed

// ---- kpi_survival_summary --------------------------------------------------------
let
    Source = Csv.Document(File.Contents("C:\\INSERT\\PATH\\kpi_survival_summary.csv"),
        [Delimiter=",", Encoding=65001, QuoteStyle=QuoteStyle.Csv]),
    Promoted = Table.PromoteHeaders(Source, [PromoteAllScalars=true]),
    Typed = Table.TransformColumnTypes(Promoted, {{"group", type text}, {"median_survival_time", type number}, {"survival_at_6", type number}, {"survival_at_12", type number}, {"survival_at_24", type number}, {"survival_at_36", type number}, {"survival_at_48", type number}, {"survival_at_60", type number}, {"survival_at_72", type number}})
in
    Typed

// ---- kpi_cox_hazards --------------------------------------------------------
let
    Source = Csv.Document(File.Contents("C:\\INSERT\\PATH\\kpi_cox_hazards.csv"),
        [Delimiter=",", Encoding=65001, QuoteStyle=QuoteStyle.Csv]),
    Promoted = Table.PromoteHeaders(Source, [PromoteAllScalars=true]),
    Typed = Table.TransformColumnTypes(Promoted, {{"concordance", type number}, {"covariate", type text}, {"coef", type number}, {"hazard_ratio", type number}, {"hr_lower_95", type number}, {"hr_upper_95", type number}, {"p_value", type number}})
in
    Typed

// ---- kpi_business_impact --------------------------------------------------------
let
    Source = Csv.Document(File.Contents("C:\\INSERT\\PATH\\kpi_business_impact.csv"),
        [Delimiter=",", Encoding=65001, QuoteStyle=QuoteStyle.Csv]),
    Promoted = Table.PromoteHeaders(Source, [PromoteAllScalars=true]),
    Typed = Table.TransformColumnTypes(Promoted, {{"cap", Int64.Type}, {"customers_at_risk", Int64.Type}, {"revenue_at_risk", type number}, {"estimated_monthly_profit_lost", type number}, {"avg_risk_of_top_cap", type number}})
in
    Typed

// ---- kpi_targeting_lift --------------------------------------------------------
let
    Source = Csv.Document(File.Contents("C:\\INSERT\\PATH\\kpi_targeting_lift.csv"),
        [Delimiter=",", Encoding=65001, QuoteStyle=QuoteStyle.Csv]),
    Promoted = Table.PromoteHeaders(Source, [PromoteAllScalars=true]),
    Typed = Table.TransformColumnTypes(Promoted, {{"list_size", Int64.Type}, {"churners", Int64.Type}, {"customers", Int64.Type}, {"churn_rate_pct", type number}, {"avg_risk", type number}, {"lift_vs_all_pp", type number}})
in
    Typed
