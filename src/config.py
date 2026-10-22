"""Configuration and dataset registry for the churn analysis project."""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_config() -> dict:
    with open(ROOT / "config.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


class Paths:
    """Resolved locations for the churn project."""

    def __init__(self, cfg: dict | None = None):
        cfg = cfg or load_config()
        self.root = ROOT
        self.raw = ROOT / "data" / "raw"
        self.processed = ROOT / "data" / "processed"
        self.reports = ROOT / "reports"
        self.figures = self.reports / "figures"
        self.sql = ROOT / "sql"
        for path in (self.processed, self.reports, self.figures):
            path.mkdir(parents=True, exist_ok=True)

    def raw_file(self, name: str) -> Path:
        return self.raw / name

    def processed_file(self, name: str) -> Path:
        return self.processed / name

    def report_file(self, name: str) -> Path:
        return self.reports / name


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------
# Each entry describes how to read one churn dataset: which column is the
# target, what value means "churned", and the business columns that matter.
DATASETS = {
    "telco": {
        "file": "Telco-Customer-Churn.csv",
        "label": "IBM Telco Customer Churn",
        "target": "Churn",
        "positive": "Yes",
        "id": "customerID",
        "tenure": "tenure",
        "monthly_revenue": "MonthlyCharges",
        "total_revenue": "TotalCharges",
        "currency": "USD",
        "segments": ["Contract", "InternetService", "PaymentMethod", "gender",
                     "SeniorCitizen", "Partner", "Dependents", "PaperlessBilling"],
        "services": ["PhoneService", "MultipleLines", "OnlineSecurity",
                     "OnlineBackup", "DeviceProtection", "TechSupport",
                     "StreamingTV", "StreamingMovies"],
        "fig_dir": Path("reports/"),
        "survival_segment": "Contract",
        "time_unit": "months",
    },
    "bank": {
        "file": "Churn_Modelling.csv",
        "label": "Bank Customer Churn (Churn Modelling)",
        "target": "Exited",
        "positive": "1",
        "id": "CustomerId",
        "tenure": "Tenure",
        "monthly_revenue": None,
        "total_revenue": None,
        "currency": "EUR",
        "segments": ["Geography", "Gender", "IsActiveMember", "HasCrCard"],
        "services": [],
        "fig_dir": Path("reports/"),
        "survival_segment": "Geography",
        "time_unit": "years",
    },
    "hr": {
        "file": "HR_Employee_Attrition.csv",
        "label": "IBM HR Analytics Employee Attrition & Performance",
        "target": "Attrition",
        "positive": "Yes",
        "id": "EmployeeNumber",
        "tenure": "YearsAtCompany",
        "monthly_revenue": "MonthlyIncome",
        "total_revenue": None,
        "currency": "USD",
        "segments": ["Department", "JobRole", "BusinessTravel", "OverTime",
                     "MaritalStatus", "Gender", "EducationField"],
        "services": [],
        "fig_dir": Path("reports/"),
        "survival_segment": "Department",
        "time_unit": "years",
    },
}

# Tenure bins for retention curves (dataset-aware)
# Telco tenure is in months (0-72), Bank/HR tenure is in years (0-10 / 0-40).
TENURE_BINS = {
    "telco": [0, 12, 24, 36, 48, 60, 72, 84],
    "bank": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    "hr": [0, 5, 10, 15, 20, 25, 30, 35, 40, 45],
}
TENURE_LABELS = {
    "telco": ["0-12", "12-24", "24-36", "36-48", "48-60", "60-72", "72-84"],
    "bank": ["0-1", "1-2", "2-3", "3-4", "4-5", "5-6", "6-7", "7-8", "8-9", "9-10", "10-11"],
    "hr": ["0-5", "5-10", "10-15", "15-20", "20-25", "25-30", "30-35", "35-40", "40-45"],
}
