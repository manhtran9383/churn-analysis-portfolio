"""
Data loading and profiling for the churn project.

Turns each raw CSV into a tidy DataFrame with a binary `churn` target, then
profiles it so the numbers quoted in the report are reproducible.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .config import DATASETS, Paths


def load_raw(paths: Paths, dataset: str) -> pd.DataFrame:
    """Read a raw dataset and encode the target as 0/1."""
    spec = DATASETS[dataset]
    df = pd.read_csv(paths.raw_file(spec["file"]), dtype=str, keep_default_na=False)
    # strip whitespace AND a UTF-8 BOM from header cells -- several public CSVs
    # (e.g. the IBM HR export) start with "\ufeffAge", which silently corrupts the
    # first column name and breaks any later column lookup.
    df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]

    df["churn"] = (df[spec["target"]].str.strip() == spec["positive"]).astype(int)
    return df


def profile(df: pd.DataFrame, spec: dict) -> dict:
    """Return a data-quality profile used by the report."""
    target = spec["target"]
    churn_rate = float(df["churn"].mean())

    blanks = {}
    for col in df.columns:
        if col in ("churn", target):
            continue
        n_blank = int((df[col].astype(str).str.strip() == "").sum())
        if n_blank:
            blanks[col] = n_blank

    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "churners": int(df["churn"].sum()),
        "retained": int((df["churn"] == 0).sum()),
        "churn_rate": round(churn_rate * 100, 2),
        "blank_cells": blanks,
        "duplicate_ids": int(df[spec["id"]].duplicated().sum()),
        "numeric_columns": [c for c in df.columns
                            if pd.to_numeric(df[c], errors="coerce").notna().sum() > len(df) * 0.9],
    }


def save_profile(paths: Paths, dataset: str, profile: dict) -> str:
    path = paths.processed_file(f"{dataset}_profile.json")
    path.write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(path)


if __name__ == "__main__":
    import sys

    paths = Paths()
    name = sys.argv[1] if len(sys.argv) > 1 else "telco"
    df = load_raw(paths, name)
    p = profile(df, DATASETS[name])
    print(json.dumps(p, indent=2, ensure_ascii=False))
