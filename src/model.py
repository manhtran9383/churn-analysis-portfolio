"""
Predictive churn modelling.

Two reproducible baselines on identical features with a single stratified
train/test split, so the numbers in the report are auditable:
  1. Logistic Regression (baseline, calibrated probabilities)
  2. GradientBoostingClassifier (stronger non-linear model)

Metrics: ROC-AUC, PR-AUC, precision / recall at the 0.5 threshold, and a
confusion matrix. Feature importance is extracted from the logistic model.
"""
from __future__ import annotations

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, classification_report,
                             confusion_matrix, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from .config import DATASETS
from .clean import clean


def drop_id_and_target(df: pd.DataFrame, spec: dict) -> tuple[pd.DataFrame, pd.Series]:
    y = df["churn"]
    X = df.drop(columns=[c for c in (spec["target"], "churn") if c in df.columns])
    return X, y


def _usable(X: pd.DataFrame) -> list[str]:
    return [c for c in X.columns if X[c].notna().all()]


def _infer_features(df: pd.DataFrame, dataset: str) -> dict:
    spec = DATASETS[dataset]
    df = clean(df, dataset)
    X, y = drop_id_and_target(df, spec)
    usable = _usable(X)
    cat_cols = [c for c in usable if X[c].dtype == "object"]
    num_cols = [c for c in usable if X[c].dtype != "object"]
    return {"X": X, "y": y, "usable": usable, "num_cols": num_cols,
            "cat_cols": cat_cols, "spec": spec}


def fit_models(df: pd.DataFrame, dataset: str, seed: int = 42) -> tuple[dict, pd.DataFrame]:
    from .config import load_config
    cfg = load_config()
    test_size = cfg.get("model", {}).get("test_size", 0.20)
    prepared = _infer_features(df, dataset)
    X = prepared["X"][prepared["usable"]]
    y = prepared["y"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y,
    )

    results = []

    # -- logistic regression (scaled numeric, one-hot categorical) ---------
    pre = []
    if prepared["num_cols"]:
        pre.append(("num", StandardScaler(), prepared["num_cols"]))
    if prepared["cat_cols"]:
        pre.append(("cat", "passthrough", prepared["cat_cols"]))
    model = Pipeline([
        ("prep", ColumnTransformer(pre, remainder="drop", sparse_threshold=0)),
        ("clf", LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced")),
    ])
    model.fit(X_train, y_train)
    results.append(_metrics(model, X_test, y_test, "logistic"))

    # -- gradient boosting (raw DataFrame, handles one-hot internally) ------
    gb = GradientBoostingClassifier(n_estimators=300, max_depth=3,
                                     learning_rate=0.05, subsample=0.8,
                                     random_state=seed, n_iter_no_change=10)
    gb.fit(X_train, y_train)
    results.append(_metrics(gb, X_test, y_test, "gradient_boosting"))

    return {"logistic": model, "gradient_boosting": gb}, pd.DataFrame(results)


def _metrics(model, X_test: pd.DataFrame, y_test: pd.Series, label: str) -> dict:
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    rep = classification_report(y_test, pred, output_dict=True)
    cm = confusion_matrix(y_test, pred)

    fi = None
    fi_list = None
    if label == "logistic":
        coef = model.named_steps["clf"].coef_[0]
        fi = (pd.DataFrame({"feature": list(X_test.columns), "importance": coef})
                .assign(importance=lambda d: d["importance"].abs())
                .sort_values("importance", ascending=False)
                .head(10))
        # flat string list so Excel writes cleanly (no nested table)
        fi_list = [f"{r.feature}: {r.importance:.4f}" for r in fi.itertuples()]

    return {
        "model": label,
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
        "pr_auc": round(float(average_precision_score(y_test, proba)), 4),
        "recall_1": round(float(rep["1"]["recall"]), 4),
        "precision_1": round(float(rep["1"]["precision"]), 4),
        "f1_1": round(float(rep["1"]["f1-score"]), 4),
        "support_1": int(y_test.sum()),
        "tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
        # flat string list -> Excel writes cleanly
        "feature_importance": fi_list,
        # DataFrame -> charts
        "feature_importance_df": fi,
    }
