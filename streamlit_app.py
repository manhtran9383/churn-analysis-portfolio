"""
Interactive demo for `notebooks/churn_ai_llm_reframing.ipynb`.

Run:
    streamlit run streamlit_app.py

Four tabs mirror the notebook:
  1. Ranking      - Recall@K / Precision@K (learn-to-rank framing)
  2. Survival     - Kaplan-Meier curves + Cox PH hazard ratios
  3. AI experiments - Naive Bayes on the service bundle, TF-IDF on support notes
  4. LLM design   - prompt-as-ranker sketch with guardrails (no API key, no network)

Everything is cached: the dataset is loaded and the models fitted once, then the
UI only re-reads the cached results.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import DATASETS, Paths
from src.clean import clean
from src.load_data import load_raw
from src.model import fit_models
from src.survival import (
    HORIZONS,
    cox_model,
    cox_table,
    km_by_group,
    km_overall,
    logrank,
    risk_score_at,
    survival_summary,
)

st.set_page_config(
    page_title="Churn as a Ranking System",
    page_icon="📊",
    layout="wide",
)

# ---------------------------------------------------------------------------
# cached data + models
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading data…")
def load(dataset: str):
    paths = Paths()
    raw = load_raw(paths, dataset)
    return raw, clean(raw, dataset)


@st.cache_data(show_spinner="Fitting churn models…")
def fit_models_cached(dataset: str, seed: int = 42):
    _, df = load(dataset)
    return fit_models(df, dataset, seed=seed)


@st.cache_data(show_spinner="Fitting survival model…")
def survival_cached(dataset: str):
    """Cache only picklable results (DataFrames / floats), not lifelines objects."""
    raw, _ = load(dataset)
    cph = cox_model(raw, dataset)
    kmf_all = km_overall(raw, dataset)
    seg_col = spec.get("survival_segment", "Department")
    curves = km_by_group(raw, dataset, seg_col)
    horizon = HORIZONS[dataset][1]
    return {
        "concordance": float(cph.concordance_index_),
        "cox_table": cox_table(cph),
        "logrank": logrank(raw, dataset, seg_col),
        "surv_prob": risk_score_at(cph, raw, dataset, horizon),
        "horizon": horizon,
        "all_summary": survival_summary(kmf_all, HORIZONS[dataset]),
        "summary": {lab: survival_summary(k, HORIZONS[dataset])
                    for lab, k in curves.items()},
        "km_all": kmf_all.survival_function_,
        "km_curves": {lab: k.survival_function_ for lab, k in curves.items()},
        "seg_col": seg_col,
    }


# ---------------------------------------------------------------------------
# sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("📊 Churn as a Ranking System")
    st.caption("Interactive demo of `notebooks/churn_ai_llm_reframing.ipynb`")
    dataset = st.selectbox(
        "Dataset", ["telco", "bank", "hr"],
        format_func=lambda d: DATASETS[d]["label"],
    )
    st.divider()
    _, df_side = load(dataset)
    st.metric("Customers", f"{len(df_side):,}")
    st.metric("Churn rate", f"{df_side['churn'].mean():.2%}")
    st.divider()
    st.markdown("**Stack** · pandas · scikit-learn · lifelines · plotly")
    st.markdown("No API keys, no network calls.")

spec = DATASETS[dataset]

# ---------------------------------------------------------------------------
# tab 1 - ranking
# ---------------------------------------------------------------------------
tab_rank, tab_surv, tab_ai, tab_llm = st.tabs(
    ["🎯 Ranking (Recall@K)", "⏳ Survival", "🧪 AI experiments", "🤖 LLM design sketch"]
)

with tab_rank:
    st.subheader("Ranking quality — who do we call first?")
    st.markdown(
        "Classification asks *\"will this customer churn?\"*. The business asks "
        "*\"give me the 500 customers most worth calling\"*. That is a ranking "
        "problem, scored with **Recall@K** and **Precision@K**."
    )

    models, results = fit_models_cached(dataset)
    gb = models["gradient_boosting"]
    _, df = load(dataset)

    X = df.drop(columns=[spec["target"], "churn"])
    y = df["churn"]
    _, X_te, _, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    proba = gb.predict_proba(X_te)[:, 1]
    y_true = y_te.values
    base_rate = y_true.mean()

    n_test = int(len(y_true))
    k_default = min(500, max(50, n_test // 2))
    k = min(int(st.slider("K — customers called first", 50, 1000, k_default, 50)),
            n_test)
    k_used = k
    idx = np.argsort(-proba)[:k]
    recall = y_true[idx].sum() / max(y_true.sum(), 1)
    precision = y_true[idx].mean()
    lift = precision / base_rate

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Recall@K", f"{recall:.1%}", f"{int(y_true[idx].sum())} churners captured")
    c2.metric("Precision@K", f"{precision:.1%}")
    c3.metric("Base rate", f"{base_rate:.1%}")
    c4.metric("Lift", f"{lift:.2f}×")
    if k < 500:
        st.caption(f"K auto-clamped to {k_used} (test set has {n_test} rows)")

    ks = [k for k in (50, 100, 200, 500, 1000) if k <= n_test]
    curve = pd.DataFrame([
        {
            "K": kk,
            "Recall@K": y_true[np.argsort(-proba)[:kk]].sum() / max(y_true.sum(), 1),
            "Precision@K": y_true[np.argsort(-proba)[:kk]].mean(),
        }
        for kk in ks
    ])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=curve["K"], y=curve["Precision@K"], mode="lines+markers",
        name="Precision@K", line=dict(color="#ED7D31", width=3),
    ))
    fig.add_trace(go.Scatter(
        x=curve["K"], y=curve["Recall@K"], mode="lines+markers",
        name="Recall@K", line=dict(color="#4472C4", width=3),
    ))
    fig.add_hline(y=base_rate, line_dash="dash", line_color="#999",
                  annotation_text=f"baseline {base_rate:.1%}")
    fig.update_layout(
        title="Ranking quality — precision collapses as the list grows",
        xaxis_title="K — customers called first",
        yaxis_title="metric",
        height=420,
    )
    st.plotly_chart(fig, width="stretch")

    st.info(
        f"**Reading this:** the top-{k} list captures **{recall:.1%}** of all "
        f"churners while being **{lift:.1f}×** better than random. Accuracy is "
        f"deliberately not shown — on a {base_rate:.0%} base rate a model that "
        f"always predicts \"no churn\" scores {1 - base_rate:.0%} accuracy and is useless."
    )

# ---------------------------------------------------------------------------
# tab 2 - survival
# ---------------------------------------------------------------------------
with tab_surv:
    st.subheader("Survival — when will they leave?")
    st.markdown(
        "Right-censoring: `churn=1` = event observed (left at `tenure`), "
        "`churn=0` = censored (still active). Kaplan-Meier and Cox PH handle "
        "the censoring correctly; classification ignores it."
    )

    s = survival_cached(dataset)

    c1, c2, c3 = st.columns(3)
    c1.metric("Cox concordance", f"{s['concordance']:.3f}")
    med = s["all_summary"].get("median_survival_time")
    time_unit = "months" if dataset == "telco" else "years"
    c2.metric("Median survival (all)", "∞ (not reached)" if med is None else f"{med:.0f} {time_unit}")
    c3.metric(f"Horizon for dynamic ranking", f"{s['horizon']} {time_unit}")

    # Kaplan-Meier plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=s["km_all"].index, y=s["km_all"].iloc[:, 0],
        name="All customers", line=dict(color="#4472C4", width=3),
    ))
    for lab, sf in s["km_curves"].items():
        fig.add_trace(go.Scatter(
            x=sf.index, y=sf.iloc[:, 0],
            name=f"{s['seg_col']} = {lab}", line=dict(width=2),
        ))
    fig.update_layout(
        title=f"Kaplan-Meier survival curve — {spec['label']}",
        xaxis_title=f"Tenure ({time_unit})",
        yaxis_title="Survival probability",
        height=480,
    )
    st.plotly_chart(fig, width="stretch")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### KM summary by segment")
        st.dataframe(
            pd.DataFrame([
                {"group": ("All" if lab == "all" else f"{s['seg_col']} = {lab}"), **v}
                for lab, v in s["summary"].items()
            ]),
            width="stretch",
            hide_index=True,
        )
    with col_b:
        st.markdown("#### Cox proportional hazards")
        st.dataframe(
            s["cox_table"][["covariate", "hazard_ratio", "hr_lower_95",
                            "hr_upper_95", "p_value"]],
            width="stretch",
            hide_index=True,
        )

    st.markdown("#### Log-rank test — do the segments survive differently?")
    lr_rows = [
        {"comparison": pair,
         "test_statistic": r["test_statistic"],
         "p_value": r["p_value"],
         "significant": "yes" if r["p_value"] < 0.05 else "no"}
        for pair, r in s["logrank"].items()
    ]
    st.dataframe(pd.DataFrame(lr_rows), width="stretch", hide_index=True)

    st.info(
        "**Honest caveats:** `TotalCharges` is excluded from the Cox covariates "
        "(it is ≈ `MonthlyCharges × tenure`, a function of the time variable). "
        "Median survival is `None` when the curve never reaches 50% within the "
        "observation window — that is a finding, not a bug."
    )

# ---------------------------------------------------------------------------
# tab 3 - AI experiments
# ---------------------------------------------------------------------------
with tab_ai:
    st.subheader("AI experiments — text as a feature")

    _, df = load(dataset)
    services = spec["services"]

    st.markdown("### 1 · Naive Bayes on the service bundle")
    if services:
        toks = []
        for _, row in df.iterrows():
            doc = []
            for c in services:
                v = str(row[c]).lower()
                if v in ("yes", "1", "true"):
                    doc.append("has_" + c.lower())
                elif v in ("no", "0", "false"):
                    doc.append("no_" + c.lower())
                else:
                    doc.append("na_" + c.lower())
            toks.append(" ".join(doc))
        df = df.assign(bundle_doc=toks)

        X_txt = df["bundle_doc"]
        y = df["churn"]
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_txt, y, test_size=0.2, random_state=42, stratify=y
        )
        vec = TfidfVectorizer(token_pattern=r"(?u)\b[\w_]+\b")
        Xtr = vec.fit_transform(X_tr)
        Xte = vec.transform(X_te)
        nb = MultinomialNB().fit(Xtr, y_tr)
        nb_proba = nb.predict_proba(Xte)[:, 1]

        num_cols = [c for c in ("tenure", "MonthlyCharges", "TotalCharges")
                    if c in df.columns]
        sc = StandardScaler().fit(df[num_cols])
        lr = LogisticRegression(max_iter=2000, class_weight="balanced")
        lr.fit(sc.transform(df.loc[X_tr.index, num_cols]), y_tr)
        lr_proba = lr.predict_proba(sc.transform(df.loc[X_te.index, num_cols]))[:, 1]

        c1, c2 = st.columns(2)
        c1.metric("NB on bundle — ROC-AUC", f"{roc_auc_score(y_te, nb_proba):.4f}")
        c1.metric("NB on bundle — PR-AUC", f"{average_precision_score(y_te, nb_proba):.4f}")
        c2.metric("Numeric logistic — ROC-AUC", f"{roc_auc_score(y_te, lr_proba):.4f}")
        c2.metric("Numeric logistic — PR-AUC", f"{average_precision_score(y_te, lr_proba):.4f}")

        st.success(
            "The service bundle alone carries real signal — but it is **worse** "
            "than the numeric model. Saying that is more valuable than only "
            "showing the winning configuration."
        )
    else:
        st.info("This dataset has no service-bundle columns.")

    st.divider()
    st.markdown("### 2 · TF-IDF on support notes (SYNTHETIC field)")
    st.warning(
        "The real dataset has no contact-history column. The notes below are "
        "**synthetic**, built from the churn label so the experiment is about "
        "the *method*, not a claimed finding."
    )

    rng = np.random.default_rng(42)
    notes_churned = [
        "called about billing error again, third time this month",
        "service keeps dropping, considering other providers",
        "requested cancellation, contract too expensive",
        "no response from support for two weeks, very unhappy",
        "want to downgrade plan, not happy with recent price increase",
    ]
    notes_loyal = [
        "quick question about invoice, resolved quickly",
        "asked how to add a line, helpful agent",
        "routine check on data usage",
        "thanks for the help setting up autopay",
        "question about roaming charges while travelling",
    ]

    def make_note(is_churn):
        pool = notes_churned if is_churn else notes_loyal
        return rng.choice(pool) + rng.choice(["", " thanks", " please advise", " asap", ""])

    df = df.assign(support_note=[make_note(c) for c in df["churn"]])

    X_tr, X_te, y_tr, y_te = train_test_split(
        df["support_note"], df["churn"], test_size=0.2, random_state=42, stratify=df["churn"]
    )
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, stop_words="english")
    Xtr = vec.fit_transform(X_tr)
    Xte = vec.transform(X_te)
    nb = MultinomialNB().fit(Xtr, y_tr)
    nb_proba = nb.predict_proba(Xte)[:, 1]

    c1, c2 = st.columns(2)
    c1.metric("TF-IDF + NB — ROC-AUC", f"{roc_auc_score(y_te, nb_proba):.4f}")
    c2.metric("TF-IDF + NB — PR-AUC", f"{average_precision_score(y_te, nb_proba):.4f}")

    feat_names = np.array(vec.get_feature_names_out())
    logp = nb.feature_log_prob_
    top_churn = feat_names[np.argsort(logp[1] - logp[0])[-12:]][::-1]
    st.markdown("**Tokens most associated with churn:** " + ", ".join(f"`{t}`" for t in top_churn))

# ---------------------------------------------------------------------------
# tab 4 - LLM design sketch
# ---------------------------------------------------------------------------
with tab_llm:
    st.subheader("Prompt-as-ranker — the LLM design, honestly costed")
    st.markdown(
        "The LLM is **not** the scoring engine (GBM is cheaper, faster, already "
        "at 0.84 AUC). Its real value is generating the retention action from "
        "the customer's situation."
    )

    st.code("""inputs  : ranked at-risk list (top 500, from the GBM)
          + customer record (contract, charges, tenure, bundle)
          + retention playbook
prompt  : "Customer X: month-to-month, $95/mo, 3 months tenure, no tech support,
           2 support tickets. Churn risk 0.87. Pick ONE action from the playbook,
           justify it in 2 sentences, and draft the outreach email."
output  : {action, justification, email_draft}
guard   : validate the JSON schema; never send without human approval""", language="text")

    PLAYBOOK = {
        "onboarding_blitz": "90-day onboarding check-in + usage review",
        "migrate_autopay": "move from electronic check to auto-pay",
        "annual_contract": "offer annual contract with a discount",
        "bundle_security": "attach online security / backup at a promo rate",
    }

    def mock_llm_rank(customer: dict) -> dict:
        """Deterministic stand-in for an LLM call: rule-based, auditable."""
        risk = customer["churn_risk"]
        if customer["contract"] == "Month-to-month" and risk > 0.6:
            action, why = "annual_contract", "month-to-month + high risk = lock-in offer"
        elif customer["payment"] == "Electronic check":
            action, why = "migrate_autopay", "electronic check is the top churn payment method"
        elif customer["tenure"] < 12:
            action, why = "onboarding_blitz", "first-year customers churn most"
        else:
            action, why = "bundle_security", "deepen the product attachment"
        return {
            "customer_id": customer["id"],
            "churn_risk": round(risk, 3),
            "action": action,
            "action_text": PLAYBOOK[action],
            "justification": why,
            "email_draft": (f"Hi, we noticed you might benefit from {PLAYBOOK[action]}. "
                            f"{why.capitalize()}."),
        }

    def validate(output: dict) -> list:
        errs = []
        if set(output) != {"customer_id", "churn_risk", "action", "action_text",
                           "justification", "email_draft"}:
            errs.append("schema mismatch")
        if output["action"] not in PLAYBOOK:
            errs.append("action not in playbook")
        if not (0.0 <= output["churn_risk"] <= 1.0):
            errs.append("risk out of range")
        if re.search(r"\b(guarantee|promise|100%)\b", output["email_draft"], re.I):
            errs.append("over-claiming language")
        return errs

    st.markdown("#### Try it")
    col1, col2, col3 = st.columns([2, 1, 1])
    cid = col1.text_input("customer_id", "7590-VHVEG")
    risk = col2.number_input("churn_risk", 0.0, 1.0, 0.87, 0.01)
    tenure = col3.number_input("tenure", 0, 72, 3, 1)
    contract = st.selectbox("contract", ["Month-to-month", "One year", "Two year"])
    payment = st.selectbox("payment", ["Electronic check", "Mailed check",
                                       "Bank transfer", "Credit card"])

    sample = {"id": cid, "churn_risk": float(risk), "contract": contract,
              "payment": payment, "tenure": int(tenure)}
    out = mock_llm_rank(sample)
    errs = validate(out)

    st.json(out)
    if errs:
        st.error("Guardrail errors: " + ", ".join(errs))
    else:
        st.success("Guardrail errors: none")
    st.warning("auto-send allowed: **False** — human approval is mandatory")

    st.divider()
    st.markdown("### Verdict — when to use the LLM, when not to")
    st.dataframe(pd.DataFrame([
        {"Task": "Score 7,000 customers", "Tool": "GBM", "Why": "0.84 AUC, ms, free, reproducible"},
        {"Task": "Rank the top-500 list", "Tool": "GBM", "Why": "ranking metrics, deterministic"},
        {"Task": "Explain a score", "Tool": "Logistic coefficients", "Why": "directly interpretable"},
        {"Task": "Draft the retention email", "Tool": "LLM", "Why": "needs language, not maths"},
        {"Task": "Pick the action per customer", "Tool": "LLM + playbook", "Why": "judgement over context"},
        {"Task": "Send the email", "Tool": "Human", "Why": "always"},
    ]), width="stretch", hide_index=True)
