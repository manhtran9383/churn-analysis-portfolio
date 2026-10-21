"""
Headless smoke test for the Streamlit demo.

Runs the whole app script with `streamlit.testing.v1.AppTest` (no browser, no
port) and asserts it executes end to end with no exception and the expected
widgets rendered.

Run:  python -m pytest tests/test_streamlit_app.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _run_app(dataset: str = "telco"):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=300)
    at.run()
    if at.exception:
        for e in at.exception:
            print(e.value)
            print(e.stack_trace)
    return at


def test_app_runs_without_exception():
    at = _run_app()
    assert not at.exception, "streamlit_app.py raised an exception"


def test_four_tabs_present():
    at = _run_app()
    labels = {t.label for t in at.tabs}
    assert any("Ranking" in l for l in labels)
    assert any("Survival" in l for l in labels)
    assert any("AI experiments" in l for l in labels)
    assert any("LLM" in l for l in labels)


def test_ranking_metrics_match_notebook():
    at = _run_app()
    metrics = {m.label: m.value for m in at.metric}
    # values are read from the executed notebook output
    assert metrics["Recall@K"] == "73.5%"
    assert metrics["Precision@K"] == "55.0%"
    assert metrics["Lift"] == "2.07×"


def test_survival_concordance():
    at = _run_app()
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Cox concordance"] == "0.840"


def test_no_runtime_errors_or_exceptions():
    at = _run_app()
    assert at.error == []
