"""
Stage - Dashboard preview (churn project).

Renders the exact Power BI report design as a self-contained HTML file so the
report can be reviewed in a browser before (or without) assembling the .pbix.
Reads the Power BI data pack written by `src.powerbi_export.py` for the same
dataset (`powerbi/data/<dataset>/`); every figure is embedded as base64 so the
file works offline and can be shared.

  powerbi/dashboard_preview_telco.html   (or dashboard_preview_bank.html)
"""
from __future__ import annotations

import base64
from datetime import datetime
from pathlib import Path

import pandas as pd

from .config import Paths

FIGURES = [
    ("Churn_by_segment.png", "Churn rate by segment"),
    ("Retention_curve_by_tenure.png", "Retention curve by tenure band"),
    ("Top_numeric_features.png", "Top numeric features vs churn"),
    ("Model_comparison.png", "Churn model comparison"),
    ("Logistic_feature_importance.png", "Logistic feature importance"),
    ("Survival_curve.png", "Kaplan-Meier survival curve"),
]


def _img64(paths: Paths, dataset: str, name: str) -> str:
    # Always read from THIS dataset's regenerated figures folder, so the
    # preview can never show figures of a different dataset.
    src = _figure_paths(paths, dataset) / name
    if not src.exists():
        return ""
    return "data:image/png;base64," + base64.b64encode(src.read_bytes()).decode()


def _table(df: pd.DataFrame, money: tuple[str, ...] = (),
           pct: tuple[str, ...] = ()) -> str:
    if df is None or df.empty:
        return "<p class='empty'>No data</p>"
    head = "".join(f"<th>{c}</th>" for c in df.columns)
    rows = []
    for _, r in df.iterrows():
        cells = []
        for c in df.columns:
            v = r[c]
            if pd.isna(v):
                cells.append("<td>—</td>")
            elif c in money:
                cells.append(f"<td class='num'>{float(v):,.0f}</td>")
            elif c in pct:
                cells.append(f"<td class='num'>{float(v):,.2f}%</td>")
            elif isinstance(v, float):
                cells.append(f"<td class='num'>{v:,.3f}</td>")
            else:
                cells.append(f"<td>{v}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


CSS = """
:root { --blue:#1f4e78; --accent:#2e75b6; --ink:#252525; --muted:#6b6b6b; --line:#e3e3e3; }
* { box-sizing:border-box; }
body { margin:0; font-family:'Segoe UI',Arial,sans-serif; color:var(--ink); background:#f5f6f8; }
header { background:var(--blue); color:#fff; padding:18px 26px; }
header h1 { margin:0; font-size:20px; font-weight:600; }
header p { margin:4px 0 0; font-size:12.5px; opacity:.82; }
.wrap { padding:20px 26px 40px; }
.cards { display:grid; grid-template-columns:repeat(6,1fr); gap:14px; margin-bottom:22px; }
.card { background:#fff; border-left:4px solid var(--accent); border-radius:3px;
        padding:12px 14px; box-shadow:0 1px 3px rgba(0,0,0,.08); }
.card .k { font-size:11.5px; color:var(--muted); text-transform:uppercase; letter-spacing:.3px; }
.card .v { font-size:22px; font-weight:600; margin:6px 0 2px; color:var(--blue); }
.card .s { font-size:11px; color:var(--muted); }
.grid { display:grid; grid-template-columns:1fr 1fr; gap:18px; }
.panel { background:#fff; border-radius:3px; padding:14px 16px; box-shadow:0 1px 3px rgba(0,0,0,.08); }
.panel h3 { margin:0 0 10px; font-size:13.5px; color:var(--blue); font-weight:600;
            border-bottom:1px solid var(--line); padding-bottom:8px; }
.panel img { width:100%; height:auto; display:block; }
table { width:100%; border-collapse:collapse; font-size:12px; }
th { background:var(--blue); color:#fff; text-align:left; padding:6px 8px; font-weight:600; }
td { padding:5px 8px; border-bottom:1px solid var(--line); }
tbody tr:nth-child(even) { background:#f2f6fa; }
td.num { text-align:right; font-variant-numeric:tabular-nums; }
.empty { color:var(--muted); font-style:italic; }
footer { padding:14px 26px; font-size:11px; color:var(--muted); }
@media print { .cards{grid-template-columns:repeat(3,1fr);} .grid{grid-template-columns:1fr;} }
"""


@staticmethod
def _figure_paths(paths: Paths, dataset: str) -> Path:
    """Figures are written per-dataset so a preview for dataset A is never
    polluted by figures from a previous full-run for dataset B."""
    return paths.root / "powerbi" / "data" / dataset / "figures"


def _regenerate_figures(paths: Paths, dataset: str) -> None:
    """Generate the 6 review figures for THIS dataset into powerbi/data/<dataset>/figures
    (via the shared report.render_figures) and reuse them in build_html."""
    from src.clean import clean
    from src.load_data import load_raw
    from src.model import fit_models
    from src.report import render_figures

    fig_dir = _figure_paths(paths, dataset)
    raw = load_raw(paths, dataset)
    cleaned = clean(raw, dataset)
    models, results = fit_models(cleaned, dataset, seed=42)
    render_figures(dataset, raw, cleaned, results, fig_dir)

    # Live-assert the deliverable is fixed: every figure must be a real PNG
    # written into the dataset figures folder (no dead <img src=''>).
    n_png = len(list(fig_dir.glob("*.png")))
    assert n_png == 6, f"preview for {dataset} emitted {n_png} figures, expected 6"


def _read(paths: Paths, dataset: str, name: str) -> pd.DataFrame | None:
    src = paths.root / "powerbi" / "data" / dataset / name
    if not src.exists():
        return None
    return pd.read_csv(src)


def _cards(paths: Paths, dataset: str) -> list[tuple[str, str, str]]:
    impact = _read(paths, dataset, "kpi_business_impact.csv")
    mc = _read(paths, dataset, "kpi_model_comparison.csv")
    surv = _read(paths, dataset, "kpi_survival_summary.csv")
    dim = _read(paths, dataset, "dim_customer.csv")

    churn_rate = f"{dim['churn'].mean():.2%}" if dim is not None else "—"
    n_cust = f"{len(dim):,}" if dim is not None else "—"

    ra = "—"
    if impact is not None and pd.notna(impact["revenue_at_risk"].iloc[0]):
        ra = f"{impact['revenue_at_risk'].iloc[0]:,.0f}"

    best = "—"
    if mc is not None and len(mc):
        b = mc.loc[mc["roc_auc"].idxmax()]
        best = f"{b['model']} {b['roc_auc']:.3f}"

    conc = "—"
    if surv is not None:
        conc = f"{surv['survival_at_12'].iloc[0]:.3f}" if "survival_at_12" in surv else "—"

    return [
        ("Customers", n_cust, "in the analysed base"),
        ("Churn rate", churn_rate, "of the base churned"),
        ("Revenue at risk / mo", ra, "top-500 at-risk customers"),
        ("Best model (ROC-AUC)", best, "same features, same split"),
        ("Survival @ 12 months", conc, "Kaplan-Meier, all customers"),
        ("Datasets", "2", "telco + bank, one registry"),
    ]


def build_html(paths: Paths, dataset: str = "telco") -> str:
    _regenerate_figures(paths, dataset)
    cards = _cards(paths, dataset)
    card_html = "".join(
        f"<div class='card'><div class='k'>{k}</div>"
        f"<div class='v'>{v}</div><div class='s'>{s}</div></div>"
        for k, v, s in cards
    )
    panels = "".join(
        f"<div class='panel'><h3>{title}</h3>"
        f"<img src='{_img64(paths, dataset, name)}' alt='{title}'></div>"
        for name, title in FIGURES
    )

    seg = _read(paths, dataset, "kpi_churn_by_segment.csv")
    ten = _read(paths, dataset, "kpi_tenure_bands.csv")
    surv = _read(paths, dataset, "kpi_survival_summary.csv")
    cox = _read(paths, dataset, "kpi_cox_hazards.csv")
    lift = _read(paths, dataset, "kpi_targeting_lift.csv")

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Customer Churn Analysis ({dataset}) — Dashboard Preview</title>
<style>{CSS}</style></head>
<body>
<header>
  <h1>CUSTOMER CHURN ANALYSIS</h1>
  <p>Power BI report design preview ({dataset}) &nbsp;•&nbsp; built {datetime.now():%Y-%m-%d %H:%M}</p>
</header>
<div class="wrap">
  <div class="cards">{card_html}</div>
  <div class="grid">{panels}</div>

  <div class="panel" style="margin-top:18px">
    <h3>Churn rate by segment</h3>
    {_table(seg, pct=("churn_rate",))}
  </div>

  <div class="grid" style="margin-top:18px">
    <div class="panel"><h3>Churn by tenure band</h3>
      {_table(ten, pct=("churn_rate",))}</div>
    <div class="panel"><h3>Kaplan-Meier survival summary</h3>
      {_table(surv)}</div>
  </div>

  <div class="grid" style="margin-top:18px">
    <div class="panel"><h3>Cox proportional hazards (top 10)</h3>
      {_table(cox.head(10) if cox is not None else None)}</div>
    <div class="panel"><h3>Targeting lift — ranked list</h3>
      {_table(lift, pct=("churn_rate_pct", "lift_vs_all_pp"))}</div>
  </div>
</div>
<footer>Source: churn-project/powerbi/data/{dataset}/ • same design as the .pbix build pack</footer>
</body></html>"""


def run(paths: Paths | None = None, dataset: str = "telco") -> dict[str, str]:
    """Write the HTML preview. Returns {preview: path}."""
    paths = paths or Paths()
    out = paths.root / "powerbi" / f"dashboard_preview_{dataset}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_html(paths, dataset), encoding="utf-8")
    return {"preview": str(out)}


if __name__ == "__main__":
    for kind, path in run().items():
        print(f"{kind}: {path}")
