#!/usr/bin/env python3
# render_offline_wandb.py  —  offline W&B .wandb extractor + HTML report
from __future__ import annotations
import sys, json, argparse, math
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
import plotly.graph_objects as go
import plotly.offline as po
import yaml
import numpy as np

# read the .wandb local event store (no login, no cloud)
from wandb.sdk.internal import datastore
from wandb.proto import wandb_internal_pb2


def find_wandb_target(p: Path) -> tuple[Path, Path]:
    p = p.resolve()
    if p.is_file() and p.suffix == ".wandb":
        return p.parent, p
    if p.is_dir():
        if p.name.startswith("offline-run-"):
            runs = list(p.glob("run-*.wandb"))
            if not runs: sys.exit(f"No run-*.wandb inside {p}")
            return p, runs[0]
        wb = p / "wandb"
        if wb.is_dir():
            offs = sorted(wb.glob("offline-run-*"), key=lambda x: x.stat().st_mtime, reverse=True)
            if offs:
                runs = list(offs[0].glob("run-*.wandb"))
                if runs: return offs[0], runs[0]
        if p.name == "wandb":
            offs = sorted(p.glob("offline-run-*"), key=lambda x: x.stat().st_mtime, reverse=True)
            if offs:
                runs = list(offs[0].glob("run-*.wandb"))
                if runs: return offs[0], runs[0]
    sys.exit(f"Couldn't locate run-*.wandb under: {p}")

def flatten(d: Dict[str, Any], prefix="") -> Dict[str, Any]:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict): out.update(flatten(v, key))
        else: out[key] = v
    return out

def read_from_wandb(wandb_file: Path) -> tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    store = datastore.DataStore()
    store.open_for_scan(str(wandb_file))
    rows, summary, config = [], {}, {}

    def upd_cfg(cfg_pb):
        for upd in cfg_pb.update:
            try: config.update(flatten({upd.key: json.loads(upd.value_json)}))
            except Exception: pass

    while True:
        data = store.scan_data()
        if data is None: break
        rec = wandb_internal_pb2.Record(); rec.ParseFromString(data)
        t = rec.WhichOneof("record_type")

        if t == "history":
            row, step = {}, None
            for it in rec.history.item:
                key = it.key
                try: val = json.loads(it.value_json)
                except Exception: continue
                if key == "_step": step = val
                else:
                    if isinstance(val, dict): row.update(flatten({key: val}))
                    else: row[key] = val
            row["_step"] = step if step is not None else len(rows)
            rows.append(row)

        elif t == "summary":
            for upd in rec.summary.update:
                try: summary.update(flatten({upd.key: json.loads(upd.value_json)}))
                except Exception: pass

        elif t == "config": upd_cfg(rec.config)
        elif t == "run" and rec.run.HasField("config"): upd_cfg(rec.run.config)

    store.close()
    if not rows: sys.exit("No metric history found in .wandb (run may have ended before first log).")
    df = pd.DataFrame(rows).sort_values("_step").reset_index(drop=True)
    return df, summary, config

def coerce_numeric_cols(df: pd.DataFrame) -> pd.DataFrame:
    # try to convert any column that *looks* numeric
    for c in list(df.columns):
        if c == "_step":  # keep step
            df[c] = pd.to_numeric(df[c], errors="coerce")
            continue
        s = df[c]
        if s.dtype == object:
            df[c] = pd.to_numeric(s, errors="coerce")
    return df

def sanitize_df(df: pd.DataFrame) -> pd.DataFrame:
    cols = list(df.columns)
    if len(cols) == 2 and "_step" in cols:
        other = [c for c in cols if c != "_step"][0]
        if not other or str(other).strip() in {"", "0", "1", "Unnamed: 0"}:
            df = df.rename(columns={other: "loss"})
    # drop all-NaN columns
    for c in list(df.columns):
        if df[c].isna().all(): df = df.drop(columns=[c])
    print(df)
    return df

def pick_x(df: pd.DataFrame) -> str:
    for k in ("tokens","num_tokens","_step","global_step","step"):
        if k in df.columns: return k
    df["_idx"] = range(len(df)); return "_idx"

def choose_y(df: pd.DataFrame, prefer: List[str] | None) -> List[str]:
    if prefer:
        chosen = [c for c in prefer if c in df.columns]
        if chosen: return chosen
    preferred = ["train/loss","loss","train_log_ppl","train/log_ppl","log_ppl","lr","learning_rate"]
    found = [c for c in preferred if c in df.columns]
    if found: return found
    numeric = []
    for c in df.columns:
        if c.startswith("_"): continue
        v = pd.to_numeric(df[c], errors="coerce")
        if v.notna().any(): numeric.append(c)
    return numeric[:8]

def maybe_smooth(series: pd.Series, smooth: float|int|None) -> pd.Series:
    if smooth is None: return series
    if isinstance(smooth, float) and 0.0 < smooth < 1.0:
        alpha = 1.0 - smooth  # e.g. 0.1 if smooth=0.9
        return series.ewm(alpha=alpha).mean()
    if isinstance(smooth, int) and smooth >= 2:
        return series.rolling(window=smooth, min_periods=1).mean()
    return series

def make_report(offline_dir: Path, df: pd.DataFrame, summary: Dict[str, Any], config: Dict[str, Any],
                ykeys: List[str] | None, out_html: Path | None, logy: bool, smooth: float|int|None,
                symlog: bool = False, clipq: tuple[float,float] | None = None) -> Path:
    xk = pick_x(df)
    ycols = choose_y(df, ykeys)
    out_html = out_html or (offline_dir / "report.html")

    divs = []
    for y in ycols:
        yraw = pd.to_numeric(df[y], errors="coerce")
        xraw = pd.to_numeric(df[xk], errors="coerce")
        mask = yraw.notna() & xraw.notna()
        xvals = xraw[mask].astype(float)
        yvals = yraw[mask].astype(float)
        if len(yvals) == 0:
            continue

        # --- robustify: optional quantile clipping before transforms ---
        ywork = yvals.copy()
        if clipq is not None:
            lo, hi = clipq
            qlo, qhi = ywork.quantile([lo, hi])
            ywork = ywork.clip(qlo, qhi)

        # --- transform: symlog OR logy OR none ---
        y_for_plot = ywork
        yaxis_type = None
        y_label = y
        if symlog:
            # signed log10 transform (keeps sign; spread both sides)
            y_for_plot = np.sign(ywork) * np.log10(1.0 + np.abs(ywork))
            y_label = f"{y} (symlog10)"
        elif logy:
            ok = ywork > 0
            xvals = xvals[ok]
            y_for_plot = ywork[ok]
            yaxis_type = "log"
            y_label = f"{y} (log)"

        # optional smoothing
        yplot = maybe_smooth(pd.Series(y_for_plot.values, index=y_for_plot.index), smooth)

        # build figure
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=xvals, y=yplot, mode="lines", name=y))
        layout = dict(
            title=y, xaxis_title=xk, yaxis_title=y_label,
            template="plotly_white", height=350,
            margin=dict(l=50, r=10, t=50, b=50),
        )
        if yaxis_type:
            layout["yaxis_type"] = yaxis_type
        fig.update_layout(**layout)

        # annotate with stats on the **clipped, untransformed** data
        try:
            y_min, y_max = float(ywork.min()), float(ywork.max())
            fig.add_annotation(xref="paper", yref="paper", x=1.0, y=1.15,
                               text=f"n={len(ywork)}  min={y_min:.4g}  max={y_max:.4g}",
                               showarrow=False, xanchor="right", font=dict(size=10, color="#666"))
        except Exception:
            pass

        divs.append(po.plot(fig, include_plotlyjs=False, output_type="div"))

    cfg_yaml = yaml.safe_dump(config, sort_keys=True, allow_unicode=True) if config else "(no config found)"
    summ_json = json.dumps(summary, indent=2) if summary else "(no summary found)"
    charts_html = "".join(f"<div class='chart'>{d}</div>" for d in divs) if divs else "<p>No plotted metrics.</p>"

    html = f"""<!doctype html>
<html><head>
<meta charset="utf-8"/>
<title>W&B Offline Report</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
body {{ margin:0; font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }}
.container {{ display:grid; grid-template-columns: 360px 1fr; gap:20px; }}
.sidebar {{ background:#f7f7f9; padding:16px; height:100vh; overflow:auto; border-right:1px solid #eee; }}
.content {{ padding:10px 20px; }}
pre {{ white-space: pre-wrap; word-break: break-word; font-size:12px; }}
h2 {{ margin-top:0; }}
.chart {{ margin: 10px 0 20px; background:#fff; border:1px solid #eee; border-radius:8px; padding:10px; }}
.meta {{ font-size:12px; color:#666; margin-bottom:10px; }}
</style>
</head>
<body>
<div class="container">
  <div class="sidebar">
    <h2>Run</h2>
    <div class="meta"><b>dir:</b> {offline_dir}</div>
    <h3>Summary</h3>
    <pre>{summ_json}</pre>
    <h3>Config</h3>
    <pre>{cfg_yaml}</pre>
  </div>
  <div class="content">
    <h2>Metrics</h2>
    {charts_html}
  </div>
</div>
</body></html>"""
    out_html.write_text(html, encoding="utf-8")
    return out_html

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="outputs/<JOBID> | .../wandb | .../offline-run-* | run-*.wandb")
    ap.add_argument("--out", type=str, default=None, help="Output HTML (default: <offline_run>/report.html)")
    ap.add_argument("--y", type=str, default=None, help="Comma-separated metric keys to plot (e.g. loss,log_ppl,lr)")
    ap.add_argument("--logy", action="store_true", help="Log scale for y-axis")
    ap.add_argument("--smooth", type=str, default=None,
                    help="Smoothing: float in (0,1) for EMA (e.g. 0.9), or int window for rolling mean")
    ap.add_argument("--symlog", action="store_true",
                help="Plot signed log10: sign(y)*log10(1+abs(y)) so negatives show up")
    ap.add_argument("--clipq", type=str, default=None,
                    help="Quantile clipping as low,high (e.g. 0.01,0.99) before plotting")
    args = ap.parse_args()

    base = Path(args.path)
    offline_dir, wandb_file = find_wandb_target(base)
    print(f"Reading: {wandb_file}")

    df, summary, config = read_from_wandb(wandb_file)
    df = coerce_numeric_cols(df)
    df = sanitize_df(df)

    # Save CSV for your own analysis
    csv_path = offline_dir / "history.csv"
    print(f"using history file at {csv_path}")
    df.to_csv(csv_path, index=False)
    print(f"Wrote: {csv_path}  (rows={len(df)}, cols={len(df.columns)})")
    print("Columns:", sorted(df.columns.tolist()))

    ykeys = [s.strip() for s in args.y.split(",")] if args.y else None
    smooth = None
    if args.smooth is not None:
        try:
            smooth = float(args.smooth) if "." in args.smooth else int(args.smooth)
        except Exception:
            print("Ignoring --smooth (could not parse).")

    out_html = Path(args.out).resolve() if args.out else None
    clipq = None
    if args.clipq:
        lo, hi = [float(x) for x in args.clipq.split(",")]
        clipq = (lo, hi)

    html_path = make_report(
        offline_dir, df, summary, config,
        ykeys, out_html, args.logy, smooth,
        symlog=args.symlog, clipq=clipq
    )
    print(f"Wrote HTML report: {html_path}\nOpen it with your browser (double-click, `xdg-open` or `open`).")

if __name__ == "__main__":
    main()

