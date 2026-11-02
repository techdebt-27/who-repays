#!/usr/bin/env python3
"""Plot exact file-level ECDFs for SF and NSF from saved boxplot data.

Install: python -m pip install numpy pandas matplotlib
Example:
  python PLOT_ATD_FILE_LEVEL_ECDF.py --input FILE_LEVEL_boxplot_data.csv \
    --layouts 3x3 1x9 --xscale both

Input columns: project, Key, group, path, metric, intro, repayment, delta.
Groups: Self-Fixed, Non-Self-Fixed.
Observation: one (project, Key, path, metric); equal weight per issue-file row.
Only finite paired measurements are included. ALL signed deltas are retained,
including zeros and extreme values. Missing values are never converted to zero.

F_g(x) = number of valid deltas <= x in group g / n_g.
Exact right-continuous step curves; duplicate values are counted, not dropped.
No binning, sampling, smoothing, rounding, or zero tolerance is applied.
Y is shared (0 to 1). X limits are shared by SF/NSF within each metric only.
Symlog changes X-axis spacing, not values, sample membership, or ECDF heights.

Exports PDF/PNG, exact ECDF point counts, summary, and settings JSON.
These are descriptive plots; no significance tests or confidence bands.
No zero-percentage annotations or lettered panel labels are added.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, ScalarFormatter, NullLocator, FixedLocator


METRICS = {
    "fan_in": "Fan-In",
    "fan_out": "Fan-Out",
    "cyclomatic_complexity": "Cyclomatic\ncomplexity",
    "cognitive_complexity": "Cognitive\ncomplexity",
    "cbo": "CBO",
    "rfc_understand": "RFC proxy",
    "lcom": "LCOM",
    "dit": "DIT",
    "noc": "NOC",
}
GROUPS = ("Self-Fixed", "Non-Self-Fixed")
COLORS = ("#4477AA", "#EE7733")
LINESTYLES = ("-", "--")
IDS = ["project", "Key", "group", "path", "metric"]
VALUES = ["intro", "repayment", "delta"]
LAYOUTS = {"3x3": (3, 3), "1x9": (1, 9), "2x5": (2, 5), "2x4": (2, 4), "9x1": (9, 1)}


def load_data(path):
    required = IDS + VALUES
    data = pd.read_csv(
        path, usecols=lambda name: name in required,
        dtype={name: str for name in IDS},
        float_precision="round_trip", low_memory=False,
    )
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Use FILE_LEVEL_boxplot_data.csv.")
    for column in IDS:
        if data[column].isna().any():
            raise ValueError(f"Missing identity in {column}")
        data[column] = data[column].str.strip()
        if data[column].eq("").any():
            raise ValueError(f"Empty identity in {column}")
    if not set(data["group"]).issubset(GROUPS):
        raise ValueError("Expected Self-Fixed / Non-Self-Fixed group labels")
    if (data.groupby(["project", "Key"])["group"].nunique() > 1).any():
        raise ValueError("Conflicting group labels within the same issue")
    if data.duplicated(["project", "Key", "path", "metric"]).any():
        raise ValueError("Duplicate issue/file/metric rows would double-count observations")
    for column in VALUES:
        data[column] = pd.to_numeric(data[column], errors="raise")
        if np.isinf(data[column].to_numpy(dtype=float)).any():
            raise ValueError(f"Infinite values in {column}")
    paired = data[["intro", "repayment"]].notna().all(axis=1)
    if data.loc[paired, "delta"].isna().any():
        raise ValueError("Missing delta for a complete pair")
    if data.loc[~paired, "delta"].notna().any():
        raise ValueError("Delta present without a complete pair")
    if not np.allclose(
        data.loc[paired, "delta"],
        data.loc[paired, "repayment"] - data.loc[paired, "intro"],
        rtol=1e-10, atol=1e-9,
    ):
        raise ValueError("Delta does not equal repayment minus introduction")
    return data


def exact_ecdf(values):
    """Unique x values, frequencies, and right-continuous F(x)."""
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("ECDF requires finite values")
    x, counts = np.unique(values, return_counts=True)
    f = np.cumsum(counts, dtype=float) / len(values) if len(values) else np.array([])
    return x, counts, f


def prepare_data(data, metrics):
    curves, summaries, point_tables = {}, [], []
    for metric in metrics:
        for group in GROUPS:
            raw = data.loc[data["metric"].eq(metric) & data["group"].eq(group)]
            valid = raw.dropna(subset=VALUES)
            values = valid["delta"].to_numpy(dtype=float)
            x, counts, f = exact_ecdf(values)
            curves[(metric, group)] = (x, f)
            if len(values):
                cumulative = np.cumsum(counts)
                point_tables.append(pd.DataFrame({
                    "metric": metric, "group": group, "delta": x,
                    "count_at_delta": counts, "cumulative_count": cumulative,
                    "n": len(values), "F_before": (cumulative-counts)/len(values),
                    "F_at_delta": f,
                }))
            summaries.append({
                "metric": metric, "group": group,
                "n_issue_file_observations": len(values),
                "n_issues": len(valid[["project", "Key"]].drop_duplicates()),
                "n_missing_pairs": len(raw)-len(valid),
                "n_negative": int((values < 0).sum()),
                "n_zero": int((values == 0).sum()),
                "n_positive": int((values > 0).sum()),
                "F_before_zero": float((values < 0).mean()) if len(values) else np.nan,
                "F_at_zero": float((values <= 0).mean()) if len(values) else np.nan,
                "min_delta": float(values.min()) if len(values) else np.nan,
                "median_delta": float(np.median(values)) if len(values) else np.nan,
                "max_delta": float(values.max()) if len(values) else np.nan,
            })
    columns = ["metric", "group", "delta", "count_at_delta", "cumulative_count", "n", "F_before", "F_at_delta"]
    points = pd.concat(point_tables, ignore_index=True) if point_tables else pd.DataFrame(columns=columns)
    return curves, pd.DataFrame(summaries), points


def grid_spec(layout, n_metrics):
    rows, cols = LAYOUTS[layout]
    if layout == "1x9":
        cols = n_metrics
    elif layout == "9x1":
        rows = n_metrics
    if n_metrics > rows*cols:
        raise ValueError(f"Layout {layout} holds {rows*cols} metrics, but {n_metrics} were selected")
    return rows, cols


def plot_limits(ax, arrays):
    """Pad in the current display scale, keeping zero and every value visible."""
    populated = [x for x in arrays if len(x)]
    if not populated:
        return -1.0, 1.0
    low = min(0.0, min(float(x[0]) for x in populated))
    high = max(0.0, max(float(x[-1]) for x in populated))
    if low == high:
        return -1.0, 1.0
    transform = ax.xaxis.get_transform()
    limits = transform.transform(np.array([low, high], dtype=float))
    pad = .05*(limits[1]-limits[0])
    padded = transform.inverted().transform(limits + np.array([-pad, pad]))
    return float(padded[0]), float(padded[1])


def plot_ecdfs(curves, metrics, layout, scale, args, output):
    rows, cols = grid_spec(layout, len(metrics))
    if rows == 1:
        size = (max(3.6, 1.72*cols), 3.0)
    elif cols == 1:
        size = (3.6, 2.0*rows + .8)
    else:
        size = (7.16 if cols <= 3 else 9.0, 2.0*rows + .8)
    style = {
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": .65,
        "path.simplify": False,
    }
    with plt.rc_context(style):
        fig, axes = plt.subplots(rows, cols, figsize=size, squeeze=False,
                                 sharex=False, sharey=True)
        xlimits = {}
        for index, metric in enumerate(metrics):
            ax = axes.flat[index]
            if scale == "symlog":
                ax.set_xscale("symlog", base=10, linthresh=args.linthresh)
            else:
                ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
                formatter = ScalarFormatter(useOffset=False)
                formatter.set_powerlimits((-3, 4))
                ax.xaxis.set_major_formatter(formatter)
            ax.xaxis.set_minor_locator(NullLocator())
            pair = [curves[(metric, group)] for group in GROUPS]
            left, right = plot_limits(ax, [x for x, _ in pair])
            xlimits[metric] = [left, right]
            for group_index, (x, f) in enumerate(pair):
                if not len(x):
                    ax.text(.03, .91-.11*group_index,
                            ("SF" if group_index == 0 else "NSF")+": no valid pairs",
                            transform=ax.transAxes, fontsize=7,
                            color=COLORS[group_index])
                    continue
                # F(left)=0; F(right)=1. With steps-post the jump is at x.
                ax.step(np.r_[left, x, right], np.r_[0.0, f, 1.0],
                        where="post", color=COLORS[group_index],
                        linestyle=LINESTYLES[group_index], linewidth=1.25,
                        alpha=.95, zorder=3+group_index)
            ax.set_xlim(left, right)
            if scale == "symlog":
                # Retain zero and a few well-separated decade labels. This only
                # thins axis labels; no observations or curve points are removed.
                transform = ax.xaxis.get_transform()
                bounds = transform.transform(np.array([left, right]))
                span = bounds[1] - bounds[0]
                candidates = ax.xaxis.get_major_locator().tick_values(left, right)
                candidates = [float(t) for t in candidates
                              if np.isfinite(t) and left <= t <= right and t != 0]
                selected = [0.0]
                positions = [float(transform.transform(np.array([0.0]))[0])]
                for tick in sorted(candidates, key=abs, reverse=True):
                    position = float(transform.transform(np.array([tick]))[0])
                    if all(abs(position - other) >= .23 * span for other in positions):
                        selected.append(tick)
                        positions.append(position)
                ax.xaxis.set_major_locator(FixedLocator(sorted(selected)))
            ax.set_ylim(-.02, 1.02)
            ax.set_yticks([0, .25, .5, .75, 1])
            ax.set_yticklabels(["0", "0.25", "0.50", "0.75", "1"])
            ax.set_xlabel(METRICS[metric], fontweight="bold", labelpad=6)
            ax.axvline(0, color=".55", linestyle=":", linewidth=.7, zorder=1)
            ax.grid(axis="y", color=".90", linewidth=.6, zorder=0)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(length=3, width=.6)
        for ax in axes.flat[len(metrics):]:
            ax.set_axis_off()
        legend = [Line2D([0], [0], color=c, linestyle=s, linewidth=1.5, label=g)
                  for c, s, g in zip(COLORS, LINESTYLES, ["SF", "NSF"])]
        fig.legend(handles=legend, loc="upper center", ncol=2, frameon=False,
                   bbox_to_anchor=(.5, 1.0))
        fig.supylabel("Cumulative proportion  F(Δ)", x=.008, fontsize=9)
        scale_note = ("Linear X scale" if scale == "linear" else
                      f"Symlog X scale; linear within ±{args.linthresh:g}")
        fig.supxlabel("Δ = Repayment − Introduction; all paired issue–files, including zeros.\n"
                      + scale_note + "; X ranges vary by metric.", fontsize=8, y=.008)
        # Reserve space for the shared legend, Y label and two-line footer.
        fig.tight_layout(rect=(.025, .12 if rows == 1 else .07, 1, .90 if rows == 1 else .95),
                         pad=.6, w_pad=.7, h_pad=1.1)
        rendered = f"{rows}x{cols}"
        selection = f"ALL_{len(metrics)}" if len(metrics) == len(METRICS) else f"SELECTED_{len(metrics)}"
        stem = f"FILE_LEVEL_{selection}_METRICS_ECDF_SF_vs_NSF_ALL_FILES_{rendered}_{scale.upper()}"
        for extension in ("png", "pdf"):
            fig.savefig(output / f"{stem}.{extension}", dpi=args.dpi, facecolor="white")
        plt.close(fig)
    return {"layout": rendered, "xscale": scale, "file_stem": stem, "xlimits": xlimits}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                    formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", "--plot-data", dest="input", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Default: FILE_LEVEL_ECDF beside input CSV")
    parser.add_argument("--layouts", nargs="+", choices=list(LAYOUTS), default=["3x3"])
    parser.add_argument("--exclude", nargs="+", choices=list(METRICS), default=[])
    parser.add_argument("--xscale", choices=["linear", "symlog", "both"], default="both")
    parser.add_argument("--linthresh", type=float, default=1.0,
                        help="Symlog linear region: -linthresh to +linthresh (default 1)")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args(argv)
    if not args.input.is_file():
        parser.error(f"Input file not found: {args.input}")
    if not np.isfinite(args.linthresh) or args.linthresh <= 0:
        parser.error("--linthresh must be positive and finite")
    if args.dpi < 72:
        parser.error("--dpi must be at least 72")
    metrics = [m for m in METRICS if m not in args.exclude]
    if not metrics:
        parser.error("At least one metric must remain")
    layouts = list(dict.fromkeys(args.layouts))
    try:
        for layout in layouts:
            grid_spec(layout, len(metrics))
        data = load_data(args.input)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if not data.loc[data["metric"].isin(metrics), "delta"].notna().any():
        parser.error("No valid paired observations for the selected metrics")
    output = args.output or args.input.parent / "FILE_LEVEL_ECDF"
    output.mkdir(parents=True, exist_ok=True)
    curves, summary, points = prepare_data(data, metrics)
    files = [output / name for name in ["FILE_LEVEL_ECDF_points.csv", "FILE_LEVEL_ECDF_summary.csv", "FILE_LEVEL_ECDF_settings.json"]]
    if args.input.resolve() in {p.resolve() for p in files}:
        parser.error("Output would overwrite input CSV")
    points.to_csv(files[0], index=False, float_format="%.17g")
    summary.to_csv(files[1], index=False, float_format="%.17g")
    scales = ["linear", "symlog"] if args.xscale == "both" else [args.xscale]
    figures = []
    for layout in layouts:
        for scale in scales:
            print(f"Plotting {layout}: {scale}", flush=True)
            figures.append(plot_ecdfs(curves, metrics, layout, scale, args, output))
    digest = hashlib.sha256()
    with args.input.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    settings = {
        "input": str(args.input.resolve()), "input_sha256": digest.hexdigest(),
        "metrics": metrics, "excluded_metrics": args.exclude,
        "outcome": "delta = repayment - introduction",
        "included": "All valid paired deltas, including zeros and extremes",
        "weighting": "Equal weight per issue-file observation within each metric/group",
        "ecdf": "F(x) = count(delta <= x) / n; exact right-continuous steps",
        "ties": "Exact equality; no rounding or zero tolerance",
        "shared_y": True, "shared_x_across_metrics": False,
        "linthresh": args.linthresh, "dpi": args.dpi, "figures": figures,
        "inference": "Descriptive only; no tests or confidence bands",
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "matplotlib": matplotlib.__version__},
    }
    files[2].write_text(json.dumps(settings, indent=2), encoding="utf-8")
    print(summary[["metric", "group", "n_issue_file_observations", "n_issues"]].to_string(index=False))
    print(f"\nSaved ECDF figures, points and summary to: {output.resolve()}")


if __name__ == "__main__":
    main()
