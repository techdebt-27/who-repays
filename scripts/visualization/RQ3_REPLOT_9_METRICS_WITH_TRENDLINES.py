#!/usr/bin/env python3
"""Replot nine RQ3 file-level metrics with colored OLS trend lines.

Run without repeating Git analysis:
  python RQ3_REPLOT_9_METRICS_WITH_TRENDLINES.py \
    --input-csv /path/to/rq3_file_level_nine_metrics.csv

Works with the output of RQ3_FILE_CHANGE_ANALYSIS_9_METRICS_CK_FILE_LEVEL.py.
The colored straight lines are unadjusted OLS visual summaries, *not* Spearman
fits or evidence for significant correlations. Raw data remain unchanged.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

X_COL = 'file_commit_count_intro_to_payment'
GROUPS = ('Self-Fixed', 'Non-Self-Fixed')
COLORS = {'Self-Fixed': '#1f77b4', 'Non-Self-Fixed': '#ff7f0e'}
DISPLAY_GROUPS = {'Self-Fixed': 'Self-fixed', 'Non-Self-Fixed': 'Non-self-fixed'}
METRICS = (
    ('fan_in', 'Fan-In'),
    ('fan_out', 'Fan-Out'),
    ('cyclomatic_complexity', 'Cyclomatic complexity'),
    ('cognitive_complexity', 'Cognitive complexity'),
    ('cbo', 'CBO'),
    ('rfc_proxy', 'RFC proxy (CountDeclMethodAll)'),
    ('lcom', 'LCOM'),
    ('dit', 'DIT'),
    ('noc', 'NOC'),
)
CATEGORIES = {
    'dependency': METRICS[:2],
    'complexity': METRICS[2:4],
    'coupling': METRICS[4:6],
    'cohesion_inheritance': METRICS[6:9],
}


def fit_linear_trend(ax, x, y, color):
    """Plot exactly the kind of first-degree OLS trend line used in old RQ3 code."""
    finite = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x, dtype=float)[finite]
    y = np.asarray(y, dtype=float)[finite]
    if x.size < 3 or np.unique(x).size < 2:
        return None
    try:
        with warnings.catch_warnings():
            rank_warning = getattr(getattr(np, 'exceptions', np), 'RankWarning', Warning)
            warnings.simplefilter('error', rank_warning)
            slope, intercept = np.polyfit(x, y, deg=1)
        xx = np.linspace(x.min(), x.max(), 200)
        (line,) = ax.plot(xx, slope * xx + intercept, color=color,
                          linewidth=2.3, alpha=0.98, zorder=5)
        return line
    except (FloatingPointError, ValueError, np.linalg.LinAlgError, rank_warning):
        return None


def render_panel(ax, frame, metric, label, marker_size=8, marker_alpha=.17):
    col = f'delta_{metric}'
    ax.axhline(0, linestyle='--', color='#777777', linewidth=.9,
               alpha=.7, zorder=1)
    for group in GROUPS:
        subset = frame.loc[frame['group'].eq(group), [X_COL, col]].apply(
            pd.to_numeric, errors='coerce')
        subset = subset.replace([np.inf, -np.inf], np.nan).dropna()
        if subset.empty:
            continue
        x = subset[X_COL].to_numpy(dtype=float)
        y = subset[col].to_numpy(dtype=float)
        ax.scatter(x, y, s=marker_size, color=COLORS[group],
                   alpha=marker_alpha, edgecolors='none', zorder=2,
                   label=DISPLAY_GROUPS[group])
        fit_linear_trend(ax, x, y, COLORS[group])
    ax.text(.025, .965, label, va='top', ha='left', fontsize=9,
            fontweight='bold', transform=ax.transAxes,
            bbox=dict(boxstyle='square,pad=.2', facecolor='white',
                      edgecolor='none', alpha=.86), zorder=6)
    ax.grid(axis='y', color='#bbbbbb', alpha=.28, linewidth=.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def save_figure(frame, metrics, output_dir, stem, ncols=3, marker_size=8,
                marker_alpha=.17):
    from matplotlib.lines import Line2D
    nrows = (len(metrics) + ncols - 1) // ncols
    if ncols == 3:
        size=(12.4, 9.6)
    else:
        size=(9.0, 4.0*nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=size, sharex=False,
                             sharey=False, squeeze=False, layout='constrained')
    for ax, (metric, label) in zip(axes.flat, metrics):
        render_panel(ax, frame, metric, label, marker_size, marker_alpha)
        ax.set_xlabel('File commit count', fontsize=9)
        ax.set_ylabel('Change (repayment − introduction)', fontsize=9)
        ax.tick_params(labelsize=8)
    for ax in axes.flat[len(metrics):]:
        ax.set_axis_off()
    handles = [Line2D([0], [0], marker='o', markersize=5, linestyle='-',
                      linewidth=2, color=COLORS[g], label=DISPLAY_GROUPS[g])
               for g in GROUPS]
    fig.legend(handles=handles, loc='outside lower center', ncol=2,
               frameon=False, fontsize=9)
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        name = output_dir / f'{stem}.{ext}'
        fig.savefig(name, dpi=300, bbox_inches='tight', facecolor='white')
        print(f'Saved: {name}')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--input-csv', required=True, type=Path,
                        help='Output file-level CSV from nine-metric RQ3 analysis')
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--marker-size', type=float, default=8)
    parser.add_argument('--marker-alpha', type=float, default=.17)
    parser.add_argument('--overview-only', action='store_true')
    args = parser.parse_args()
    if not args.input_csv.is_file():
        parser.error(f'Input CSV not found: {args.input_csv}')
    frame = pd.read_csv(args.input_csv, low_memory=False)
    required = {'group', X_COL} | {f'delta_{m}' for m, _ in METRICS}
    missing = required - set(frame.columns)
    if missing:
        parser.error('Missing columns: ' + ', '.join(sorted(missing)))
    print('Input rows:', len(frame))
    print('Group counts:', frame['group'].value_counts(dropna=False).to_dict())
    print('All 9 metric deltas present as columns; finite counts:')
    for metric, _ in METRICS:
        valid = pd.to_numeric(frame[f'delta_{metric}'], errors='coerce')
        print(f'  {metric:24s} {np.isfinite(valid).sum()}')
    out = args.output_dir or args.input_csv.parent / 'RQ3_TRENDLINES'
    plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'font.family': 'DejaVu Sans', 'axes.unicode_minus': True})
    save_figure(frame, METRICS, out, 'rq3_all_nine_scatter_3x3_WITH_TRENDLINES',
                3, args.marker_size, args.marker_alpha)
    if not args.overview_only:
        for name, metrics in CATEGORIES.items():
            ncols = 2 if len(metrics) <= 2 else 3
            save_figure(frame, metrics, out, f'rq3_{name}_scatter_WITH_TRENDLINES',
                        ncols, args.marker_size, args.marker_alpha)


if __name__ == '__main__':
    main()
