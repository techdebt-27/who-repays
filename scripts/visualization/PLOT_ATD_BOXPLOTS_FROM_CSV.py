#!/usr/bin/env python3
"""Recreate ATD SF/NSF boxplots from saved CSVs only (no statistical tests).

INSTALL: python -m pip install numpy pandas matplotlib
Place this script beside ATD_DIRECT_COMPARISON_RESULTS, then run:
  python PLOT_ATD_BOXPLOTS_FROM_CSV.py
  python PLOT_ATD_BOXPLOTS_FROM_CSV.py --layouts 3x3 1x9
  python PLOT_ATD_BOXPLOTS_FROM_CSV.py --input-dir /path/to/results --layouts 3x3
To hide both complexity metrics (plotting only):
  python PLOT_ATD_BOXPLOTS_FROM_CSV.py --layouts 1x9 2x4 --exclude cyclomatic_complexity cognitive_complexity

Inputs from ATD analysis v2.3+:
  PRIMARY_boxplot_data.csv       raw issue-level deltas (not rounded quartiles)
  PRIMARY_boxplot_metadata.csv   labels, saved Holm p-values, sample counts
Both files must come from the same run; the metadata checksum verifies this.

If neither new CSV exists, the script automatically uses these older outputs:
  issue_metric_deltas.csv and PRIMARY_direct_tests.csv
This allows replotting previous results without rerunning the analysis.
PRIMARY_manuscript_table.csv alone is NOT sufficient to recreate boxplots.

Optional explicit paths:
  --data-csv /path/to/PRIMARY_boxplot_data.csv
  --metadata-csv /path/to/PRIMARY_boxplot_metadata.csv
Always provide both overrides together.

Default outputs: <input CSV folder>/BOXPLOTS_FROM_CSV, in PDF and PNG.
--output changes the output folder. --layouts accepts 3x3, 1x9, 2x4.
Dimensions use rows x columns: 1x9 = 1 row, up to 9 columns; 2x4 = 2 rows,
4 columns. X and Y are shared across all populated panels; unused cells are hidden.
For 1x9, excluded metrics remove their columns completely. Seven selected metrics
produce exactly 1x7, without empty axes or reserved blank columns. The canvas
width scales with the number of visible metrics; output filenames use _1x7.
2x4 holds at most eight metrics: use --exclude or choose 3x3 / 1x9 for all nine.
When --layouts is omitted, all layouts with sufficient capacity are generated.
1x9 uses 14.32 x 2.6 inches for nine metrics (11.14 x 2.6 for seven);
2x4 uses 7.16 x 4.4 inches.
3x3 retains the detailed original layout, sample counts and saved Holm p-values.
1x9 and 2x4 retain compact publication layouts, 9-point text, embedded fonts.
Metric names appear below their respective boxplots, without (a), (b), etc.
There is no overall figure title. The detailed 3x3 layout shows the saved Holm
p-value below each metric name; group sample counts remain inside each panel.
All finite deltas are plotted, including zero values and outliers.
--exclude hides selected metrics from the figure; unused panels are hidden.
It does not change the CSVs, sample values, p-values or nine-test Holm family.
Subset figures use PRIMARY_SELECTED_<N>_METRICS filenames, leaving full figures
intact. The default without --exclude still displays all nine metrics.

This file is self-contained. It does not import the original analysis script,
need SciPy, aggregate extracted entities, bootstrap, or recompute p-values.
Only graphical boxplot summaries are calculated from the saved issue values.
This standalone renderer supports hiding metrics with --exclude and shared
X/Y axes. It accepts CSVs already exported by the analysis script without rerunning it.
Sharing Y preserves original values; it does not standardize different metric units.
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
matplotlib.use('Agg')
import matplotlib.pyplot as plt

VERSION = '1.2.2'
SCRIPT_DIR = Path(__file__).resolve().parent
GROUPS = ('Self-Fixed', 'Non-Self-Fixed')
COLORS = ('#4477AA', '#EE7733')
STRUCTURAL_METRICS = ('fan_in','fan_out','cyclomatic_complexity','cognitive_complexity')
PRIMARY_METRICS = (*STRUCTURAL_METRICS,'cbo','rfc_understand','lcom','dit','noc')


def _boxplot_stem(metrics, layout='3x3'):
    selection = 'ALL_9' if len(metrics) == len(PRIMARY_METRICS) else f'SELECTED_{len(metrics)}'
    stem = f'PRIMARY_{selection}_METRICS_BOXPLOTS_SF_vs_NSF'
    if layout == '1x9':
        layout = f'1x{len(metrics)}'
    return stem if layout == '3x3' else f'{stem}_{layout}'


def _finish_shared_axes(axes, n_metrics):
    """Hide unused panels; show X labels at the bottom and Y labels at left."""
    ncols = axes.shape[1]
    for ax in axes.flat[n_metrics:]:
        ax.set_axis_off()
    for column in range(ncols):
        populated = list(range(column, n_metrics, ncols))
        for i in populated:
            axes.flat[i].tick_params(axis='x', labelbottom=(i == populated[-1]))
    for i in range(n_metrics):
        axes.flat[i].tick_params(axis='y', labelleft=(i % ncols == 0))


def _combined_primary_boxplots_detailed(data, tests, out, metrics=PRIMARY_METRICS):
    """Selected primary signed-delta boxplots; reuse existing Holm results.

    All panels share the same linear y-axis scale. All finite observations, including
    outliers, are displayed. Sample counts are annotated within each panel.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    primary = tests.loc[tests.family == 'primary'].set_index('metric')
    fig, axes = plt.subplots(3, 3, figsize=(14, 12), sharex=True, sharey=True,
                             layout='constrained')
    for index, (ax, metric) in enumerate(zip(axes.flat, metrics)):
        part = data.loc[data.metric == metric]
        arrays = []
        for group in GROUPS:
            values = part.loc[part.group == group, 'delta'].to_numpy(dtype=float)
            arrays.append(values[np.isfinite(values)])
        for position, (values, color) in enumerate(zip(arrays, COLORS), start=1):
            if not len(values):
                ax.text(position, .5, 'No valid data', ha='center', va='center',
                        transform=ax.get_xaxis_transform(), fontsize=9)
                continue
            ax.boxplot([values], positions=[position], widths=.5,
                       patch_artist=True, showfliers=True, whis=1.5,
                       boxprops=dict(facecolor=color, alpha=.7),
                       medianprops=dict(color='black', linewidth=1.6),
                       flierprops=dict(marker='o', markersize=3,
                                       markerfacecolor=color, markeredgecolor=color,
                                       alpha=.45))
        row = primary.loc[metric] if metric in primary.index else None
        label = row['label'] if row is not None else metric
        if metric == 'rfc_understand':
            label = 'RFC proxy\n(CountDeclMethodAll)'
        pvalue = row['p_holm'] if row is not None else np.nan
        ptext = 'NA (not tested)' if pd.isna(pvalue) else f'{pvalue:.3g}'
        ax.set_xlabel(f'{label}\nHolm p = {ptext}', fontsize=11, labelpad=8)
        ax.set_xticks([1, 2], ['SF', 'NSF'])
        ax.text(.5, .96, f'SF n={len(arrays[0])}; NSF n={len(arrays[1])}',
                transform=ax.transAxes, ha='center', va='top', fontsize=9)
        ax.set_xlim(.4, 2.6)
        ax.axhline(0, color='gray', linewidth=.8, linestyle='--', zorder=0)
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
    _finish_shared_axes(axes, len(metrics))
    fig.supylabel('Delta = Repayment − Introduction', fontsize=11)
    fig.supxlabel('One observation per issue; shared y-axis scale (original metric units). '
                  'Boxes: Q1–Q3; line: median; whiskers: 1.5 × IQR; points: outliers.\n'
                  'Holm correction across nine primary tests. Dashed line: zero change.',
                  fontsize=10)
    try:
        for ext in ('png', 'pdf'):
            fig.savefig(out / f'{_boxplot_stem(metrics)}.{ext}',
                        dpi=300, bbox_inches='tight')
    finally:
        plt.close(fig)


def _compact_primary_boxplots(data, tests, out, layout, metrics=PRIMARY_METRICS):
    """Publication alternatives; layout names denote rows x columns.

    Draw directly at final size, preserving exact PDF page dimensions.
    All displayed metrics use one shared linear scale and all finite issue deltas.
    Font selection uses installed fonts; no font download is required.
    """
    from matplotlib import font_manager, ticker
    layouts = {'1x9': (1, 9, (14.32, 2.6)),
               '2x4': (2, 4, (7.16, 4.4))}
    nrows, ncols, size = layouts[layout]
    if layout == '1x9':
        # Allocate only visible columns, instead of hiding axes that still
        # reserve space in the figure's GridSpec.
        ncols = len(metrics)
        size = (size[0] * ncols / 9, size[1])
    # TrueType fonts avoid CFF/OpenType embedding mismatches in PDF readers.
    available = {font.name for font in font_manager.fontManager.ttflist
                 if Path(font.fname).suffix.lower() == '.ttf'}
    font = next((name for name in ('Times New Roman', 'Arial', 'Liberation Serif',
                                  'DejaVu Serif')
                 if name in available), 'DejaVu Serif')
    primary = tests.loc[tests.family == 'primary'].set_index('metric')
    style = {'font.family': font, 'font.size': 9, 'axes.titlesize': 9,
             'axes.labelsize': 9, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
             'axes.linewidth': .6, 'pdf.fonttype': 42, 'ps.fonttype': 42,
             'text.usetex': False, 'savefig.bbox': None}
    with plt.rc_context(style):
        fig, axes = plt.subplots(nrows, ncols, figsize=size, squeeze=False,
                                 sharex=True, sharey=True, layout='constrained')
        fig.get_layout_engine().set(h_pad=.025, w_pad=.04,
                                    hspace=.035, wspace=.055)
        try:
            for index, (ax, metric) in enumerate(zip(axes.flat, metrics)):
                part = data.loc[data.metric == metric]
                for position, (group, color) in enumerate(zip(GROUPS, COLORS), 1):
                    values = part.loc[part.group == group, 'delta'].to_numpy(dtype=float)
                    values = values[np.isfinite(values)]
                    if not len(values):
                        ax.text(position, .5, 'No data', ha='center', va='center',
                                transform=ax.get_xaxis_transform(), fontsize=9)
                        continue
                    ax.boxplot([values], positions=[position], widths=.48,
                               patch_artist=True, showfliers=True, whis=1.5,
                               boxprops=dict(facecolor=color, edgecolor='black',
                                             linewidth=.65, alpha=.7),
                               medianprops=dict(color='black', linewidth=1.0),
                               whiskerprops=dict(color='black', linewidth=.65),
                               capprops=dict(color='black', linewidth=.65),
                               flierprops=dict(marker='o', markersize=1.7,
                                               markerfacecolor='none',
                                               markeredgecolor=color,
                                               markeredgewidth=.45, alpha=.65))
                label = primary.loc[metric, 'label'] if metric in primary.index else metric
                if metric == 'rfc_understand':
                    label = 'RFC proxy'
                if metric in ('cyclomatic_complexity', 'cognitive_complexity'):
                    label = label.replace(' complexity', '\ncomplexity')
                ax.set_xlabel(label, fontsize=9, labelpad=6)
                ax.set_xticks([1, 2], ['SF', 'NSF'])
                ax.set_xlim(.4, 2.6)
                ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
                formatter = ticker.ScalarFormatter(useOffset=False)
                formatter.set_powerlimits((-3, 4))
                ax.yaxis.set_major_formatter(formatter)
                # Keep scientific-notation multiplier inside the plot, away from titles.
                ax.yaxis.get_offset_text().set_visible(False)
                ax.tick_params(axis='both', length=2, width=.6, pad=2)
                ax.axhline(0, color='.45', linewidth=.6, linestyle='--', zorder=0)
                ax.grid(axis='y', color='.9', linewidth=.4)
                ax.set_axisbelow(True)
                ax.spines[['top', 'right']].set_visible(False)
            _finish_shared_axes(axes, len(metrics))
            fig.supylabel('Change (Δ)', fontsize=9)
            fig.canvas.draw()
            for ax in axes.flat[:len(metrics):ncols]:
                multiplier = ax.yaxis.get_major_formatter().get_offset()
                if multiplier:
                    ax.text(.98, .96, multiplier, transform=ax.transAxes,
                            ha='right', va='top', fontsize=9)
            # No tight bounding-box crop: retain the intended print dimensions.
            stem = _boxplot_stem(metrics, layout)
            for ext in ('pdf', 'png'):
                fig.savefig(Path(out) / f'{stem}.{ext}', dpi=300,
                            facecolor='white', bbox_inches=None)
        finally:
            plt.close(fig)


def resolve_layouts(layouts, n_metrics):
    """Reject undersized explicit layouts before writing any output files."""
    capacities = {'3x3': 9, '1x9': 9, '2x4': 8}
    if layouts is None:
        return [name for name, capacity in capacities.items() if n_metrics <= capacity]
    invalid = set(layouts) - set(capacities)
    if invalid:
        raise ValueError(f'Unknown combined boxplot layouts: {sorted(invalid)}')
    for layout in layouts:
        if n_metrics > capacities[layout]:
            raise ValueError(f'Layout {layout} holds at most {capacities[layout]} metrics, '
                             f'but {n_metrics} are selected. Use --exclude '
                             'cyclomatic_complexity cognitive_complexity to plot seven metrics, '
                             'or choose 3x3 / 1x9 for all nine.')
    return list(dict.fromkeys(layouts))


def combined_primary_boxplots(data, tests, out, layouts=None, exclude=()):
    """Export requested layouts; statistical calculations are not repeated."""
    unknown = set(exclude) - set(PRIMARY_METRICS)
    if unknown:
        raise ValueError(f'Unknown excluded metrics: {sorted(unknown)}')
    metrics = tuple(metric for metric in PRIMARY_METRICS if metric not in exclude)
    if not metrics:
        raise ValueError('At least one metric must remain visible.')
    layouts = resolve_layouts(layouts, len(metrics))
    Path(out).mkdir(parents=True, exist_ok=True)
    for layout in dict.fromkeys(layouts):
        if layout == '3x3':
            _combined_primary_boxplots_detailed(data, tests, out, metrics)
        else:
            _compact_primary_boxplots(data, tests, out, layout, metrics)


def load_boxplot_inputs(data_path, metadata_path):
    """Validate the saved cohort and reuse statistical annotations as stored."""
    data_path, metadata_path = Path(data_path), Path(metadata_path)
    for path in (data_path, metadata_path):
        if not path.is_file():
            raise FileNotFoundError(f'CSV not found: {path}')
    # round_trip recovers float64 values written with %.17g by the exporter.
    data = pd.read_csv(data_path, float_precision='round_trip',
                       dtype={'project': str, 'Key': str, 'group': str, 'metric': str})
    tests = pd.read_csv(metadata_path, float_precision='round_trip',
                        dtype={'metric': str, 'label': str,
                               'boxplot_schema_version': str, 'data_sha256': str})
    for frame, columns, name in (
        (data, {'project','Key','group','metric','delta'}, data_path.name),
        (tests, {'metric','label','family','SF_n','NSF_n','p_holm'}, metadata_path.name),
    ):
        missing = columns - set(frame.columns)
        if missing:
            raise ValueError(f'{name}: missing columns {sorted(missing)}. '
                             'Use issue-level data and direct-test metadata, not the manuscript table.')
    if 'boxplot_schema_version' in tests:
        if set(tests.boxplot_schema_version) != {'1.0'}:
            raise ValueError('Unsupported/incomplete boxplot CSV schema; expected version 1.0.')
        if 'data_sha256' not in tests:
            raise ValueError('Boxplot metadata is missing its data checksum.')
    if 'data_sha256' in tests:
        actual = hashlib.sha256(data_path.read_bytes()).hexdigest()
        if set(tests.data_sha256) != {actual}:
            raise ValueError('Data checksum mismatch. Use the data and metadata CSVs '
                             'from the same analysis run, without editing their contents.')
    tests = tests.loc[tests.family == 'primary'].copy()
    if tests.metric.duplicated().any() or set(tests.metric) != set(PRIMARY_METRICS):
        raise ValueError('Metadata must contain exactly one row for each of the nine primary metrics.')
    if tests.label.isna().any():
        raise ValueError('Missing metric labels in metadata.')
    # Old issue_metric_deltas.csv also includes exploratory metrics.
    data = data.loc[data.metric.isin(PRIMARY_METRICS)].copy()
    if data[['project','Key','group','metric']].isna().any().any():
        raise ValueError('Missing issue identity or group in the plotting data.')
    unexpected = set(data.group) - set(GROUPS)
    if unexpected:
        raise ValueError(f'Unexpected group labels: {sorted(unexpected)}')
    if data.duplicated(['project','Key','metric']).any():
        raise ValueError('Duplicate issue/metric rows would give some issues extra weight.')
    if (data.groupby(['project','Key'])['group'].nunique() > 1).any():
        raise ValueError('Conflicting SF/NSF labels for the same issue.')
    data['delta'] = pd.to_numeric(data.delta, errors='raise')
    if np.isinf(data.delta.to_numpy(dtype=float)).any():
        raise ValueError('Infinite deltas are not valid boxplot inputs.')
    # Missing values remain missing; a genuinely observed zero stays in the plot.
    if 'included_in_plot' in data:
        supplied = data.included_in_plot.astype(str).str.lower()
        actual = data.delta.notna().map({True: 'true', False: 'false'})
        if not supplied.equals(actual):
            raise ValueError('included_in_plot does not match the finite delta values.')
    for name in ('SF_n', 'NSF_n', 'p_holm'):
        tests[name] = pd.to_numeric(tests[name], errors='raise')
    if ((tests.p_holm.dropna() < 0) | (tests.p_holm.dropna() > 1)).any():
        raise ValueError('Holm p-values must be missing or between zero and one.')
    for row in tests.to_dict('records'):
        part = data.loc[data.metric == row['metric']]
        for group, name in zip(GROUPS, ('SF_n', 'NSF_n')):
            actual = int(part.loc[part.group == group, 'delta'].notna().sum())
            expected = row[name]
            if not np.isfinite(expected) or expected != actual:
                raise ValueError(f"Sample count mismatch for {row['metric']} / {group}: "
                                 f'CSV has {actual}, metadata reports {expected}. '
                                 'Use files from the same analysis run.')
    return data, tests


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input-dir', type=Path,
                        default=SCRIPT_DIR / 'ATD_DIRECT_COMPARISON_RESULTS')
    parser.add_argument('--data-csv', type=Path)
    parser.add_argument('--metadata-csv', type=Path)
    parser.add_argument('--output', type=Path,
                        help='Default: BOXPLOTS_FROM_CSV beside the input data CSV')
    parser.add_argument('--layouts', nargs='+', choices=['3x3','1x9','2x4'],
                        default=None,
                        help='Default: all fitting layouts; 1x9 shrinks to the visible metric count; 2x4 holds at most eight')
    parser.add_argument('--exclude', nargs='+', choices=PRIMARY_METRICS, default=[],
                        help='Hide these metrics from figures only; CSVs and Holm correction stay unchanged')
    args = parser.parse_args(argv)
    visible_metrics = [metric for metric in PRIMARY_METRICS if metric not in args.exclude]
    if not visible_metrics:
        parser.error('At least one metric must remain visible.')
    try:
        layouts = resolve_layouts(args.layouts, len(visible_metrics))
    except ValueError as exc:
        parser.error(str(exc))
    if (args.data_csv is None) != (args.metadata_csv is None):
        parser.error('Provide --data-csv and --metadata-csv together.')
    if args.data_csv is not None:
        data_path, metadata_path = args.data_csv, args.metadata_csv
    else:
        data_path = args.input_dir / 'PRIMARY_boxplot_data.csv'
        metadata_path = args.input_dir / 'PRIMARY_boxplot_metadata.csv'
        if not data_path.exists() and not metadata_path.exists():
            data_path = args.input_dir / 'issue_metric_deltas.csv'
            metadata_path = args.input_dir / 'PRIMARY_direct_tests.csv'
            print('Using existing issue-level data and direct-test CSVs (legacy outputs).')
    try:
        data, tests = load_boxplot_inputs(data_path, metadata_path)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    output = args.output if args.output is not None else data_path.parent / 'BOXPLOTS_FROM_CSV'
    output.mkdir(parents=True, exist_ok=True)
    combined_primary_boxplots(data, tests, output, layouts, args.exclude)
    rendered_layouts = [f'1x{len(visible_metrics)}' if layout == '1x9' else layout
                        for layout in layouts]
    manifest = {
        'plot_script_version': VERSION, 'layouts': layouts,
        'rendered_layouts': rendered_layouts,
        'visible_metrics': visible_metrics, 'excluded_metrics': sorted(set(args.exclude)),
        'holm_family_size': len(PRIMARY_METRICS),
        'shared_x': True, 'shared_y': True, 'show_outliers': True,
        'delta_definition': 'repayment - introduction',
        'statistical_tests_recomputed': False,
        'python': platform.python_version(), 'numpy': np.__version__,
        'pandas': pd.__version__, 'matplotlib': matplotlib.__version__,
        'inputs': [{'path': str(path.resolve()),
                    'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                   for path in (data_path, metadata_path)],
    }
    (output / 'boxplot_render_manifest.json').write_text(
        json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print('Loaded existing results; no hypothesis tests or bootstrap were run.')
    print(f'Showing {len(visible_metrics)} metrics; saved Holm p-values retain the nine-test family.')
    print(tests.loc[tests.metric.isin(visible_metrics),['label','SF_n','NSF_n','p_holm']].to_string(index=False))
    print(f'Requested layouts: {", ".join(layouts)}; rendered: {", ".join(rendered_layouts)}')
    print(f'PDF/PNG output: {output.resolve()}')


if __name__ == '__main__':
    main()


