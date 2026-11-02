#!/usr/bin/env python3
"""Standalone file-level SF/NSF boxplots for the nine primary ATD metrics.

Input: entity_metrics.csv exported by ATD_ALL_METRICS_DIRECT_SF_VS_NSF.py.
This is the entity-level export, NOT issue_metric_deltas.csv or
PRIMARY_boxplot_data.csv (both are already aggregated to issue level).

Granularity: one observation per (project, Key, canonical Java file, metric).
File Fan-In, Fan-Out, cyclomatic and cognitive complexity use the matched
file-level values as exported. CK class metrics (CBO, RFC proxy, LCOM, DIT,
NOC) use the arithmetic mean of classes in each canonical file whose metric
is available at BOTH introduction and repayment. A delta is the repayment
mean minus the introduction mean. Missing values are never treated as zero.
The same file in two different issues yields two issue-file observations.

Plots describe file-level distributions. They deliberately do not display the
issue-level Holm p-values: those tests use a different unit of analysis.
Files within an issue can be correlated; the script does not infer significance.
Each metric has its own Y axis, shared by SF and NSF within its panel. The
two views show all paired files and, separately, files with nonzero changes.
Outlier markers are omitted by default for readable central distributions;
no observations are removed when computing the boxes. Metric labels appear
beneath panels. The changed-only view is conditional and is labeled as such.

Install: python -m pip install numpy pandas matplotlib
Run alongside the ATD_DIRECT_COMPARISON_RESULTS folder:
  python PLOT_ATD_FILE_LEVEL_BOXPLOTS.py --layouts 1x9
  python PLOT_ATD_FILE_LEVEL_BOXPLOTS.py --layouts 1x9 2x4 --exclude cyclomatic_complexity cognitive_complexity
  python PLOT_ATD_FILE_LEVEL_BOXPLOTS.py --entity-csv /path/to/entity_metrics.csv --layouts 3x3
To replot previously saved file-level data without repeating aggregation:
  python PLOT_ATD_FILE_LEVEL_BOXPLOTS.py --plot-data /path/to/FILE_LEVEL_boxplot_data.csv --layouts 3x3 1x9
Optional: --view all|changed|both; --showfliers; --y-axis per-metric|shared.

All runs write FILE_LEVEL_boxplot_data.csv (one row per issue-file-metric),
FILE_LEVEL_boxplot_summary.csv (descriptive SF/NSF summaries), PDF/PNG figures,
and file_level_boxplot_manifest.json. Default output next to entity_metrics.csv
in FILE_LEVEL_BOXPLOTS; --output changes it. --layouts defaults to all grids
that fit: 3x3, 1x9, and 2x4 (the last needs <=8 selected metrics). 1x9 adapts
to 1xN after --exclude, so there are no empty reserved columns. Values and
deltas use the original metric units, not standardized scores.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
import posixpath
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

VERSION = '1.2.0'
SCRIPT_DIR = Path(__file__).resolve().parent
GROUPS = ('Self-Fixed', 'Non-Self-Fixed')
COLORS = ('#4477AA', '#EE7733')
STRUCTURAL_METRICS = ('fan_in','fan_out','cyclomatic_complexity','cognitive_complexity')
PRIMARY_METRICS = (*STRUCTURAL_METRICS,'cbo','rfc_understand','lcom','dit','noc')
METRIC_LABELS = {'fan_in':'Fan-In','fan_out':'Fan-Out',
                 'cyclomatic_complexity':'Cyclomatic complexity',
                 'cognitive_complexity':'Cognitive complexity',
                 'cbo':'CBO','rfc_understand':'RFC proxy (CountDeclMethodAll)',
                 'lcom':'LCOM','dit':'DIT','noc':'NOC'}
SOURCE_BY_METRIC = {metric: ('structural' if metric in STRUCTURAL_METRICS else 'ck')
                    for metric in PRIMARY_METRICS}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _boxplot_stem(metrics, layout='3x3', view='all'):
    selection = 'ALL_9' if len(metrics) == len(PRIMARY_METRICS) else f'SELECTED_{len(metrics)}'
    suffix = 'ALL_FILES' if view == 'all' else 'CHANGED_FILES_ONLY'
    stem = f'FILE_LEVEL_{selection}_METRICS_BOXPLOTS_SF_vs_NSF_{suffix}'
    if layout == '1x9':
        layout = f'1x{len(metrics)}'
    return stem if layout == '3x3' else f'{stem}_{layout}'


def _finish_shared_axes(axes, n_metrics, shared_y=False):
    """Hide unused panels; show bottom X labels and all independent Y labels."""
    ncols = axes.shape[1]
    for ax in axes.flat[n_metrics:]:
        ax.set_axis_off()
    for column in range(ncols):
        populated = list(range(column, n_metrics, ncols))
        for i in populated:
            axes.flat[i].tick_params(axis='x', labelbottom=(i == populated[-1]))
    if shared_y:
        for i in range(n_metrics):
            axes.flat[i].tick_params(axis='y', labelleft=(i % ncols == 0))


def _metric_arrays(data, metric, view):
    """Return plotted values and unconditional counts per SF/NSF group."""
    part = data.loc[data.metric == metric]
    arrays, counts = [], []
    for group in GROUPS:
        raw = part.loc[part.group == group, 'delta'].to_numpy(dtype=float)
        full = raw[np.isfinite(raw)]
        changed = full[~np.isclose(full, 0, atol=1e-9, rtol=0)]
        arrays.append(full if view == 'all' else changed)
        counts.append((len(full), len(changed)))
    return arrays, counts


def _combined_primary_boxplots_detailed(data, tests, out, metrics=PRIMARY_METRICS,
                                        view='all', y_axis='per-metric',
                                        showfliers=False):
    """Detailed 3x3 figure with unconditional sample counts in each panel."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    primary = tests.loc[tests.family == 'primary'].set_index('metric')
    fig, axes = plt.subplots(3, 3, figsize=(14, 12), sharex=True,
                             sharey=(y_axis == 'shared'),
                             layout='constrained')
    for index, (ax, metric) in enumerate(zip(axes.flat, metrics)):
        arrays, _ = _metric_arrays(data, metric, view)
        for position, (values, color) in enumerate(zip(arrays, COLORS), start=1):
            if not len(values):
                ax.text(position, .5, 'No changed files' if view == 'changed'
                        else 'No valid data', ha='center', va='center',
                        transform=ax.get_xaxis_transform(), fontsize=9)
                continue
            ax.boxplot([values], positions=[position], widths=.5,
                       patch_artist=True, showfliers=showfliers, whis=1.5,
                       boxprops=dict(facecolor=color, alpha=.7),
                       medianprops=dict(color='black', linewidth=1.6),
                       flierprops=dict(marker='o', markersize=3,
                                       markerfacecolor=color, markeredgecolor=color,
                                       alpha=.45))
        row = primary.loc[metric] if metric in primary.index else None
        label = row['label'] if row is not None else metric
        if metric == 'rfc_understand':
            label = 'RFC proxy\n(CountDeclMethodAll)'
        ax.set_xlabel(label, fontsize=11, labelpad=8)
        ax.set_xticks([1, 2], ['SF', 'NSF'])
        ax.text(.5, .98, f'SF n={len(arrays[0])}; NSF n={len(arrays[1])}',
                transform=ax.transAxes, ha='center', va='top', fontsize=8)
        ax.set_xlim(.4, 2.6)
        ax.margins(y=.22)
        ax.axhline(0, color='gray', linewidth=.8, linestyle='--', zorder=0)
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
    _finish_shared_axes(axes, len(metrics), shared_y=(y_axis == 'shared'))
    fig.supylabel('Delta = Repayment − Introduction', fontsize=11)
    view_note = ('All paired issue–files, including zero changes.' if view == 'all'
                 else 'Only issue–files with nonzero Δ.')
    fliers_note = ('Outliers are shown.' if showfliers else 'Outlier markers are omitted.')
    scale_note = ('Each metric has its own Y scale.' if y_axis == 'per-metric'
                  else 'All metrics share the Y scale.')
    fig.supxlabel(view_note + ' ' + scale_note + ' '
                  'Boxes: Q1–Q3; line: median; whiskers: 1.5 × IQR.\n'
                  + fliers_note + ' Dashed line: zero change. Descriptive only.',
                  fontsize=10)
    try:
        for ext in ('png', 'pdf'):
            fig.savefig(out / f'{_boxplot_stem(metrics, view=view)}.{ext}',
                        dpi=300, bbox_inches='tight')
    finally:
        plt.close(fig)


def _compact_primary_boxplots(data, tests, out, layout, metrics=PRIMARY_METRICS,
                              view='all', y_axis='per-metric', showfliers=False):
    """Publication alternatives; layout names denote rows x columns.

    Draw directly at final size, preserving exact PDF page dimensions.
    SF and NSF use the same Y scale within each metric panel.
    Font selection uses installed fonts; no font download is required.
    """
    from matplotlib import font_manager, ticker
    layouts = {'1x9': (1, 9, (14.32, 3.2)),
               '2x4': (2, 4, (7.16, 5.4))}
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
                                 sharex=True, sharey=(y_axis == 'shared'),
                                 layout='constrained')
        fig.get_layout_engine().set(h_pad=.025, w_pad=.04,
                                    hspace=.035, wspace=.055)
        try:
            for index, (ax, metric) in enumerate(zip(axes.flat, metrics)):
                arrays, _ = _metric_arrays(data, metric, view)
                for position, (group, color) in enumerate(zip(GROUPS, COLORS), 1):
                    values = arrays[position - 1]
                    if not len(values):
                        ax.text(position, .5, 'No changes' if view == 'changed'
                                else 'No data', ha='center', va='center',
                                transform=ax.get_xaxis_transform(), fontsize=9)
                        continue
                    ax.boxplot([values], positions=[position], widths=.48,
                               patch_artist=True, showfliers=showfliers, whis=1.5,
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
                ax.margins(y=.24)
                for i, short in enumerate(('SF', 'NSF')):
                    ax.text(.5, .98 - i * .085,
                            f'{short} n={len(arrays[i])}',
                            transform=ax.transAxes, ha='center', va='top',
                            fontsize=6.2 if layout == '1x9' else 7)
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
            _finish_shared_axes(axes, len(metrics), shared_y=(y_axis == 'shared'))
            fig.supylabel('Change (Δ)', fontsize=9)
            fig.canvas.draw()
            for ax in axes.flat[:len(metrics)]:
                multiplier = ax.yaxis.get_major_formatter().get_offset()
                if multiplier:
                    ax.text(.98, .96, multiplier, transform=ax.transAxes,
                            ha='right', va='top', fontsize=9)
            scale_note = ('Y scale per metric.' if y_axis == 'per-metric'
                          else 'Y scale shared across metrics.')
            fig.supxlabel(
                ('All paired files' if view == 'all' else 'Only files with nonzero Δ')
                + '; '
                + ('outliers shown' if showfliers else 'outlier markers omitted')
                + '; ' + scale_note, fontsize=8)
            # No tight bounding-box crop: retain the intended print dimensions.
            stem = _boxplot_stem(metrics, layout, view)
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


def combined_primary_boxplots(data, tests, out, layouts=None, exclude=(),
                              view='both', y_axis='per-metric',
                              showfliers=False):
    """Export requested layouts; statistical calculations are not repeated."""
    unknown = set(exclude) - set(PRIMARY_METRICS)
    if unknown:
        raise ValueError(f'Unknown excluded metrics: {sorted(unknown)}')
    metrics = tuple(metric for metric in PRIMARY_METRICS if metric not in exclude)
    if not metrics:
        raise ValueError('At least one metric must remain visible.')
    layouts = resolve_layouts(layouts, len(metrics))
    Path(out).mkdir(parents=True, exist_ok=True)
    views = ('all', 'changed') if view == 'both' else (view,)
    for current_view in views:
        for layout in dict.fromkeys(layouts):
            if layout == '3x3':
                _combined_primary_boxplots_detailed(
                    data, tests, out, metrics, current_view, y_axis, showfliers)
            else:
                _compact_primary_boxplots(
                    data, tests, out, layout, metrics, current_view, y_axis,
                    showfliers)


def read_and_aggregate_entities(path):
    """Build paired file deltas; a CK class contributes only if both values exist."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Entity CSV not found: {path}')
    dtype = {name: str for name in ('project','Key','group','source','entity','path','metric')}
    raw = pd.read_csv(path, dtype=dtype, low_memory=False, float_precision='round_trip')
    required = set(dtype) | {'intro','repayment'}
    if required - set(raw.columns):
        raise ValueError(f'{path.name}: missing columns {sorted(required - set(raw.columns))}. '
                         'Use entity_metrics.csv; issue_metric_deltas.csv cannot recover file-level data.')
    raw = raw.loc[raw.metric.isin(PRIMARY_METRICS)].copy()
    if raw.empty:
        raise ValueError('No primary metric entities found in the CSV.')
    identifiers = ['project','Key','group','source','entity','path','metric']
    if raw[identifiers].isna().any().any():
        raise ValueError('Missing issue, group, source, entity, file path, or metric identity.')
    if not set(raw.group).issubset(GROUPS):
        raise ValueError(f'Unknown SF/NSF group labels: {sorted(set(raw.group) - set(GROUPS))}')
    if (raw.groupby(['project','Key']).group.nunique() > 1).any():
        raise ValueError('The same issue has conflicting SF/NSF labels.')
    raw['path'] = raw.path.map(lambda value: posixpath.normpath(value.strip().replace('\\','/')))
    if raw.path.isin({'','.','__issue__'}).any():
        raise ValueError('File paths are required; issue-level structural rows cannot provide file granularity.')
    if raw.duplicated(['project','Key','source','entity','metric']).any():
        raise ValueError('Duplicate entity/metric rows would double-count an entity.')
    if raw.duplicated(['project','Key','source','path','metric']).loc[raw.source.eq('structural')].any():
        raise ValueError('More than one structural row per issue/file/metric; file values must be unique.')
    wrong = raw.loc[raw.metric.map(SOURCE_BY_METRIC).ne(raw.source),
                    ['metric','source']]
    if not wrong.empty:
        raise ValueError('Metric/source mismatch; expected structural file values and CK class values: '
                         + str(wrong.drop_duplicates().head(5).to_dict('records')))
    for col in ('intro','repayment'):
        raw[col] = pd.to_numeric(raw[col], errors='raise')
        if np.isinf(raw[col].to_numpy(dtype=float)).any():
            raise ValueError(f'Infinite {col} values are invalid.')

    keys = ['project','Key','group','path','metric']
    rows = []
    for (project,key,group,file_path,metric), part in raw.groupby(keys,sort=True):
        paired = part.intro.notna() & part.repayment.notna()
        valid = part.loc[paired]
        intro = float(valid.intro.mean()) if len(valid) else np.nan
        repayment = float(valid.repayment.mean()) if len(valid) else np.nan
        rows.append(dict(project=project,Key=key,group=group,path=file_path,
                         metric=metric,label=METRIC_LABELS[metric],
                         source=SOURCE_BY_METRIC[metric],
                         n_entities_total=len(part),n_matched_entities=len(valid),
                         n_intro_only=int((part.intro.notna() & part.repayment.isna()).sum()),
                         n_repayment_only=int((part.intro.isna() & part.repayment.notna()).sum()),
                         intro=intro,repayment=repayment,delta=repayment-intro))
    return pd.DataFrame(rows)


def load_file_level_deltas(path):
    """Read previously saved issue-file deltas for immediate layout changes."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'File-level boxplot CSV not found: {path}')
    dtype = {name: str for name in ('project','Key','group','path','metric')}
    data = pd.read_csv(path,dtype=dtype,float_precision='round_trip')
    if not {'project','Key','group','path','metric','intro','repayment','delta'}.issubset(data.columns):
        raise ValueError('This CSV needs project, Key, group, path, metric, intro, repayment, delta. '
                         'An issue-level boxplot CSV cannot substitute for file-level data.')
    if data[['project','Key','group','path','metric']].isna().any().any():
        raise ValueError('Missing issue or file identity in saved file-level data.')
    if data.duplicated(['project','Key','path','metric']).any():
        raise ValueError('Duplicate issue/file/metric rows in file-level CSV.')
    if not set(data.group).issubset(GROUPS):
        raise ValueError('Unexpected SF/NSF group label in file-level CSV.')
    if not set(data.metric).issubset(PRIMARY_METRICS):
        raise ValueError('Unexpected metric in file-level CSV.')
    for col in ('intro','repayment','delta'):
        data[col]=pd.to_numeric(data[col],errors='raise')
        if np.isinf(data[col].to_numpy(dtype=float)).any():
            raise ValueError(f'Infinite {col} in saved file-level CSV.')
    complete=data[['intro','repayment']].notna().all(axis=1)
    if not np.allclose(data.loc[complete,'delta'],
                       data.loc[complete,'repayment']-data.loc[complete,'intro'],
                       rtol=1e-12,atol=1e-12):
        raise ValueError('Saved delta does not equal repayment minus introduction.')
    if data.loc[~complete,'delta'].notna().any():
        raise ValueError('A delta cannot be present when either endpoint is missing.')
    return data


def file_level_summary(data):
    """Descriptive only; repeated files from one issue are not independent."""
    rows=[]
    for metric in PRIMARY_METRICS:
        for group in GROUPS:
            part=data.loc[(data.metric==metric)&(data.group==group)].dropna(subset=['delta'])
            values=part.delta.to_numpy(dtype=float)
            q1,median,q3=np.quantile(values,[.25,.5,.75]) if len(values) else (np.nan,)*3
            rows.append(dict(metric=metric,label=METRIC_LABELS[metric],group=group,
                             n_issue_file_observations=len(values),
                             n_distinct_issues=part[['project','Key']].drop_duplicates().shape[0],
                             n_zero_delta=int(np.isclose(values, 0, atol=1e-9, rtol=0).sum()),
                             n_changed_delta=int((~np.isclose(values, 0, atol=1e-9, rtol=0)).sum()),
                             pct_zero_delta=(100 * np.isclose(values, 0, atol=1e-9, rtol=0).mean()
                                             if len(values) else np.nan),
                             median_delta=median,q1_delta=q1,q3_delta=q3))
    return pd.DataFrame(rows)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input-dir',type=Path,
                        default=SCRIPT_DIR/'ATD_DIRECT_COMPARISON_RESULTS',
                        help='Folder with entity_metrics.csv')
    parser.add_argument('--entity-csv',type=Path,
                        help='Override entity_metrics.csv; cannot be combined with --plot-data')
    parser.add_argument('--plot-data',type=Path,
                        help='Replot saved FILE_LEVEL_boxplot_data.csv without reaggregating')
    parser.add_argument('--output',type=Path,
                        help='Default: FILE_LEVEL_BOXPLOTS next to entity_metrics.csv')
    parser.add_argument('--layouts',nargs='+',choices=['3x3','1x9','2x4'],default=None,
                        help='All fitting layouts by default; 1x9 shrinks to 1xN')
    parser.add_argument('--exclude',nargs='+',choices=PRIMARY_METRICS,default=[],
                        help='Hide selected metrics from figures only')
    parser.add_argument('--view',choices=['all','changed','both'],default='both',
                        help='All paired files, nonzero-change files, or both (default)')
    parser.add_argument('--y-axis',choices=['per-metric','shared'],default='per-metric',
                        help='Independent metric panels (default); SF/NSF still share each panel')
    parser.add_argument('--showfliers',action='store_true',
                        help='Draw outlier markers; may compress boxes with extreme values')
    args=parser.parse_args(argv)
    if args.entity_csv and args.plot_data:
        parser.error('Choose --entity-csv or --plot-data, not both.')
    visible=[metric for metric in PRIMARY_METRICS if metric not in args.exclude]
    if not visible:
        parser.error('At least one metric must remain visible.')
    try:
        layouts=resolve_layouts(args.layouts,len(visible))
        input_path=args.plot_data or args.entity_csv or (args.input_dir/'entity_metrics.csv')
        data=(load_file_level_deltas(input_path) if args.plot_data
              else read_and_aggregate_entities(input_path))
    except (ValueError,OSError) as exc:
        parser.error(str(exc))
    output=args.output or (input_path.parent if args.plot_data
                           else input_path.parent/'FILE_LEVEL_BOXPLOTS')
    output.mkdir(parents=True,exist_ok=True)
    input_sha256=file_sha256(input_path)
    plot_csv=output/'FILE_LEVEL_boxplot_data.csv'
    if not args.plot_data or input_path.resolve() != plot_csv.resolve():
        data.sort_values(['project','Key','path','metric']).to_csv(
            plot_csv,index=False,float_format='%.17g')
    summary=file_level_summary(data)
    summary.to_csv(output/'FILE_LEVEL_boxplot_summary.csv',index=False,float_format='%.17g')
    labels=pd.DataFrame({'metric':PRIMARY_METRICS,
                         'label':[METRIC_LABELS[k] for k in PRIMARY_METRICS],
                         'family':['primary']*len(PRIMARY_METRICS)})
    combined_primary_boxplots(data,labels,output,layouts,args.exclude,
                              args.view,args.y_axis,args.showfliers)
    manifest=dict(version=VERSION,granularity='issue-file',
                  structural_aggregation='native file-level metric',
                  ck_aggregation='mean of classes matched at both boundaries in the same file',
                  delta='repayment - introduction',statistical_tests_recomputed=False,
                  displayed_p_values=False,shared_x=True,
                  y_axis_across_metrics=args.y_axis,
                  view=args.view,show_outlier_markers=args.showfliers,
                  zero_tolerance=1e-9,
                  requested_layouts=layouts,
                  rendered_layouts=[f'1x{len(visible)}' if name=='1x9' else name
                                    for name in layouts],
                  visible_metrics=visible,excluded_metrics=sorted(set(args.exclude)),
                  input_file=str(input_path.resolve()),
                  input_sha256=input_sha256,
                  python=platform.python_version(),numpy=np.__version__,
                  pandas=pd.__version__,matplotlib=matplotlib.__version__)
    (output/'file_level_boxplot_manifest.json').write_text(
        json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(summary[['label','group','n_issue_file_observations','n_distinct_issues']].to_string(index=False))
    print(f'File-level figures: {", ".join(manifest["rendered_layouts"])}; '
          f'view: {args.view}; Y-axis: {args.y_axis}; '
          f'outputs: {output.resolve()}')


if __name__=='__main__':
    main()


