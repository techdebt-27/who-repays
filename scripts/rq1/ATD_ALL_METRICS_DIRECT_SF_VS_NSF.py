#!/usr/bin/env python3
"""Unified ATD structural analysis: direct SF versus NSF, one observation/issue.

INSTALL: python -m pip install numpy pandas scipy matplotlib
RUN with both combined CSV files beside this script:
  python ATD_ALL_METRICS_DIRECT_SF_VS_NSF.py

Default input files (read separately, aggregate by issue, then analyze together):
  combined_all_csv__CK_CLASS_LEVEL_PER_BARIS.csv
  combined_all_FIN_FOUT_COMPLEXITY.csv

If both files are in a different folder:
  python ATD_ALL_METRICS_DIRECT_SF_VS_NSF.py --input-dir '/path/to/CSV'
Explicit --ck-input and --structural-input paths still override these defaults.
Default output: ATD_DIRECT_COMPARISON_RESULTS beside this script.

Structural input must contain paired FILE-level values (not methods/classes),
Key, indicator, self_fixed (or Is Self-Fixed (Intro=Payment)), and a file path.
Default columns: intro_file_fan_in/payment_file_fan_in, ...fan_out,
intro_file_cyclomatic_complexity_sum/payment_file_cyclomatic_complexity_sum,
intro_file_cognitive_complexity_sum/payment_file_cognitive_complexity_sum.
Legacy names without _sum remain supported. These are pre-extracted file sums;
issue aggregation still defaults to the mean of those file values.
Use --write-config mapping.json to export ALL column/aggregation settings;
edit column mappings then pass --config mapping.json. File-level structural
rows are required; pre-aggregated issue exports cannot establish entity pairing.
This program analyzes already extracted values; it does not run Understand,
CK, or Sonar. It does not substitute WMC for cyclomatic/cognitive complexity.

DESIGN
* Primary = 9 direct comparisons: 4 structural + CBO, RFC proxy, LCOM, DIT, NOC.
* Delta is SIGNED aggregate(repayment) - aggregate(introduction), not |delta|.
* MWU two-sided, asymptotic with tie/continuity correction; positive Cliff's
  delta means SF tends to have larger signed changes, not better quality.
* Holm across the full predeclared primary family; extra CK variants have a
  separate exploratory family. Missing tests retain their family slots.
* 95% percentile bootstrap CIs resample issues independently within SF/NSF;
  CIs are marginal, not simultaneous and not adjusted for multiple testing.
* Secondary paired Wilcoxon has its own correction family. It cannot establish
  SF-NSF differences. Its signed-rank interpretation assumes symmetric changes.
* Primary uses mean for ALL nine metrics (including DIT), restricted separately
  for each metric to entities with valid values at BOTH boundaries. Delta of
  means equals mean of paired entity deltas. Each issue has equal test weight.
* Matched analysis describes traceable surviving entities, not entity additions
  or removals. Missing observations are never replaced with zeros. Counts of
  unmatched entities are saved, and phase-wise sensitivity includes them where
  observed. A missing metric does not itself establish entity deletion.
* Three automatic sensitivity scenarios: matched sums of four structural
  metrics; matched maximum DIT; phase-wise means for all nine primary metrics.
  Holm correction pools all 14 sensitivity hypotheses into ONE separate family.
  Primary Holm uses nine hypotheses; exploratory CK tests and secondary Wilcoxon
  have their own families. CIs remain marginal. Sensitivity results are supporting
  analyses, not additional opportunities to select a significant primary result.
* All rows are collected BEFORE aggregation (memory use scales with input).
  Exact duplicate entities are collapsed; conflicting duplicates/labels fail.
* Pooled unadjusted tests assume independent issues. Shared commits/files and
  project clustering can violate this assumption. Contextual/confounder models
  and isolated repayment-commit effects are outside revision suggestion #2.
* Non-significance does not establish equivalence or no meaningful difference.
* CountDeclMethodAll is retained as an RFC PROXY, not asserted to be canonical
  RFC. Verify extraction semantics in the manuscript. All main aggregations
  use mean. File complexity is already a sum within each file; its issue-level
  mean is average file complexity, NOT average method complexity. Summed
  fan-in/out values are aggregate counts, not unique architectural edges.
* Means do not adjust for file size, baseline, lifetime, or project context.
  All modes aggregate AFTER deduplication, independently for CK classes and
  structural files; class rows never multiply structural file observations.

Outputs: PRIMARY_direct_tests.csv, PRIMARY_manuscript_table.csv,
EXPLORATORY_direct_tests.csv, SECONDARY_within_group_wilcoxon.csv,
issue_metric_deltas.csv, issue_level_wide.csv, entity_metrics.csv,
coverage_by_project.csv, input_audit.csv, run_manifest.json, analysis_notes.txt,
and PDF/PNG plots of delta, ECDF, and decrease/zero/increase proportions.
PRIMARY_ALL_9_METRICS_BOXPLOTS_SF_vs_NSF.png/.pdf combines the nine primary
boxplots in one 3-by-3 figure, with independent y-scales and all outliers shown.
Additional files with suffixes _1x9 and _2x5 use compact publication layouts
(rows x columns): 1x9 = 14.32 x 2.6 inches; 2x5 = 7.16 x 4.4 inches. The tenth
panel in 2x5 is hidden. Both use 9-point text, short labels, embedded PDF fonts,
and no p-values/sample counts in the panels (see PRIMARY_manuscript_table.csv).
All layouts are generated by default. Select, for example:
  python ATD_ALL_METRICS_DIRECT_SF_VS_NSF.py --combined-layouts 1x9 2x5
All combined layouts share the categorical x-axis (SF, NSF). Tick labels appear
only on the lowest populated panel per column; y-scales remain independent.
In the detailed 3x3 layout sample counts are inside each panel, not shared ticks.
Caption for compact layouts: Issue-level metric changes from ATD introduction
to repayment for SF and NSF items. Delta = repayment minus introduction.
Boxes: Q1-Q3; line: median; whiskers: most extreme observations within 1.5 IQR
of the box; points: outliers. Panels use independent vertical scales; dashed
lines mark zero change. RFC proxy denotes CountDeclMethodAll.
Plots are saved to disk (no GUI window); --no-plots disables all figures.
Reusable boxplot inputs are ALWAYS exported, including with --no-plots:
  PRIMARY_boxplot_data.csv: one row per issue/metric, including missing deltas;
    included_in_plot identifies finite deltas used by every combined layout.
  PRIMARY_boxplot_metadata.csv: nine metric labels and the existing primary
    test results, including Holm p-values and sample counts. Contains a SHA256
    checksum of the data CSV, schema version, and delta definition.
All layouts use the SAME issue values, not a separate dataset per layout.
Run PLOT_ATD_BOXPLOTS_FROM_CSV.py independently to recreate any of the layouts
without reading extraction inputs, running tests, or importing this script.
SENSITIVITY_direct_tests.csv pools the 14 sensitivity tests; corresponding
scenario folders contain their datasets, tables and figures. Their p_holm
values all use the pooled 14-test family. Primary and sensitivity cohorts may
vary when only one boundary has usable values; per-metric sample sizes are saved.

SciPy reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html
"""
from __future__ import annotations
import argparse
import copy
import glob
import hashlib
import json
import platform
import posixpath
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.stats import mannwhitneyu, rankdata, wilcoxon
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

VERSION = '2.3.1'

# EDIT THIS DIRECTORY if you prefer running without command-line arguments.
SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = SCRIPT_DIR
CK_INPUT_FILENAME = 'combined_all_csv__CK_CLASS_LEVEL_PER_BARIS.csv'
STRUCTURAL_INPUT_FILENAME = 'combined_all_FIN_FOUT_COMPLEXITY.csv'
OUTPUT_DIR = SCRIPT_DIR / 'ATD_DIRECT_COMPARISON_RESULTS'
GROUPS = ('Self-Fixed', 'Non-Self-Fixed')
COLORS = ('#4477AA', '#EE7733')
CK_ALIASES = {
 'cbo': ['cbo_CountClassCoupled','CountClassCoupled','cbo','CBO'],
 'cbo_modified': ['cbo_modified_CountClassCoupledModified','CountClassCoupledModified','modified_cbo','CBO_Modified'],
 'rfc_understand': ['rfc_understand_CountDeclMethodAll','CountDeclMethodAll','rfc','RFC'],
 'rfc_custom_approx': ['rfc_custom_approx','RFC_Custom_Approx','rfc_custom'],
 'lcom': ['lcom_PercentLackOfCohesion','PercentLackOfCohesion','lcom','LCOM'],
 'lcom_modified': ['lcom_modified_PercentLackOfCohesionModified','PercentLackOfCohesionModified','modified_lcom','LCOM_Modified'],
 'dit': ['dit_MaxInheritanceTree','MaxInheritanceTree','dit','DIT'],
 'noc': ['noc_CountClassDerived','CountClassDerived','noc','NOC'],
 'count_class_base': ['count_class_base_CountClassBase','CountClassBase','base_classes','IFANIN'],
 'method_count': ['method_count_CountDeclMethod','CountDeclMethod','method_count'],
 'sloc': ['sloc_CountLineCode','CountLineCode','sloc'],
 'executable_loc': ['executable_loc_CountLineCodeExe','CountLineCodeExe','executable_loc'],
}

def default_config():
    config = {'columns': {
      'key': ['Key','key','issue_key'], 'project': ['project','Project','project_name'],
      'indicator': ['indicator','indicator_normalized'],
      'group': ['self_fixed','Is Self-Fixed (Intro=Payment)','self_fixed_group','self_fixed_normalized'],
      'path': ['canonical_file_path','intro_file_path','file_path','file'],
      'class': ['class_name','intro_class_name','payment_class_name'],
    }, 'metrics': {}}
    for key, label in [('fan_in','Fan-In'),('fan_out','Fan-Out'),
                       ('cyclomatic_complexity','Cyclomatic complexity'),
                       ('cognitive_complexity','Cognitive complexity')]:
        config['metrics'][key] = dict(label=label, source='structural', family='primary', agg='mean',
            intro=[f'intro_file_{key}',f'intro_{key}'],
            repayment=[f'payment_file_{key}',f'repayment_file_{key}',f'payment_{key}',f'repayment_{key}'])
    # Exact column names verified against sample-kafka.csv.
    for key in ['cyclomatic_complexity','cognitive_complexity']:
        config['metrics'][key]['intro'].insert(0,f'intro_file_{key}_sum')
        config['metrics'][key]['repayment'][0:0] = [f'payment_file_{key}_sum',f'repayment_file_{key}_sum']
    labels = {'cbo':'CBO', 'rfc_understand':'RFC proxy (CountDeclMethodAll)',
              'rfc_custom_approx':'RFC custom approximation', 'lcom':'LCOM',
              'dit':'DIT','noc':'NOC'}
    primary = {'cbo','rfc_understand','lcom','dit','noc'}
    for key in CK_ALIASES:
        config['metrics'][key] = dict(label=labels.get(key,key.replace('_',' ').title()),
            source='ck', family='primary' if key in primary else 'exploratory',
            agg='mean', intro=[f'intro_{key}'],
            repayment=[f'payment_{key}',f'repayment_{key}'])
    return config


def aliases(value):
    return [value] if isinstance(value,str) else value


def first(row, names):
    for name in aliases(names):
        value = row.get(name)
        if value is not None and not (isinstance(value,float) and np.isnan(value)) and str(value).strip():
            return value
    return None


def path_norm(value):
    # Java/Git paths are case sensitive: do not lowercase identities.
    return posixpath.normpath(str(value).strip().replace('\\','/')) if value is not None else ''


def group_name(value):
    s = str(value).strip().lower().replace('_','-')
    if s in {'true','1','1.0','yes','y','self-fixed','sf'}: return GROUPS[0]
    if s in {'false','0','0.0','no','n','non-self-fixed','nsf'}: return GROUPS[1]
    raise ValueError(f'Unknown/missing self-fixed label: {value!r}')


def number(value):
    if value is None or str(value).strip().lower() in {'','nan','none','null','<na>'}: return np.nan
    try: n = float(value)
    except (ValueError,TypeError):
        raise ValueError(f'Non-numeric metric value: {value!r}; fix the input rather than silently dropping it.')
    return n if np.isfinite(n) else np.nan


def parse_json(value, expected):
    if value is None or str(value).strip().lower() in {'','nan','none','null','<na>'}: return expected()
    parsed = json.loads(value) if isinstance(value,str) else value
    if expected is list and isinstance(parsed,dict): parsed = [parsed]
    if not isinstance(parsed,expected): raise ValueError(f'Expected JSON {expected.__name__}')
    return parsed


def input_paths(patterns):
    found = set()
    for pattern in patterns:
        p = Path(pattern)
        matches = sorted(p.glob('*.csv')) if p.is_dir() else [Path(x) for x in glob.glob(pattern)]
        if not matches: raise FileNotFoundError(f'No CSV files match {pattern}')
        found.update(p.resolve() for p in matches)
    return sorted(found)


def read_entities(paths, source, config, args, audit):
    """Normalize inputs to one entity/metric row, retaining boundary missingness."""
    records = []
    specs = {k:v for k,v in config['metrics'].items() if v['source']==source}
    cols = config['columns']
    for path in paths:
        header = pd.read_csv(path,nrows=0).columns
        raw_ck = source=='ck' and 'Intro_CK_ClassDetails_JSON' in header
        if raw_ck and 'Payment_CK_ClassDetails_JSON' not in header:
            raise ValueError(f'{path}: missing Payment_CK_ClassDetails_JSON')
        found_specs = specs if raw_ck else {k:s for k,s in specs.items()
            if any(c in header for c in aliases(s['intro'])) and any(c in header for c in aliases(s['repayment']))}
        if not found_specs: raise ValueError(f'{path}: no paired {source} metric columns. Use --write-config then --config.')
        scanned = kept = skipped_indicator = skipped_nonjava = 0
        for chunk in pd.read_csv(path,chunksize=args.chunksize,low_memory=False):
            for row in chunk.to_dict('records'):
                scanned += 1
                key = first(row,cols['key'])
                if key is None: raise ValueError(f'{path}: missing issue Key')
                key = str(key).strip()
                indicator = first(row,cols['indicator'])
                if args.indicator != 'all':
                    if indicator is None: raise ValueError(f'{path}: indicator required for filtering')
                    if str(indicator).strip().lower()!=args.indicator.lower():
                        skipped_indicator += 1
                        continue
                group = group_name(first(row,cols['group']))
                project = first(row,cols['project'])
                if project is None:
                    if '-' not in key: raise ValueError(f'{key}: provide project column or Jira-style KEY-123')
                    project = key.rsplit('-',1)[0]
                meta = dict(project=str(project).strip().upper(),Key=key,group=group,source=source)
                kept += 1
                if raw_ck:
                    mapping = {path_norm(k):path_norm(v) if v else None for k,v in
                               parse_json(row.get('Payment_Path_Map_JSON'),dict).items()}
                    phases = {}
                    for phase,col in [('intro','Intro_CK_ClassDetails_JSON'),('repayment','Payment_CK_ClassDetails_JSON')]:
                        values = {}
                        for item in parse_json(row.get(col),list):
                            if not isinstance(item,dict): raise ValueError(f'{key}: malformed class JSON entry')
                            fp = first(item,['file_path','understand_file_relname','path','filename','file'])
                            cn = first(item,['class_name','class_longname','longname','name','class'])
                            if fp is None or cn is None: raise ValueError(f'{key}: class JSON needs file path and class name')
                            fp = path_norm(fp)
                            if not args.include_non_java and not fp.lower().endswith('.java'):
                                skipped_nonjava += 1
                                continue
                            canonical = (mapping.get(fp,fp) or fp) if phase=='intro' else fp
                            entity = canonical+'::'+str(cn).strip()
                            metrics = {k:number(first(item,CK_ALIASES[k])) for k in found_specs}
                            if entity in values: raise ValueError(f'{key}: duplicate class identity within JSON: {entity}')
                            values[entity] = (canonical,metrics)
                        phases[phase] = values
                    for entity in sorted(set(phases['intro']) | set(phases['repayment'])):
                        left = phases['intro'].get(entity)
                        right = phases['repayment'].get(entity)
                        for metric in found_specs:
                            records.append(dict(meta,entity=entity,path=(left or right)[0],metric=metric,
                                intro=left[1][metric] if left else np.nan,
                                repayment=right[1][metric] if right else np.nan))
                else:
                    issue_level = source=='structural' and args.structural_level=='issue'
                    fp = first(row,cols['path'])
                    if fp is None and not issue_level: raise ValueError(f'{path}: entity path missing; configure columns.path')
                    fp = path_norm(fp) if fp is not None else ''
                    if not issue_level and not args.include_non_java and not fp.lower().endswith('.java'):
                        skipped_nonjava += 1
                        continue
                    if source=='ck':
                        entity = first(row,['class_key'])
                        if entity is None:
                            cn = first(row,cols['class'])
                            if cn is None: raise ValueError(f'{key}: CK row needs class_key or class name')
                            entity = fp+'::'+str(cn).strip()
                    else: entity = '__issue__' if issue_level else fp
                    for metric,spec in found_specs.items():
                        records.append(dict(meta,entity=str(entity),path=fp,metric=metric,
                            intro=number(first(row,spec['intro'])),repayment=number(first(row,spec['repayment']))))
        audit.append(dict(file=str(path),source=source,input_rows=scanned,rows_after_indicator=kept,
                          filtered_indicator=skipped_indicator,filtered_nonjava=skipped_nonjava))
    return records


def clean_entities(records):
    if not records: raise ValueError('No eligible metric records remain.')
    frame = pd.DataFrame(records)
    identity = ['project','Key']
    conflict = frame.groupby(identity)['group'].nunique()
    if (conflict>1).any(): raise ValueError(f'Conflicting SF/NSF labels: {conflict[conflict>1].index.tolist()[:10]}')
    eid = identity+['source','entity','metric']
    before = len(frame)
    unique = frame.drop_duplicates(subset=eid+['group','intro','repayment'])
    bad = unique.duplicated(eid,keep=False)
    if bad.any():
        raise ValueError('Conflicting duplicate entity values. Do not mix lifecycle snapshots or raw/derived exports.\n'
                         +unique.loc[bad,eid+['intro','repayment']].head(10).to_string(index=False))
    return unique.reset_index(drop=True),before-len(unique)


def aggregate_entities(frame, config, args):
    result = []
    for (project,key,group,metric), part in frame.groupby(['project','Key','group','metric'],sort=True):
        spec = config['metrics'][metric]
        a,b = part.intro,part.repayment
        matched = a.notna() & b.notna()
        if args.pairing=='matched': a,b = a[matched],b[matched]
        agg = spec['agg']
        if spec['source']=='structural' and args.structural_level=='issue': agg='mean'
        def reduce(v):
            v=v.dropna()
            return float(getattr(v,agg)()) if len(v) else np.nan
        intro,repayment=reduce(a),reduce(b)
        result.append(dict(project=project,Key=key,group=group,metric=metric,label=spec['label'],
            family=spec['family'],aggregation=agg,pairing=args.pairing,
            n_entities=len(part),n_intro_available=int(part.intro.notna().sum()),
            n_repayment_available=int(part.repayment.notna().sum()),n_matched=int(matched.sum()),
            n_intro_only=int((part.intro.notna() & part.repayment.isna()).sum()),
            n_repayment_only=int((part.intro.isna() & part.repayment.notna()).sum()),
            n_neither=int((part.intro.isna() & part.repayment.isna()).sum()),
            n_intro_used=int(a.notna().sum()),n_repayment_used=int(b.notna().sum()),
            intro=intro,repayment=repayment,delta=repayment-intro))
    return pd.DataFrame(result)


def cliffs_delta(x,y):
    sy=np.sort(y)
    greater=np.searchsorted(sy,x,side='left').sum()
    less=(len(sy)-np.searchsorted(sy,x,side='right')).sum()
    return float((greater-less)/(len(x)*len(y)))


def holm(pvalues):
    """Preserve planned family size: untestable p's count as 1 internally."""
    p=np.asarray(pvalues,dtype=float)
    valid=np.isfinite(p)
    work=np.where(valid,p,1.)
    order=np.argsort(work,kind='stable')
    adjusted=np.minimum(1.,np.maximum.accumulate(work[order]*(len(work)-np.arange(len(work)))))
    out=np.empty_like(work); out[order]=adjusted; out[~valid]=np.nan
    return out


def bootstrap_ci(x,y,n_boot,seed):
    if len(x)<2 or len(y)<2: return [np.nan]*4
    rng=np.random.default_rng(seed)
    effects=np.empty(n_boot); medians=np.empty(n_boot)
    for i in range(n_boot):
        a=rng.choice(x,len(x),replace=True); b=rng.choice(y,len(y),replace=True)
        effects[i]=cliffs_delta(a,b); medians[i]=np.median(a)-np.median(b)
    return [*np.quantile(effects,[.025,.975]),*np.quantile(medians,[.025,.975])]


def descriptive(values):
    if not len(values): return dict(n=0,median=np.nan,q1=np.nan,q3=np.nan,iqr=np.nan,
                                   pct_decrease=np.nan,pct_zero=np.nan,pct_increase=np.nan)
    q1,median,q3=np.quantile(values,[.25,.5,.75])
    return dict(n=len(values),median=median,q1=q1,q3=q3,iqr=q3-q1,
                pct_decrease=100*np.mean(values<0),pct_zero=100*np.mean(values==0),pct_increase=100*np.mean(values>0))


def direct_tests(data,config,args):
    rows=[]
    for metric,spec in config['metrics'].items():
        part=data[data.metric==metric]
        x=part.loc[part.group==GROUPS[0],'delta'].dropna().to_numpy()
        y=part.loc[part.group==GROUPS[1],'delta'].dropna().to_numpy()
        row=dict(metric=metric,label=spec['label'],family=spec['family'],aggregation=spec['agg'],
                 U=np.nan,p_value=np.nan,cliffs_delta=np.nan,cliffs_ci_low=np.nan,cliffs_ci_high=np.nan,
                 median_difference=np.nan,median_diff_ci_low=np.nan,median_diff_ci_high=np.nan,
                 status='insufficient_data',direction='not_estimated')
        row.update({'SF_'+k:v for k,v in descriptive(x).items()})
        row.update({'NSF_'+k:v for k,v in descriptive(y).items()})
        if len(x) and len(y):
            effect=cliffs_delta(x,y)
            row.update(cliffs_delta=effect,median_difference=float(np.median(x)-np.median(y)),
                       direction='SF larger signed delta' if effect>0 else 'NSF larger signed delta' if effect<0 else 'no rank tendency')
        if len(x)>=2 and len(y)>=2:
            if np.all(np.concatenate([x,y])==x[0]): u,p=len(x)*len(y)/2,1.
            else: u,p=mannwhitneyu(x,y,alternative='two-sided',method='asymptotic',use_continuity=True)
            seed=args.seed+int(hashlib.sha256(metric.encode()).hexdigest()[:8],16)
            ci=bootstrap_ci(x,y,args.bootstrap,seed)
            row.update(U=u,p_value=p,cliffs_ci_low=ci[0],cliffs_ci_high=ci[1],
                       median_diff_ci_low=ci[2],median_diff_ci_high=ci[3],status='ok')
        rows.append(row)
    result=pd.DataFrame(rows)
    for _,indices in result.groupby('family').groups.items():
        result.loc[indices,'p_holm']=holm(result.loc[indices,'p_value'])
        result.loc[indices,'family_size']=len(indices)
    result['significant_holm']=result.p_holm.map(lambda p:'not_tested' if pd.isna(p) else str(p<args.alpha))
    return result


def secondary_tests(data,config,args):
    rows=[]
    for metric,spec in config['metrics'].items():
        for group in GROUPS:
            part=data[(data.metric==metric)&(data.group==group)].dropna(subset=['intro','repayment'])
            d=part.delta.to_numpy(); stat=p=rbc=np.nan
            if len(d)>=2:
                nz=d[d!=0]
                if not len(nz): stat,p,rbc=0.,1.,0.
                else:
                    stat,p=wilcoxon(d,zero_method='wilcox',alternative='two-sided',method='approx')
                    ranks=rankdata(abs(nz)); rbc=float((ranks*np.sign(nz)).sum()/ranks.sum())
            rows.append(dict(metric=metric,label=spec['label'],family=spec['family'],group=group,
                             n=len(d),n_nonzero=int(np.count_nonzero(d)),W=stat,p_value=p,
                             paired_rank_biserial=rbc))
    result=pd.DataFrame(rows)
    # All within-group tests belong to one separate secondary family.
    result['p_holm']=holm(result.p_value)
    return result


def manuscript_table(tests):
    def fmt(x): return 'NA' if pd.isna(x) else f'{x:.4g}'
    def ci(value,lo,hi): return f'{fmt(value)} [{fmt(lo)}, {fmt(hi)}]'
    records=[]
    for r in tests.to_dict('records'):
        records.append({'Metric':r['label'],'N_SF':r['SF_n'],'N_NSF':r['NSF_n'],
          'SF_delta_median_[Q1,Q3]':ci(r['SF_median'],r['SF_q1'],r['SF_q3']),
          'NSF_delta_median_[Q1,Q3]':ci(r['NSF_median'],r['NSF_q1'],r['NSF_q3']),
          'Median_difference_SF_minus_NSF_[95%CI]':ci(r['median_difference'],r['median_diff_ci_low'],r['median_diff_ci_high']),
          'U':r['U'],'p_Holm':r['p_holm'],
          'Cliffs_delta_[95%CI]':ci(r['cliffs_delta'],r['cliffs_ci_low'],r['cliffs_ci_high']),
          'Status':r['status']})
    return pd.DataFrame(records)


def figures(data,tests,out):
    for r in tests.to_dict('records'):
        metric=r['metric']; part=data[data.metric==metric]
        arrays=[part.loc[part.group==g,'delta'].dropna().to_numpy() for g in GROUPS]
        if not all(len(a) for a in arrays): continue
        fig,axes=plt.subplots(1,3,figsize=(12,3.6),layout='constrained')
        bp=axes[0].boxplot(arrays,patch_artist=True,showfliers=True)
        for box,color in zip(bp['boxes'],COLORS): box.set_facecolor(color); box.set_alpha(.65)
        axes[0].set_xticks([1,2],[f'SF (n={len(arrays[0])})',f'NSF (n={len(arrays[1])})'])
        axes[0].axhline(0,color='gray',lw=.8,ls='--')
        axes[0].set_ylabel('Delta = Repayment − Introduction')
        axes[0].set_title('Issue-level signed change')
        for a,g,c in zip(arrays,GROUPS,COLORS):
            vals,counts=np.unique(a,return_counts=True)
            axes[1].step(np.r_[vals[0],vals],np.r_[0,np.cumsum(counts)/len(a)],where='post',color=c,label=g)
        axes[1].set_xlabel('Delta');axes[1].set_ylabel('Cumulative proportion');axes[1].legend(fontsize=8)
        axes[1].set_title('ECDF (ties retained)')
        bottom=np.zeros(2)
        for label,color,predicate in [('Decrease','#4477AA',lambda a:a<0),('Zero','#BBBBBB',lambda a:a==0),('Increase','#EE7733',lambda a:a>0)]:
            vals=np.array([100*np.mean(predicate(a)) for a in arrays])
            axes[2].bar(['SF','NSF'],vals,bottom=bottom,color=color,label=label)
            bottom+=vals
        axes[2].set_ylim(0,100);axes[2].set_ylabel('Issues (%)');axes[2].legend(fontsize=8)
        axes[2].set_title('Direction of change')
        p=r['p_holm']; ptext='NA' if pd.isna(p) else f'{p:.3g}'
        fig.suptitle(f"{r['label']} | {r['family']} | Holm p={ptext}",fontsize=11)
        stem=f"{r['family'].upper()}_{metric}_SF_vs_NSF"
        for ext in ['png','pdf']: fig.savefig(out/f'{stem}.{ext}',dpi=220)
        plt.close(fig)


STRUCTURAL_METRICS = ('fan_in','fan_out','cyclomatic_complexity','cognitive_complexity')
PRIMARY_METRICS = (*STRUCTURAL_METRICS,'cbo','rfc_understand','lcom','dit','noc')


def export_boxplot_inputs(data, tests, out):
    """Save lossless numeric plotting inputs independently of --no-plots.

    Raw issue deltas are essential: rounded medians/quartiles cannot recreate
    whiskers and outliers. Metadata preserves the precomputed inference.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    order = {metric: i + 1 for i, metric in enumerate(PRIMARY_METRICS)}
    values = data.loc[data.metric.isin(PRIMARY_METRICS)].copy()
    values['metric_order'] = values.metric.map(order)
    values['included_in_plot'] = np.isfinite(values['delta'].to_numpy(dtype=float))
    values = values.sort_values(['metric_order', 'group', 'project', 'Key'])
    data_path = out / 'PRIMARY_boxplot_data.csv'
    values.to_csv(data_path, index=False, float_format='%.17g')
    metadata = tests.loc[tests.family == 'primary'].copy()
    if metadata.metric.duplicated().any() or set(metadata.metric) != set(PRIMARY_METRICS):
        raise ValueError('Boxplot metadata requires exactly the nine primary metrics.')
    metadata['metric_order'] = metadata.metric.map(order)
    metadata['boxplot_schema_version'] = '1.0'
    metadata['analysis_version'] = VERSION
    metadata['delta_definition'] = 'repayment - introduction'
    metadata['data_file'] = data_path.name
    metadata['data_sha256'] = hashlib.sha256(data_path.read_bytes()).hexdigest()
    metadata_path = out / 'PRIMARY_boxplot_metadata.csv'
    metadata.sort_values('metric_order').to_csv(metadata_path, index=False,
                                               float_format='%.17g')
    print(f'Boxplot CSV inputs: {data_path.name}, {metadata_path.name}', flush=True)
    return data_path, metadata_path


def _combined_primary_boxplots_detailed(data, tests, out):
    """Nine primary signed-delta boxplots; use existing Holm results unchanged.

    Each panel has its own linear y-axis. All finite observations, including
    outliers, are displayed. Sample counts are annotated within each panel.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    primary = tests.loc[tests.family == 'primary'].set_index('metric')
    fig, axes = plt.subplots(3, 3, figsize=(14, 12), sharex=True, sharey=False,
                             layout='constrained')
    for index, (ax, metric) in enumerate(zip(axes.flat, PRIMARY_METRICS)):
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
        ax.set_title(f'({chr(97 + index)}) {label}\nHolm p = {ptext}', fontsize=11)
        ax.set_xticks([1, 2], ['SF', 'NSF'])
        ax.text(.5, .96, f'SF n={len(arrays[0])}; NSF n={len(arrays[1])}',
                transform=ax.transAxes, ha='center', va='top', fontsize=9)
        ax.set_xlim(.4, 2.6)
        ax.set_ylabel('Delta = Repayment − Introduction', fontsize=9)
        ax.axhline(0, color='gray', linewidth=.8, linestyle='--', zorder=0)
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('SF versus NSF: changes in all nine primary metrics', fontsize=16)
    fig.supxlabel('One observation per issue; independent y-axis scales. '
                  'Boxes: Q1–Q3; line: median; whiskers: 1.5 × IQR; points: outliers.\n'
                  'Holm correction across nine primary tests. Dashed line: zero change.',
                  fontsize=10)
    try:
        for ext in ('png', 'pdf'):
            fig.savefig(out / f'PRIMARY_ALL_9_METRICS_BOXPLOTS_SF_vs_NSF.{ext}',
                        dpi=300, bbox_inches='tight')
    finally:
        plt.close(fig)


def _compact_primary_boxplots(data, tests, out, layout):
    """Publication alternatives; layout names denote rows x columns.

    Draw directly at final size, preserving exact PDF page dimensions.
    Each metric uses all finite issue deltas and an independent linear scale.
    Font selection uses installed fonts; no font download is required.
    """
    from matplotlib import font_manager, ticker
    layouts = {'1x9': (1, 9, (14.32, 2.6)),
               '2x5': (2, 5, (7.16, 4.4))}
    nrows, ncols, size = layouts[layout]
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
                                 sharex=True, sharey=False, layout='constrained')
        fig.get_layout_engine().set(h_pad=.025, w_pad=.04,
                                    hspace=.035, wspace=.055)
        try:
            for index, (ax, metric) in enumerate(zip(axes.flat, PRIMARY_METRICS)):
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
                ax.set_title(f'({chr(97 + index)}) {label}', loc='left', pad=2)
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
            for ax in axes.flat[len(PRIMARY_METRICS):]:
                ax.set_axis_off()
            # Show X labels on the lowest populated panel of each column.
            # For 2x5, column 5 is labeled on row 1 because row 2 is empty.
            for column in range(ncols):
                last = max(i for i in range(len(PRIMARY_METRICS)) if i % ncols == column)
                for i in range(column, len(PRIMARY_METRICS), ncols):
                    axes.flat[i].tick_params(axis='x', labelbottom=(i == last))
            fig.supylabel('Change (Δ)', fontsize=9)
            fig.canvas.draw()
            for ax in axes.flat[:len(PRIMARY_METRICS)]:
                multiplier = ax.yaxis.get_major_formatter().get_offset()
                if multiplier:
                    ax.text(.98, .96, multiplier, transform=ax.transAxes,
                            ha='right', va='top', fontsize=9)
            # No tight bounding-box crop: retain the intended print dimensions.
            stem = f'PRIMARY_ALL_9_METRICS_BOXPLOTS_SF_vs_NSF_{layout}'
            for ext in ('pdf', 'png'):
                fig.savefig(Path(out) / f'{stem}.{ext}', dpi=300,
                            facecolor='white', bbox_inches=None)
        finally:
            plt.close(fig)


def combined_primary_boxplots(data, tests, out, layouts=('3x3', '1x9', '2x5')):
    """Export requested layouts; statistical calculations are not repeated."""
    invalid = set(layouts) - {'3x3', '1x9', '2x5'}
    if invalid:
        raise ValueError(f'Unknown combined boxplot layouts: {sorted(invalid)}')
    Path(out).mkdir(parents=True, exist_ok=True)
    for layout in dict.fromkeys(layouts):
        if layout == '3x3':
            _combined_primary_boxplots_detailed(data, tests, out)
        else:
            _compact_primary_boxplots(data, tests, out, layout)


def run_sensitivities(entities, config, args):
    """Vary aggregation or pairing one at a time; pool 14 supporting tests."""
    scenarios = [
        ('matched_sum_structural', STRUCTURAL_METRICS, 'sum', 'matched'),
        ('matched_max_dit', ('dit',), 'max', 'matched'),
        ('phase_wise_mean', PRIMARY_METRICS, 'mean', 'phase-wise'),
    ]
    bundles=[]
    for name, metrics, aggregation, pairing in scenarios:
        print(f'Running sensitivity: {name}',flush=True)
        cfg=copy.deepcopy(config)
        cfg['metrics']={key:cfg['metrics'][key] for key in metrics}
        for spec in cfg['metrics'].values():
            spec.update(agg=aggregation,family='primary')
        options=copy.copy(args)
        options.pairing=pairing
        subset=entities[entities.metric.isin(metrics)]
        if subset.empty:
            # Preserve schema so unavailable scenarios still retain Holm slots.
            data=pd.DataFrame(columns=['project','Key','group','metric','intro','repayment','delta'])
        else:
            data=aggregate_entities(subset,cfg,options)
        tests=direct_tests(data,cfg,options)
        tests['family']='sensitivity'
        tests['scenario']=name
        tests['pairing']=pairing
        bundles.append((name,data,tests))
    all_tests=pd.concat([tests for _,_,tests in bundles],ignore_index=True)
    all_tests['p_holm']=holm(all_tests.p_value)
    all_tests['family_size']=len(all_tests)
    all_tests['correction_family']='all_14_sensitivity_tests'
    all_tests['significant_holm']=all_tests.p_holm.map(
        lambda p:'not_tested' if pd.isna(p) else str(p<args.alpha))
    all_tests.to_csv(args.output/'SENSITIVITY_direct_tests.csv',index=False)
    manifest=[]
    for name,data,_ in bundles:
        folder=args.output/'sensitivity'/name
        folder.mkdir(parents=True,exist_ok=True)
        tests=all_tests[all_tests.scenario==name].copy()
        data.to_csv(folder/'issue_metric_deltas.csv',index=False)
        tests.to_csv(folder/'SENSITIVITY_direct_tests.csv',index=False)
        manuscript_table(tests).to_csv(folder/'SENSITIVITY_manuscript_table.csv',index=False)
        if not args.no_plots: figures(data,tests,folder)
        manifest.append(dict(scenario=name,metrics=tests.metric.tolist(),
            aggregation=tests.aggregation.iloc[0],pairing=tests.pairing.iloc[0],
            correction_family='all_14_sensitivity_tests',family_size=len(all_tests)))
    return manifest


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--input-dir',type=Path,default=INPUT_DIR,help='Folder containing the two combined CSVs; default: script folder')
    p.add_argument('--ck-input',nargs='+',default=None,help='Override CK input; default: combined_all_csv__CK_CLASS_LEVEL_PER_BARIS.csv under --input-dir')
    p.add_argument('--structural-input',nargs='+',default=None,help='Override structural input; default: combined_all_FIN_FOUT_COMPLEXITY.csv under --input-dir')
    p.add_argument('--output',type=Path,default=OUTPUT_DIR)
    p.add_argument('--config',type=Path)
    p.add_argument('--write-config',type=Path,help='Write editable mapping and exit')
    p.add_argument('--structural-level',choices=['file'],default='file',help='File-level structural rows required for matched analysis')
    p.add_argument('--pairing',choices=['matched'],default='matched',help='Primary uses matched entities; phase-wise sensitivity runs automatically')
    p.add_argument('--indicator',default='viomod',help='Case-insensitive exact match; all disables filter')
    p.add_argument('--include-non-java',action='store_true')
    p.add_argument('--allow-missing-primary',action='store_true',help='Explicitly permit unavailable primary metrics; retain Holm slots')
    p.add_argument('--bootstrap',type=int,default=5000)
    p.add_argument('--seed',type=int,default=20261002)
    p.add_argument('--chunksize',type=int,default=10000)
    p.add_argument('--alpha',type=float,default=.05)
    p.add_argument('--no-plots',action='store_true')
    p.add_argument('--combined-layouts', nargs='+', choices=['3x3','1x9','2x5'],
                   default=['3x3','1x9','2x5'],
                   help='Combined primary boxplots, rows x columns; default: all three layouts')
    args=p.parse_args(argv)
    config=default_config()
    if args.write_config:
        args.write_config.parent.mkdir(parents=True,exist_ok=True)
        args.write_config.write_text(json.dumps(config,indent=2)+'\n');print(args.write_config);return
    if args.config: config=json.loads(args.config.read_text())
    for key,spec in config['metrics'].items():
        if spec['source'] not in {'ck','structural'} or spec['family'] not in {'primary','exploratory'}:
            p.error(f'Invalid source/family for {key}')
        if spec['source']=='ck' and key not in CK_ALIASES: p.error(f'Unknown CK metric {key}')
        if spec['agg'] not in {'mean','median','max','sum','min'}: p.error(f'Invalid aggregation for {key}')
    if not all(k in config['metrics'] and config['metrics'][k]['family']=='primary' for k in PRIMARY_METRICS) or sum(v['family']=='primary' for v in config['metrics'].values())!=9:
        p.error('The primary family must contain exactly the nine documented metrics.')
    if any(spec['agg']!='mean' for spec in config['metrics'].values()):
        p.error('Version 2 requires mean aggregation in the main configuration. Regenerate old configs with --write-config; sum/max sensitivities run automatically.')
    if args.bootstrap<100 or args.chunksize<1 or not 0<args.alpha<1: p.error('Use bootstrap >=100, chunksize >=1, and 0<alpha<1')
    if args.ck_input is None:
        args.ck_input = [str(args.input_dir / CK_INPUT_FILENAME)]
    if args.structural_input is None:
        args.structural_input = [str(args.input_dir / STRUCTURAL_INPUT_FILENAME)]
    for label, inputs in [('CK',args.ck_input), ('Structural',args.structural_input)]:
        print(f'{label} input: {inputs}',flush=True)
    paths={'ck':input_paths(args.ck_input),'structural':input_paths(args.structural_input)}
    args.output.mkdir(parents=True,exist_ok=True)
    if any(args.output.resolve() in path.parents for values in paths.values() for path in values):
        p.error('Output directory must not contain selected input CSVs')
    audit=[];records=[]
    for source,files in paths.items(): records.extend(read_entities(files,source,config,args,audit))
    entities,dedup=clean_entities(records)
    data=aggregate_entities(entities,config,args)
    pd.DataFrame(audit).to_csv(args.output/'input_audit.csv',index=False)
    entities.to_csv(args.output/'entity_metrics.csv',index=False)
    data.to_csv(args.output/'issue_metric_deltas.csv',index=False)
    missing=[k for k,s in config['metrics'].items() if s['family']=='primary' and
             not data.loc[data.metric==k,'delta'].notna().any()]
    if missing and not args.allow_missing_primary:
        raise ValueError(f'Primary metrics unavailable: {missing}. Check mapping and input data. Audit CSVs saved. '
                         'Only use --allow-missing-primary if intentionally reporting incomplete coverage.')
    print(f'Analyzing {data[["project","Key"]].drop_duplicates().shape[0]} unique issues; '
          f'{dedup} identical entity/metric duplicates removed.',flush=True)
    turnover = data[(data.n_intro_only > 0) | (data.n_repayment_only > 0)]
    if not turnover.empty and args.pairing == 'phase-wise':
        print(f'NOTE: {len(turnover)} issue/metric combinations use different entity sets at the two boundaries. '
              'See n_intro_only/n_repayment_only in issue_metric_deltas.csv. '
              'Use --pairing matched for a matched-entity sensitivity analysis.',flush=True)
    tests=direct_tests(data,config,args)
    untested=tests[(tests.family=='primary') & (tests.status!='ok')]
    if not untested.empty:
        print('NOTE: some primary SF-NSF tests have insufficient data (at least 2 complete issues per group required). '
              'Their p-values remain NA; this is not a non-significant test result.',flush=True)
    for family in ['primary','exploratory']:
        table=tests[tests.family==family]
        table.to_csv(args.output/f'{family.upper()}_direct_tests.csv',index=False)
        manuscript_table(table).to_csv(args.output/f'{family.upper()}_manuscript_table.csv',index=False)
    export_boxplot_inputs(data, tests, args.output)
    secondary_tests(data,config,args).to_csv(args.output/'SECONDARY_within_group_wilcoxon.csv',index=False)
    wide=data.pivot(index=['project','Key','group'],columns='metric',values=['intro','repayment','delta'])
    wide.columns=['_'.join(c) for c in wide.columns];wide.reset_index().to_csv(args.output/'issue_level_wide.csv',index=False)
    coverage=data.assign(complete_pair=data.delta.notna()).groupby(['project','group','metric']).agg(
        issues=('Key','size'),paired_issues=('complete_pair','sum'),intro_entities=('n_intro_available','sum'),
        repayment_entities=('n_repayment_available','sum'),matched_entities=('n_matched','sum')).reset_index()
    coverage.to_csv(args.output/'coverage_by_project.csv',index=False)
    sensitivity_manifest=run_sensitivities(entities,config,args)
    manifest=dict(sensitivity_analyses=sensitivity_manifest,version=VERSION,args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        config=config,duplicates_removed=dedup,python=platform.python_version(),numpy=np.__version__,
        pandas=pd.__version__,scipy=scipy.__version__,matplotlib=matplotlib.__version__,inputs=[])
    for source,files in paths.items():
        for path in files:
            digest=hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda:stream.read(1024*1024),b''): digest.update(block)
            manifest['inputs'].append(dict(source=source,path=str(path),sha256=digest.hexdigest()))
    (args.output/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (args.output/'analysis_notes.txt').write_text(__doc__+'\n\nMissing primary metrics: '+str(missing)+'\n')
    if not args.no_plots:
        figures(data,tests,args.output)
        combined_primary_boxplots(data,tests,args.output,args.combined_layouts)
    print(tests.loc[tests.family=='primary',['label','SF_n','NSF_n','p_holm','cliffs_delta','status']].to_string(index=False))
    print(f'Outputs: {args.output.resolve()}')


if __name__=='__main__':
    main()
