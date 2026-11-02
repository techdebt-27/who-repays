#!/usr/bin/env python3
"""Issue-level correlation between ATD file changes and time-to-fix.

Inputs
------
1. GitHub replication package: dataset/ATD-FINAL-DATASET-TRACED.csv
2. TechDebt 2027 result: rq3_file_level_nine_metrics.csv

The primary metric is the mean number of file-modifying commits per tracked
file within an issue. This is NOT the number of distinct commits per issue.

Example:
python analyze_changes_vs_time_to_fix.py \
  --traced ATD-FINAL-DATASET-TRACED.csv \
  --file-metrics rq3_file_level_nine_metrics.csv \
  --outdir results/changes_vs_time_to_fix

Optional: --paired-metric delta_fan_in (restrict to files with valid paired
Fan-In metrics, closer to that metric's RQ2 analytic cohort).
Optional: --issue-metrics ISSUES_with_LevelItem_LevelProcess_PLUS_ChangeBreadth.csv
(includes Num_Commits_Total; unique commits per issue, old time-to-fix study).
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--traced', required=True, type=Path)
    p.add_argument('--file-metrics', required=True, type=Path)
    p.add_argument('--issue-metrics', type=Path, default=None)
    p.add_argument('--paired-metric', default=None,
                   help='e.g. delta_fan_in; use only rows with a valid paired metric')
    p.add_argument('--outdir', type=Path, default=Path('results/changes_vs_time_to_fix'))
    p.add_argument('--bootstrap', type=int, default=2000)
    p.add_argument('--seed', type=int, default=2027)
    args = p.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    issues = pd.read_csv(args.traced, low_memory=False)
    cols = ['Key','Intro Date','Payment Date']
    missing = set(cols).difference(issues.columns)
    if missing:
        raise ValueError(f'Traced dataset missing required columns: {sorted(missing)}')
    if 'indicator' in issues.columns:
        issues = issues[issues['indicator'].astype(str).str.casefold().eq('viomod')].copy()
    issues['Key'] = issues['Key'].astype(str).str.strip().str.upper()
    issues = issues.drop_duplicates('Key').copy()
    issues['intro_at'] = pd.to_datetime(issues['Intro Date'], errors='coerce', utc=True)
    issues['payment_at'] = pd.to_datetime(issues['Payment Date'], errors='coerce', utc=True)
    issues['time_to_fix_days'] = (issues['payment_at'] - issues['intro_at']).dt.total_seconds()/86400

    if 'Is Self-Fixed (Intro=Payment)' in issues.columns:
        flag = issues['Is Self-Fixed (Intro=Payment)'].astype(str).str.strip().str.lower()
        grp_map = {'true':'SF','false':'NSF','yes':'SF','no':'NSF','1':'SF','0':'NSF'}
        issues['repayment_group'] = flag.map(grp_map)
    else:
        issues['repayment_group'] = np.nan

    usecols = {'Key','file_commit_count_intro_to_payment','intro_file_path','group'}
    if args.paired_metric:
        usecols.add(args.paired_metric)
    files = pd.read_csv(args.file_metrics, usecols=lambda c: c in usecols, low_memory=False)
    needed = {'Key','file_commit_count_intro_to_payment'}
    if not needed.issubset(files.columns):
        raise ValueError(f'File-level dataset missing {sorted(needed.difference(files.columns))}')
    if args.paired_metric and args.paired_metric not in files.columns:
        raise ValueError(f'Paired metric {args.paired_metric!r} not found in file-level data')
    files['Key'] = files['Key'].astype(str).str.strip().str.upper()
    files['file_commit_count_intro_to_payment'] = pd.to_numeric(
        files['file_commit_count_intro_to_payment'],errors='coerce')
    files = files[files['file_commit_count_intro_to_payment'].ge(0)].copy()
    if args.paired_metric:
        files = files[pd.to_numeric(files[args.paired_metric],errors='coerce').notna()].copy()

    # The exported file is normally one row per (Key, intro_file_path). Dedupe
    # to prevent counting repeated file rows as independent observations.
    if 'intro_file_path' in files.columns:
        files = files.drop_duplicates(['Key','intro_file_path'])
    else:
        print('WARNING: intro_file_path absent; cannot deduplicate file observations',file=sys.stderr)

    agg = files.groupby('Key',as_index=False).agg(
        mean_file_commit_count=('file_commit_count_intro_to_payment','mean'),
        median_file_commit_count=('file_commit_count_intro_to_payment','median'),
        sum_file_commit_count=('file_commit_count_intro_to_payment','sum'),
        n_tracked_files=('file_commit_count_intro_to_payment','size'))
    if 'group' in files.columns:
        file_group = files.groupby('Key')['group'].first().astype(str).str.casefold()
        group_map = {'self-fixed':'SF','non-self-fixed':'NSF', 'sf':'SF','nsf':'NSF'}
        agg['file_repayment_group'] = agg['Key'].map(file_group.map(group_map))
    merged = agg.merge(issues[['Key','time_to_fix_days','repayment_group']],on='Key',how='left',validate='one_to_one')
    if 'file_repayment_group' in merged:
        disagreement = (merged['repayment_group'].notna() &
                        merged['file_repayment_group'].notna() &
                        merged['repayment_group'].ne(merged['file_repayment_group']))
        if disagreement.any():
            raise ValueError(f'Group inconsistency for {int(disagreement.sum())} ATD issues.')
        merged['repayment_group'] = merged['repayment_group'].fillna(merged['file_repayment_group'])
        merged = merged.drop(columns='file_repayment_group')
    merged['project'] = merged['Key'].str.split('-',n=1).str[0]

    if args.issue_metrics:
        older = pd.read_csv(args.issue_metrics, low_memory=False)
        if not {'Key','Num_Commits_Total'}.issubset(older.columns):
            raise ValueError('Optional issue-metrics CSV requires Key and Num_Commits_Total')
        older['Key'] = older['Key'].astype(str).str.strip().str.upper()
        older = older.drop_duplicates('Key')
        older['Num_Commits_Total'] = pd.to_numeric(older['Num_Commits_Total'],errors='coerce')
        merged = merged.merge(older[['Key','Num_Commits_Total']],on='Key',how='left',validate='one_to_one')

    before=len(merged)
    merged=merged[merged['time_to_fix_days'].notna() &
                  merged['time_to_fix_days'].ge(0) &
                  merged['repayment_group'].isin(['SF','NSF'])].copy()
    print(f'Unique issues in structural file data: {before}; with valid time-to-fix and group: {len(merged)}')
    print('Groups:', merged['repayment_group'].value_counts().to_dict())
    print('NOTE: sum_file_commit_count is file touches, not distinct commits per issue.')
    merged.to_csv(args.outdir/'issue_level_changes_and_ttf.csv',index=False)

    rng = np.random.default_rng(args.seed)
    metrics=['mean_file_commit_count','median_file_commit_count','sum_file_commit_count']
    if 'Num_Commits_Total' in merged:
        metrics.append('Num_Commits_Total')

    rows=[]
    for col in metrics:
        for group in ['All','SF','NSF']:
            sub=merged if group=='All' else merged[merged['repayment_group']==group]
            sub=sub[['Key',col,'time_to_fix_days']].dropna()
            n=len(sub)
            x=sub[col].to_numpy(dtype=float)
            y=sub['time_to_fix_days'].to_numpy(dtype=float)
            if n<4 or len(np.unique(x))<2 or len(np.unique(y))<2:
                rho=pval=ci_low=ci_high=np.nan
            else:
                rho,pval=spearmanr(x,y)
                boots=[]
                for _ in range(args.bootstrap):
                    idx=rng.integers(0,n,size=n)
                    bx,by=x[idx],y[idx]
                    if len(np.unique(bx))>1 and len(np.unique(by))>1:
                        boots.append(spearmanr(bx,by).statistic)
                ci_low,ci_high=(np.percentile(boots,[2.5,97.5])
                                if boots else (np.nan,np.nan))
            rows.append(dict(metric=col,group=group,n_issues=n,rho=rho,p_value=pval,
                             ci_95_low=ci_low,ci_95_high=ci_high))
    result=pd.DataFrame(rows)
    result.to_csv(args.outdir/'spearman_changes_vs_ttf.csv',index=False)
    print(result.to_string(index=False,float_format=lambda x:f'{x:.5g}'))
    print(f'Outputs written to {args.outdir.resolve()}')
    print('CAUTION: correlations reflect association, not causation; longer time windows give more opportunities for modifications.')

if __name__=='__main__':
    main()
