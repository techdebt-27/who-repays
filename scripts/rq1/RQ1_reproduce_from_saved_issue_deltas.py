#!/usr/bin/env python3
"""Recompute nine RQ1 SF-vs-NSF issue-level MWU tests from saved per-issue deltas.

This does NOT reconstruct entity matching; use it to independently reproduce the
between-group test statistics from saved matched-entity issue deltas.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

METRICS=['fan_in','fan_out','cbo','rfc_understand','lcom','dit','noc',
         'cyclomatic_complexity','cognitive_complexity']

def run(path):
    d=pd.read_csv(path)
    required={'Key','project','metric','group','delta'}
    if not required.issubset(d.columns):
        raise ValueError(f'Missing columns: {sorted(required-set(d.columns))}')
    if d.duplicated(['project','Key','metric']).any():
        raise ValueError('Duplicate project+issue+metric records; do not double-count issues')
    if 'family' in d:
        d=d[d.family.eq('primary')]
    rows=[]
    for metric in METRICS:
        ss=d[d.metric.eq(metric)]
        x=pd.to_numeric(ss.loc[ss.group.eq('Self-Fixed'),'delta'],errors='coerce').dropna().to_numpy()
        y=pd.to_numeric(ss.loc[ss.group.eq('Non-Self-Fixed'),'delta'],errors='coerce').dropna().to_numpy()
        if not len(x) or not len(y):
            rows.append(dict(metric=metric,n_sf=len(x),n_nsf=len(y),U=np.nan,p_raw=np.nan,cliffs_delta=np.nan, median_sf=np.nan,median_nsf=np.nan))
            continue
        z=mannwhitneyu(x,y,alternative='two-sided',method='asymptotic',use_continuity=True)
        rows.append(dict(metric=metric,n_sf=len(x),n_nsf=len(y),U=float(z.statistic),p_raw=float(z.pvalue),
                         cliffs_delta=2*float(z.statistic)/(len(x)*len(y))-1,
                         median_sf=float(np.median(x)),median_nsf=float(np.median(y))))
    res=pd.DataFrame(rows)
    _,holm,_,_=multipletests(res.p_raw.fillna(1),method='holm')
    res['p_holm']=holm
    return res

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True,type=Path)
    p.add_argument('--outdir',required=True,type=Path)
    a=p.parse_args();a.outdir.mkdir(parents=True,exist_ok=True)
    df=run(a.input); out=a.outdir/'RQ1_recomputed_from_saved_issue_deltas.csv'
    df.to_csv(out,index=False)
    print(df.to_string(index=False))
    print('Written:',out)
