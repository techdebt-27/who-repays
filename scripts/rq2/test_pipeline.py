#!/usr/bin/env python3
"""Synthetic smoke/invariant tests. Figures from these data are NEVER empirical evidence."""
import os,sys, tempfile,subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0,str(Path(__file__).parent))
from analyze_rq2 import (METRICS,GROUPS,X,validate_input,project_permutation,
                          partial_project_spearman,bootstrap_contrast,analyze,figure3)

def make_sample(path, n_issue=60):
    rg=np.random.default_rng(142)
    data=[]
    for i in range(n_issue):
        project=('KAFKA','CAMEL','SPARK')[i%3]
        group=GROUPS[i%2]
        nfiles=rg.integers(2,6)
        for j in range(nfiles):
            commits=int(rg.integers(1,60))
            row={'project_key':project,'Key':f'{project}-{i+1000}', 'group':group,'indicator':'viomod',
                 'intro_file_path':f'src/org/Test{i}_{j}.java','exists_at_intro':True,'deleted':False,
                 X:commits,'touched_commits':';'.join(f'x{k}' for k in range(commits))}
            for m in METRICS:
                a=float(rg.normal(10,2)); d=float((commits*.24 if group==GROUPS[0] else commits*.40)+rg.normal(0,8))
                if m in ('dit','noc'):d=float(rg.normal(0,2))
                row[f'intro_{m}']=a;row[f'payment_{m}']=a+d;row[f'delta_{m}']=d
            data.append(row)
    pd.DataFrame(data).to_csv(path,index=False)

def test():
    rg=np.random.default_rng(5)
    with tempfile.TemporaryDirectory() as td:
        td=Path(td);path=td/'synthetic_rq3_file_level.csv';make_sample(path)
        raw=pd.read_csv(path)
        d,coverage,duplicates=validate_input(path)
        assert len(d)==len(raw) and duplicates==0
        assert np.allclose(d.delta_fan_in,d.payment_fan_in-d.intro_fan_in)
        issue=d[d.group==GROUPS[0]].groupby(['project_key','Key'],as_index=False)[[X,'delta_fan_in']].mean()
        rr=spearmanr(issue[X],issue.delta_fan_in).statistic
        obs,p,nullmean=project_permutation(issue[X].to_numpy(),issue.delta_fan_in.to_numpy(),issue.project_key.to_numpy(),199,rg)
        assert np.isclose(rr,obs)
        assert 0<=p<=1
        assert np.isfinite(partial_project_spearman(issue[X].to_numpy(),issue.delta_fan_in.to_numpy(),issue.project_key.to_numpy()))
        primary,file,contrast=analyze(d,td/'out',199,199,199,199,15)
        assert len(primary)==18 and len(file)==18 and len(contrast)==18
        for t in (primary,file,contrast):assert all((t.p_holm.dropna()>=0)&(t.p_holm.dropna()<=1))
        assert set(contrast.method)=={'pooled_common_projects','project_demeaned_ranks'}
        assert len(pd.read_csv(td/'out'/'RQ3_issue_level_matched_file_pairs.csv'))==nissues(d)
        figure3(d,td/'out',250,15)
        assert (td/'out'/'Figure3_RQ2_file_commits_vs_nine_metric_deltas.pdf').exists()
        assert (td/'out'/'Figure3_RQ2_file_commits_vs_nine_metric_deltas.png').exists()
        # Require delta boundary consistency
        corrupted=raw.copy();corrupted.loc[0,'delta_fan_in']+=123;corrupted.to_csv(td/'bad.csv',index=False)
        try:validate_input(td/'bad.csv');raise AssertionError('Incorrect delta accepted')
        except ValueError as e:assert 'delta mismatch' in str(e)
        print('PASS: synthetic statistical analysis, 9 metrics, 18+18+18 tests, Figure 3, delta audit, complete output.')

def nissues(d):return sum(len(d[d.group==g][['project_key','Key']].drop_duplicates()) for g in GROUPS)*len(METRICS)

if __name__=='__main__':test()
