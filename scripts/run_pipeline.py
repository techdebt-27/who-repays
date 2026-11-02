#!/usr/bin/env python3
"""Easy-to-run nine-metric TechDebt replication pipeline.

Full source: python run_pipeline.py --mode source --steps extract,rq1,rq2 --confirm-destructive-git
Existing extraction: python run_pipeline.py --mode source --steps rq1,rq2
Bundled historical data: python run_pipeline.py --mode processed --steps rq1,rq2

The source extractor changes Git worktrees: use disposable clones and explicitly
confirm. Processed mode does not reproduce data extraction or entity matching.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
PIPE=ROOT/'scripts'/'pipeline'
EXTRACT=ROOT/'scripts'/'extraction'/'understand_dependency_complexity_extraction_9_metrics.py'

def config_from(path):
    if not path:return {}
    p=Path(path).resolve()
    if not p.is_file():raise FileNotFoundError(p)
    cfg=json.loads(p.read_text(encoding='utf-8'))
    # Relative paths intentionally resolved against package root, for predictable paths.
    for field in ('traced_csv_dir','repos_root','work_tmpdir','output_dir','scitools_bin'):
        if field in cfg and cfg[field] and field!='scitools_bin':
            value=Path(cfg[field]);cfg[field]=str((ROOT/value).resolve() if not value.is_absolute() else value)
    return cfg

def command(script,args,env=None):
    cmd=[sys.executable,str(script),*map(str,args)]
    print('\n$ '+' '.join(cmd),flush=True)
    subprocess.run(cmd,cwd=ROOT,env=env,check=True)

def require_dir(path,description):
    if not path or not Path(path).is_dir():raise FileNotFoundError(description+': '+str(path))

def run_extract(cfg):
    traced=Path(cfg.get('traced_csv_dir',ROOT/'data/input'))
    repos=cfg.get('repos_root')
    scratch=cfg.get('work_tmpdir',str(ROOT/'data'/'scratch'))
    require_dir(traced,'Traced dataset folder');require_dir(repos,'Apache clone root')
    csv_files=sorted(traced.glob('*.csv'))
    if not csv_files:raise FileNotFoundError(f'No *.csv traced datasets in {traced}')
    out=Path(cfg.get('output_dir',ROOT/'results/generated'))
    final=out/'extraction'/'FINAL'
    ck=out/'extraction'/'CK-METRICS'
    final.mkdir(parents=True,exist_ok=True);ck.mkdir(parents=True,exist_ok=True)
    Path(scratch).mkdir(parents=True,exist_ok=True)
    for inp in csv_files:
        # Require one project per CSV to produce unambiguous paired FINAL/CK files.
        import pandas as pd
        keys=pd.read_csv(inp,usecols=['Key'],dtype=str)['Key'].dropna().astype(str).str.split('-').str[0].str.upper().unique()
        if len(keys)!=1:raise ValueError(f'{inp}: expected a single project Key prefix, got {keys}')
        project=keys[0]
        base=project+'-UNDERSTAND-METRICS-CC-COMPLEXITY-FINAL.csv'
        env=os.environ.copy()
        env.update({'ATD_DATA_DIRECTORY':str(ROOT),'ATD_INPUT_CSV':str(inp),
                    'ATD_REPOS_ROOT':str(repos),'ATD_OUTPUT_CSV':str(final/base),
                    'ATD_CK_OUTPUT_CSV':str(ck/(project+'-UNDERSTAND-CK-METRICS-9-METRICS.csv')),
                    'WORK_TMPDIR':str(scratch)})
        if cfg.get('scitools_bin'):env['SCITOOLS_UND_BIN']=str(cfg['scitools_bin'])
        command(EXTRACT,[],env)
    return final,ck

def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config',type=Path,default=ROOT/'config/config.example.json')
    p.add_argument('--mode',choices=['source','processed'],default='processed')
    p.add_argument('--steps',default='rq1,rq2',help='Comma-separated: extract,rq1,rq2 (or all)')
    p.add_argument('--confirm-destructive-git',action='store_true')
    p.add_argument('--fast',action='store_true',help='Smoke/testing ONLY: reduced bootstrap/permutation iterations')
    p.add_argument('--no-plots',action='store_true',help='Skip RQ1 plots')
    a=p.parse_args(); cfg=config_from(a.config)
    steps=['extract','rq1','rq2'] if a.steps=='all' else [x.strip() for x in a.steps.split(',')]
    if not steps or any(x not in ('extract','rq1','rq2') for x in steps):p.error('Use --steps extract,rq1,rq2 or a subset')
    if a.mode=='processed' and 'extract' in steps:p.error('Extraction needs --mode source')
    if 'extract' in steps and not a.confirm_destructive_git:
        p.error('Source extraction runs git reset --hard and git clean; provide --confirm-destructive-git after using disposable clones')
    out=Path(cfg.get('output_dir',ROOT/'results/generated'));out.mkdir(parents=True,exist_ok=True)
    final=out/'extraction/FINAL';ck=out/'extraction/CK-METRICS'
    if 'extract' in steps:final,ck=run_extract(cfg)
    if 'rq1' in steps:
        dest=out/'rq1';dest.mkdir(parents=True,exist_ok=True)
        if a.mode=='source':
            require_dir(final,'Extracted FINAL directory')
            command(PIPE/'prepare_rq1_from_extraction.py',['--final-dir',final,'--outdir',dest])
            issue=dest/'RQ1_issue_metric_deltas.csv'
            filecsv=dest/'RQ1_file_metric_deltas_exploratory.csv'
        else:
            issue=ROOT/'data/processed/PRIMARY_boxplot_data.csv'
            if not issue.is_file():raise FileNotFoundError('Bundled PRIMARY_boxplot_data.csv missing')
            filecsv=None
            print('RQ1: reusing SAVED historical issue-level deltas (not re-extracting classes)')
            # Legacy saved file names RFC as rfc_understand, prep handles aliases below.
            command(PIPE/'normalize_saved_rq1.py',['--input',issue,'--output',dest/'RQ1_issue_metric_deltas.csv'])
            issue=dest/'RQ1_issue_metric_deltas.csv'
        args=['--issue-csv',issue,'--outdir',dest]
        if filecsv and filecsv.is_file():args+=['--file-csv',filecsv]
        if a.no_plots:args+=['--no-plots']
        command(PIPE/'run_rq1.py',args)
    if 'rq2' in steps:
        dest=out/'rq2'
        kw=['--mode',a.mode,'--outdir',dest,
            '--permutations',199 if a.fast else cfg.get('permutations',9999),
            '--issue-bootstraps',199 if a.fast else cfg.get('issue_bootstraps',1999),
            '--file-bootstraps',199 if a.fast else cfg.get('file_bootstraps',1999),
            '--contrast-bootstraps',199 if a.fast else cfg.get('contrast_bootstraps',1999),
            '--seed',cfg.get('seed',20261021)]
        if a.mode=='source':
            require_dir(final,'Extracted FINAL directory');require_dir(ck,'Extracted CK directory')
            require_dir(cfg.get('repos_root'),'Apache clone root')
            kw+=['--final-dir',final,'--ck-dir',ck,'--repos-root',cfg['repos_root']]
        else:kw+=['--file-level',ROOT/'data/processed/rq3_file_level_nine_metrics.csv']
        command(PIPE/'run_rq2.py',kw)
    manifest={'mode':a.mode,'steps':steps,'fast_test_only':a.fast,
              'source_extraction_ran':('extract' in steps),'note':'VioMod only; project-aware RQ2 issue permutations and whole-issue file bootstrap; matched endpoint deltas',
              'seed':cfg.get('seed',20261021)}
    (out/'pipeline_run_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('\nCompleted requested steps. Results:',out.resolve())
if __name__=='__main__':main()
