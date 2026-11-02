#!/usr/bin/env python3
"""One-command pipeline for TechDebt RQ2, retaining the audited extraction source.

Examples: python run_full_rq2.py --mode full --final-dir ... --ck-dir ... --repos-root ... --outdir ...
          python run_full_rq2.py --mode analyze --file-level ... --outdir ...
"""
from pathlib import Path
import argparse,subprocess,sys,json


def call(module,args):
    cmd=[sys.executable,str(Path(__file__).with_name(module))]+list(map(str,args))
    print('COMMAND:', ' '.join(cmd),flush=True)
    subprocess.run(cmd,check=True)


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('--mode',choices=('full','extract','analyze'),default='full')
    p.add_argument('--final-dir',type=Path,help='FINAL folder of *UNDERSTAND-METRICS-CC-COMPLEXITY-FINAL.csv')
    p.add_argument('--ck-dir',type=Path,help='CK-METRICS folder of *-UNDERSTAND-CK-METRICS*.csv')
    p.add_argument('--repos-root',type=Path,help='Root containing local cloned Apache project repositories')
    p.add_argument('--outdir',type=Path,required=True)
    p.add_argument('--file-level',type=Path,help='Existing rq3_file_level_nine_metrics.csv for --mode analyze')
    p.add_argument('--cached-git',type=Path,help='Existing Git tracking CSV; reuse Git history but rejoin CK summaries')
    p.add_argument('--rq1-cohort',type=Path,help='Optional RQ1_issue_metric_deltas.csv for metric-specific cohort restriction')
    p.add_argument('--allow-network',action='store_true',help='Explicitly enable Git cloning/fetching; off by default')
    p.add_argument('--csv-pattern',default='*UNDERSTAND-METRICS-CC-COMPLEXITY-FINAL.csv')
    p.add_argument('--ck-pattern',default='*-UNDERSTAND-CK-METRICS*.csv')
    p.add_argument('--permutations',type=int,default=9999)
    p.add_argument('--issue-bootstraps',type=int,default=1999)
    p.add_argument('--file-bootstraps',type=int,default=1999)
    p.add_argument('--contrast-bootstraps',type=int,default=1999)
    p.add_argument('--seed',type=int,default=20261021)
    p.add_argument('--plot-max-per-group-metric',type=int,default=0)
    opt=p.parse_args();opt.outdir.mkdir(parents=True,exist_ok=True)
    extraction_dir=opt.outdir/'extraction'
    if opt.mode in ('full','extract'):
        if not opt.final_dir or not opt.ck_dir or not opt.repos_root:
            p.error('--final-dir, --ck-dir and --repos-root required for extraction')
        args=['--input-dir',opt.final_dir,'--ck-dir',opt.ck_dir,'--repos-root',opt.repos_root,
             '--output-dir',extraction_dir,'--csv-pattern',opt.csv_pattern,'--ck-pattern',opt.ck_pattern,
             '--skip-figures']
        if opt.allow_network:args+=['--allow-network']
        if opt.cached_git:args+=['--cached-git',opt.cached_git]
        if opt.rq1_cohort:args+=['--primary-cohort',opt.rq1_cohort]
        call('legacy_git_ck_extractor.py',args)
    if opt.mode in ('full','analyze'):
        filelevel=(extraction_dir/'rq3_file_level_nine_metrics.csv') if opt.mode=='full' else opt.file_level
        if not filelevel:p.error('--file-level required when --mode analyze')
        args=['--input',filelevel,'--output',opt.outdir,'--permutations',opt.permutations,
              '--issue-bootstraps',opt.issue_bootstraps,'--file-bootstraps',opt.file_bootstraps,
              '--contrast-bootstraps',opt.contrast_bootstraps,'--seed',opt.seed,
              '--plot-max-per-group-metric',opt.plot_max_per_group_metric]
        if opt.rq1_cohort:args+=['--rq1-cohort',opt.rq1_cohort]
        call('analyze_rq2.py',args)
    print('SUCCESS. Outputs in:',opt.outdir)

if __name__=='__main__':main()
