#!/usr/bin/env python3
"""Audited RQ2 reanalysis from raw file-within-issue export.

PRIMARY: project-stratified within-project issue permutation Spearman (18 tests).
SECONDARY: file-level Spearman, bootstrap WHOLE issues within projects (18 tests).
DIRECT SF vs NSF: project-overlap subset; bootstrap issues within project x group;
  report marginal difference and project-adjusted rank-residual correlation difference
  with separate Holm corrections over nine metrics (approximate bootstrap p).

A direct SF--NSF contrast is observational; no causal attribution. Permutations
assume issue exchangeability within projects; issue-cluster bootstrap does not
resolve shared files across issues or correlation between different projects.
"""
from __future__ import annotations
import argparse, json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, rankdata
from statsmodels.stats.multitest import multipletests

METRICS = ('fan_in','fan_out','cbo','rfc_proxy','lcom','dit','noc','cyclomatic_complexity','cognitive_complexity')
LABELS = {'fan_in':'Fan-In','fan_out':'Fan-Out','cbo':'CBO','rfc_proxy':'RFC proxy',
          'lcom':'LCOM','dit':'DIT','noc':'NOC','cyclomatic_complexity':'Cyclomatic complexity',
          'cognitive_complexity':'Cognitive complexity'}
GROUPS=('Self-Fixed','Non-Self-Fixed')
COLOR={'Self-Fixed':'#1f77b4','Non-Self-Fixed':'#ff7f0e'}
X='file_commit_count_intro_to_payment'


def norm_group(v):
    s=str(v).strip().lower().replace('_','-').replace(' ','-')
    if s in {'self-fixed','self','sf','true','1','yes'}:return GROUPS[0]
    if s in {'non-self-fixed','nonself-fixed','nsf','false','0','no'}:return GROUPS[1]
    raise ValueError(f'Unrecognized repayment-group label: {v!r}')


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()


def require(columns, frame):
    missing=set(columns)-set(frame)
    if missing:raise ValueError('Missing required columns: '+', '.join(sorted(missing)))


def validate_input(path, indicator='viomod', cohort=None, strict_9=True):
    src=pd.read_csv(path,low_memory=False)
    require({'Key','group','project_key','intro_file_path',X},src)
    if indicator!='all':
        require({'indicator'},src)
        src=src[src.indicator.astype(str).str.strip().str.lower().eq(indicator.lower())].copy()
    if src.empty:raise ValueError('No observations after indicator filter')
    src['group']=src.group.map(norm_group)
    src['Key']=src.Key.astype(str).str.strip().str.upper()
    src['project_key']=src.project_key.astype(str).str.strip().str.upper()
    src['intro_file_path']=src.intro_file_path.astype(str).str.replace('\\\\','/',regex=False).str.strip()
    src=src[src.intro_file_path.str.lower().str.endswith('.java')].copy()
    if src.empty:raise ValueError('No Java observations')
    src[X]=pd.to_numeric(src[X],errors='raise')
    bad=(~np.isfinite(src[X]))|(src[X]<0)|(src[X]%1!=0)
    if bad.any():raise ValueError(f'Invalid commit counts: {int(bad.sum())} rows')
    for col in ['exists_at_intro','deleted']:
        if col in src:
            truth=src[col].astype(str).str.lower().str.strip().map({'true':True,'false':False,'1':True,'0':False,'yes':True,'no':False})
            if truth.isna().any():raise ValueError(f'Unrecognized {col}: {src.loc[truth.isna(),col].unique()}')
            src=src[truth if col=='exists_at_intro' else ~truth].copy()
    # If git-tracking commit hashes are exported, audit count integrity.
    if 'touched_commits' in src:
        recorded=src.touched_commits.fillna('').astype(str).map(
            lambda z:len(set(v.strip() for v in z.split(';') if v.strip())))
        unequal=recorded.ne(src[X])
        if unequal.any():
            raise ValueError(f'touched_commits mismatch for {int(unequal.sum())} rows; first issue {src.loc[unequal,"Key"].iloc[0]}')
    # Recompute every delta from its two lifecycle snapshots; fail closed on mismatch.
    coverage=[]
    for metric in METRICS:
        alias='rfc_understand' if metric=='rfc_proxy' else metric
        left=next((c for c in (f'intro_{metric}',f'intro_{alias}') if c in src),None)
        right=next((c for c in (f'payment_{metric}',f'payment_{alias}',f'repayment_{metric}') if c in src),None)
        if not (left and right):raise ValueError(f'{metric}: missing intro/payment columns; cannot independently verify deltas')
        a=pd.to_numeric(src[left],errors='raise').replace([np.inf,-np.inf],np.nan)
        b=pd.to_numeric(src[right],errors='raise').replace([np.inf,-np.inf],np.nan)
        if src[left].notna().sum()!=a.notna().sum() or src[right].notna().sum()!=b.notna().sum():
            raise ValueError(f'{metric}: non-finite source values')
        fresh=b-a
        dcol=f'delta_{metric}'
        if dcol in src:
            old=pd.to_numeric(src[dcol],errors='raise')
            mismatch=(old.notna()!=fresh.notna()) | (old.notna() & fresh.notna() & (~np.isclose(old.fillna(0),fresh.fillna(0),rtol=1e-9,atol=1e-7)))
            if mismatch.any():raise ValueError(f'{metric}: {int(mismatch.sum())} delta mismatches; issue {src.loc[mismatch,"Key"].iloc[0]}')
        src[f'intro_{metric}']=a
        src[f'payment_{metric}']=b
        src[dcol]=fresh
        coverage.append({'metric':metric,'n_matched':int(fresh.notna().sum()),
                         'n_intro_only':int((a.notna() & b.isna()).sum()),
                         'n_repayment_only':int((a.isna() & b.notna()).sum())})
    ids=['project_key','Key','intro_file_path']
    exact_cols=['group',X]+[f'intro_{m}' for m in METRICS]+[f'payment_{m}' for m in METRICS]+[f'delta_{m}' for m in METRICS]
    old_len=len(src)
    src=src.drop_duplicates(subset=ids+exact_cols).copy()
    if src.duplicated(ids).any():raise ValueError('Conflicting duplicate issue/file rows: '+str(src.loc[src.duplicated(ids,keep=False),ids].head().to_dict('records')))
    if (src.groupby(['project_key','Key']).group.nunique()>1).any():raise ValueError('Issue has conflicting SF/NSF groups')
    if cohort:
        allowed=pd.read_csv(cohort,dtype=str)
        require({'project','Key','metric','group','delta'},allowed)
        allowed=allowed[allowed.delta.notna()].copy()
        allowed['project']=allowed.project.str.upper().str.strip()
        allowed['Key']=allowed.Key.str.upper().str.strip()
        allowed['group']=allowed.group.map(norm_group)
        good={(r.project,r.Key,r.metric,r.group) for r in allowed.itertuples(index=False)}
        for metric in METRICS:
            good_row=np.fromiter(((p,k,metric,g) in good for p,k,g in zip(src.project_key,src.Key,src.group)),dtype=bool,count=len(src))
            src.loc[~good_row,f'delta_{metric}']=np.nan
    diag=pd.DataFrame(coverage)
    diag['n_after_filter']=len(src)
    diag['n_valid_after_cohort']=[src[f'delta_{m}'].notna().sum() for m in METRICS]
    if strict_9 and (diag.n_valid_after_cohort==0).any():raise ValueError('At least one metric has 0 valid pairs')
    return src,diag,old_len-len(src)


def rho(x,y):
    if len(x)<3 or np.unique(x).size<2 or np.unique(y).size<2:return np.nan
    return float(spearmanr(x,y).statistic)


def grp_blocks(values):
    return [np.flatnonzero(values==v) for v in np.unique(values)]


def project_permutation(x,y,project,B,rng):
    observed=rho(x,y)
    if not np.isfinite(observed):return np.nan,np.nan,np.nan
    blocks=[ix for ix in grp_blocks(project) if len(ix)>1]
    if not blocks:return np.nan,np.nan,np.nan
    xx=rankdata(x); yy=rankdata(y)
    xx=xx-xx.mean(); yy=yy-yy.mean()
    den=np.linalg.norm(xx)*np.linalg.norm(yy)
    null=np.empty(B,dtype=float)
    idx=np.arange(len(x))
    for b in range(B):
        idx[:]=np.arange(len(x))
        for ix in blocks:idx[ix]=rng.permutation(ix)
        null[b]=float(xx@yy[idx]/den)
    hi=(1+np.count_nonzero(null>=observed-1e-12))/(B+1)
    lo=(1+np.count_nonzero(null<=observed+1e-12))/(B+1)
    return observed,float(min(1,2*min(hi,lo))),float(np.mean(null))


def stratified_issue_bootstrap(x,y,project,B,rng):
    if B<=0:return np.nan,np.nan,0
    blocks=grp_blocks(project); reps=[]
    for _ in range(B):
        ind=np.concatenate([rng.choice(ix,size=len(ix),replace=True) for ix in blocks])
        z=rho(x[ind],y[ind])
        if np.isfinite(z):reps.append(z)
    if len(reps)<max(50,int(.5*B)):return np.nan,np.nan,len(reps)
    return *np.quantile(reps,[.025,.975]),len(reps)


def centered_boot_p(draws,observed):
    if len(draws)<50:return np.nan
    centered=np.asarray(draws)-observed
    return float((1+np.count_nonzero(np.abs(centered)>=abs(observed)-1e-12))/(len(draws)+1))


def issue_cluster_file_bootstrap(data,metric,B,rng):
    if B<=0:return (np.nan,)*3+(0,)
    ycol=f'delta_{metric}'
    x=data[X].to_numpy(float);y=data[ycol].to_numpy(float)
    observed=rho(x,y)
    if not np.isfinite(observed):return (np.nan,)*3+(0,)
    clusters=[];project=[]
    for (p,k), ix in data.groupby(['project_key','Key'],sort=False).indices.items():
        clusters.append(np.asarray(ix,dtype=int)); project.append(p)
    blocks=grp_blocks(np.asarray(project)); draws=[]
    for _ in range(B):
        chosen=np.concatenate([rng.choice(ix,size=len(ix),replace=True) for ix in blocks])
        ids=np.concatenate([clusters[i] for i in chosen]); rr=rho(x[ids],y[ids])
        if np.isfinite(rr):draws.append(rr)
    if len(draws)<max(50,int(.5*B)):return np.nan,np.nan,np.nan,len(draws)
    low,high=np.quantile(draws,[.025,.975]);p=centered_boot_p(draws,observed)
    return float(low),float(high),p,len(draws)


def partial_project_spearman(x,y,projects):
    """Project-demeaned globally ranked variables; descriptive adjusted association."""
    if len(x)<3 or len(np.unique(x))<2 or len(np.unique(y))<2:return np.nan
    a=rankdata(x).astype(float);b=rankdata(y).astype(float)
    for block in grp_blocks(np.asarray(projects)):
        a[block]-=a[block].mean();b[block]-=b[block].mean()
    denom=np.linalg.norm(a)*np.linalg.norm(b)
    return float(a@b/denom) if denom>0 else np.nan


def bootstrap_contrast(left,right,B,rng,adjusted):
    """Within project/group whole-issue resampling; same common-project population."""
    fn=partial_project_spearman if adjusted else lambda x,y,p:rho(x,y)
    def arrays(frame):
        return frame[X].to_numpy(float),frame.y.to_numpy(float),frame.project_key.to_numpy(str)
    x0,y0,p0=arrays(left);x1,y1,p1=arrays(right)
    obs0,obs1=fn(x0,y0,p0),fn(x1,y1,p1)
    if not np.isfinite(obs0) or not np.isfinite(obs1):return (np.nan,)*6+(0,)
    d=obs1-obs0; blocks0=grp_blocks(p0);blocks1=grp_blocks(p1)
    draws=[]
    for _ in range(B):
        ix0=np.concatenate([rng.choice(ix,size=len(ix),replace=True) for ix in blocks0])
        ix1=np.concatenate([rng.choice(ix,size=len(ix),replace=True) for ix in blocks1])
        r0=fn(x0[ix0],y0[ix0],p0[ix0]);r1=fn(x1[ix1],y1[ix1],p1[ix1])
        if np.isfinite(r0) and np.isfinite(r1):draws.append(r1-r0)
    if len(draws)<max(50,int(.5*B)):return float(obs0),float(obs1),float(d),np.nan,np.nan,np.nan,len(draws)
    lo,hi=np.quantile(draws,[.025,.975]);pv=centered_boot_p(draws,d)
    return float(obs0),float(obs1),float(d),float(lo),float(hi),pv,len(draws)


def holm(frame,pcol):
    raw=pd.to_numeric(frame[pcol],errors='coerce')
    frame['p_holm']=multipletests(raw.fillna(1).to_numpy(),method='holm')[1]
    frame.loc[raw.isna(),'p_holm']=np.nan
    frame['significant_holm']=frame.p_holm<.05
    return frame


def category(r):
    if not np.isfinite(r):return 'NA'
    a=abs(r)
    return 'Negligible' if a<.1 else 'Weak' if a<.3 else 'Moderate' if a<.5 else 'Strong'


def analyze(df,out,permutations,issue_bootstraps,file_bootstraps,contrast_bootstraps,seed):
    rng=np.random.default_rng(seed)
    primary=[];secondary=[];pairs=[];contrasts=[]
    for metric in METRICS:
        samples={}
        for group in GROUPS:
            raw=df[df.group.eq(group)][['project_key','Key',X,f'delta_{metric}']].dropna().copy()
            raw=raw.rename(columns={f'delta_{metric}':'y'})
            if len(raw)<3:raise ValueError(f'Insufficient matched file rows for {metric}/{group}')
            issue=raw.groupby(['project_key','Key'],as_index=False)[[X,'y']].mean()
            if len(issue)<3:raise ValueError(f'Insufficient issues for {metric}/{group}')
            issue['group']=group;issue['metric']=metric
            pairs.append(issue.rename(columns={X:'issue_mean_commit_count','y':'issue_mean_file_delta'}))
            samples[group]=issue
            x=issue[X].to_numpy(float);y=issue.y.to_numpy(float);p=issue.project_key.to_numpy(str)
            rr,pv,center=project_permutation(x,y,p,permutations,rng)
            ci_lo,ci_hi,nboot=stratified_issue_bootstrap(x,y,p,issue_bootstraps,rng)
            primary.append({'metric':metric,'label':LABELS[metric],'group':group,
               'n_issues':len(issue),'n_files':len(raw),'n_projects':len(np.unique(p)),
               'rho':rr,'effect_category':category(rr),'direction':'Positive' if rr>0 else 'Negative' if rr<0 else 'Zero',
               'raw_p_project_permutation':pv,'null_rho_mean':center,
               'ci95_low':ci_lo,'ci95_high':ci_hi,'successful_bootstraps':nboot})
            rr_file=rho(raw[X].to_numpy(float),raw.y.to_numpy(float))
            cilo,cihi,fp,nb=issue_cluster_file_bootstrap(raw.rename(columns={'y':f'delta_{metric}'}),metric,file_bootstraps,rng)
            secondary.append({'metric':metric,'label':LABELS[metric],'group':group,
               'n_files':len(raw),'n_issues':len(issue),'n_projects':len(np.unique(p)),
               'rho_file':rr_file,'effect_category':category(rr_file),
               'direction':'Positive' if rr_file>0 else 'Negative' if rr_file<0 else 'Zero',
               'raw_p_centered_issue_bootstrap':fp,'ci95_low_issue_bootstrap':cilo,
               'ci95_high_issue_bootstrap':cihi,'successful_bootstraps':nb})
            print(f'{metric:24s} {group:14s} issues={len(issue):4d} rho_issue={rr:+.3f} rho_file={rr_file:+.3f}',flush=True)
        # Direct contrast: use ONLY projects containing at least one SF and one NSF issue.
        a=samples[GROUPS[0]];b=samples[GROUPS[1]]
        common=set(a.project_key)&set(b.project_key)
        aa=a[a.project_key.isin(common)];bb=b[b.project_key.isin(common)]
        for mode,adjusted in [('pooled_common_projects',False),('project_demeaned_ranks',True)]:
            if len(aa)<3 or len(bb)<3:
                stat=(np.nan,)*6+(0,)
            else:stat=bootstrap_contrast(aa,bb,contrast_bootstraps,rng,adjusted)
            r0,r1,d,lo,hi,pv,success=stat
            contrasts.append({'metric':metric,'label':LABELS[metric],'method':mode,
               'n_common_projects':len(common),'n_sf_issues':len(aa),'n_nsf_issues':len(bb),
               'rho_sf':r0,'rho_nsf':r1,'delta_rho_nsf_minus_sf':d,
               'ci95_low':lo,'ci95_high':hi,'raw_p_centered_cluster_bootstrap':pv,
               'successful_bootstraps':success})
    t1=holm(pd.DataFrame(primary),'raw_p_project_permutation')
    t2=holm(pd.DataFrame(secondary),'raw_p_centered_issue_bootstrap')
    t3=pd.DataFrame(contrasts)
    out.mkdir(parents=True,exist_ok=True)
    for mode in t3.method.unique():
        ix=t3.method.eq(mode)
        t3.loc[ix,'p_holm']=holm(t3.loc[ix].copy(),'raw_p_centered_cluster_bootstrap').p_holm.to_numpy()
    t3['significant_holm']=t3.p_holm<.05
    t1.to_csv(out/'RQ3_PRIMARY_issue_spearman_project_permutation.csv',index=False)
    t2.to_csv(out/'RQ3_SENSITIVITY_file_spearman_issue_cluster_bootstrap.csv',index=False)
    t3.to_csv(out/'RQ2_DIRECT_SF_NSF_correlation_difference.csv',index=False)
    pd.concat(pairs,ignore_index=True).to_csv(out/'RQ3_issue_level_matched_file_pairs.csv',index=False)
    return t1,t2,t3


def figure3(df,out,plot_max,seed):
    rng=np.random.default_rng(seed)
    fig,axes=plt.subplots(3,3,figsize=(14.8,10.1),layout='constrained')
    for ax,metric in zip(axes.flat,METRICS):
        for group in GROUPS:
            d=df[df.group.eq(group)][[X,f'delta_{metric}']].dropna()
            if len(d)<2:continue
            x=d[X].to_numpy(float);y=d[f'delta_{metric}'].to_numpy(float)
            if plot_max>0 and len(d)>plot_max:
                sel=rng.choice(len(d),size=plot_max,replace=False)
                ax.scatter(x[sel],y[sel],s=6,alpha=.17,color=COLOR[group],rasterized=True)
            else:ax.scatter(x,y,s=6,alpha=.17,color=COLOR[group],rasterized=True)
            if np.unique(x).size>1:
                slope,intercept=np.polyfit(x,y,1)
                xx=np.linspace(x.min(),x.max(),120)
                ax.plot(xx,slope*xx+intercept,color=COLOR[group],lw=1.8)
        ax.axhline(0,color='gray',ls='--',lw=.8,alpha=.8)
        ax.set_title(LABELS[metric],fontsize=10,fontweight='bold')
        ax.set_xlabel('File commit count',fontsize=9)
        ax.set_ylabel('Signed file-level delta',fontsize=9)
        ax.grid(axis='y',alpha=.18)
        ax.tick_params(labelsize=8)
        ax.spines[['right','top']].set_visible(False)
    h=[plt.Line2D([0],[0],color=COLOR[g],lw=2,label='SF' if g==GROUPS[0] else 'NSF') for g in GROUPS]
    fig.legend(handles=h,loc='outside lower center',ncol=2,frameon=False)
    for ext in ('png','pdf'):
        fig.savefig(out/f'Figure3_RQ2_file_commits_vs_nine_metric_deltas.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)


def summary_text(t1,t2,t3):
    lines=['RQ2 result summaries','Primary: 18 issue-level Spearman correlations, p from within-project permutations.',
           'Sensitivity: file-level Spearman, whole issue cluster bootstrap within project.',
           'Direct contrasts: common projects only; bootstrap issues within project x group.','']
    for m in METRICS:
        rows=t1[t1.metric.eq(m)]
        rs=rows.set_index('group')
        vals=' / '.join(f'{g}: rho={rs.loc[g,"rho"]:+.3f}, p_Holm={rs.loc[g,"p_holm"]:.4g}' for g in GROUPS)
        lines.append(f'{LABELS[m]}: {vals}')
    lines+=['','Caution: project-stratified permutation tests pooled rho against within-project shuffled issue associations; bootstrap contrast p-values are approximate, not causal and not a randomized group effect.',
            'Repeated physical files across issues and temporal overlaps may induce dependence beyond the issue clusters.']
    return '\n'.join(lines)+'\n'


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',required=True,type=Path,help='Raw rq3_file_level_nine_metrics.csv')
    ap.add_argument('--output',required=True,type=Path)
    ap.add_argument('--indicator',default='viomod')
    ap.add_argument('--rq1-cohort',type=Path,default=None,help='optional RQ1 issue-metric CSV; filters separately per metric')
    ap.add_argument('--permutations',type=int,default=9999)
    ap.add_argument('--issue-bootstraps',type=int,default=1999)
    ap.add_argument('--file-bootstraps',type=int,default=1999)
    ap.add_argument('--contrast-bootstraps',type=int,default=1999)
    ap.add_argument('--seed',type=int,default=20261021)
    ap.add_argument('--plot-max-per-group-metric',type=int,default=0,help='0 plots all points, otherwise subsamples displayed points ONLY; fits use full data')
    ap.add_argument('--skip-plots',action='store_true')
    opt=ap.parse_args()
    for key in ('permutations','issue_bootstraps','file_bootstraps','contrast_bootstraps'):
        if getattr(opt,key)<199:ap.error(f'--{key.replace("_","-")} should be at least 199')
    if not opt.input.is_file():ap.error('Missing file-level input: '+str(opt.input))
    opt.output.mkdir(parents=True,exist_ok=True)
    d,coverage,dedupe=validate_input(opt.input,opt.indicator,opt.rq1_cohort)
    d.to_csv(opt.output/'rq3_file_level_nine_metrics_VALIDATED.csv',index=False)
    coverage.to_csv(opt.output/'RQ2_coverage_audit.csv',index=False)
    t1,t2,t3=analyze(d,opt.output,opt.permutations,opt.issue_bootstraps,opt.file_bootstraps,opt.contrast_bootstraps,opt.seed)
    if not opt.skip_plots:figure3(d,opt.output,opt.plot_max_per_group_metric,opt.seed)
    (opt.output/'RQ2_README_RESULTS.txt').write_text(summary_text(t1,t2,t3))
    (opt.output/'RQ2_run_manifest.json').write_text(json.dumps({
       'input_file':str(opt.input),'sha256':sha256(opt.input),'indicator':opt.indicator,
       'cohort':str(opt.rq1_cohort) if opt.rq1_cohort else None,
       'n_files':len(d),'n_issues':int(d[['project_key','Key']].drop_duplicates().shape[0]),
       'exact_duplicates_removed':dedupe,'seed':opt.seed,
       'permutations':opt.permutations,'issue_bootstraps':opt.issue_bootstraps,
       'file_bootstraps':opt.file_bootstraps,'contrast_bootstraps':opt.contrast_bootstraps,
       'direct_contrast':'Within-project/group issue bootstrap on common-project subset; p approximate centered bootstrap',
       'warning':'File clusters within issues handled; recurring same files across issues and project selection remain potential dependence/confounding.'},indent=2)+'\n')
    print('\nRESULTS DIRECTORY:',opt.output.resolve())

if __name__=='__main__':main()
