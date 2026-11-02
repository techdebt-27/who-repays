#!/usr/bin/env python3
"""End-to-end LOCAL synthetic Git/CSV integration test; no network required."""
import tempfile,subprocess,json,sys
from pathlib import Path
import pandas as pd, numpy as np
ROOT=Path(__file__).parent

def git(repo,*args):
    r=subprocess.run(['git','-C',str(repo),*args],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,check=True)
    return r.stdout.strip()

def main():
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp);final=root/'FINAL';ck=root/'CK-METRICS';repos=root/'repos'; out=root/'output'
        for p in [final,ck,repos,out]:p.mkdir()
        source=[]; ckrows=[]; rng=np.random.default_rng(77)
        for project,repo_name in [('KAFKA','kafka'),('CAMEL','camel')]:
            repo=repos/repo_name;repo.mkdir()
            git(repo,'init','-q');git(repo,'config','user.email','test@example.org');git(repo,'config','user.name','test')
            (repo/'src').mkdir()
            for j in range(3):(repo/f'src/C{j}.java').write_text(f'class C{j} {{ int a; }}\n')
            git(repo,'add','.');git(repo,'commit','-qm','intro')
            commits=[git(repo,'rev-parse','HEAD')]
            for i in range(1,15):
                j=i%3
                with open(repo/f'src/C{j}.java','a') as o:o.write(f'// commit {i}\n')
                git(repo,'add','.');git(repo,'commit','-qm',f'change {i}')
                commits.append(git(repo,'rev-parse','HEAD'))
            payment=commits[-1]
            for i in range(12):
                sf=i%2==0; key=f'{project}-{900+i}';intro=commits[i]
                isec=[];psec=[];a_ck=[];b_ck=[]
                for j in range(3):
                    path=f'src/C{j}.java';base=2+i+j;delta=float(rng.normal(i,3))
                    isec.append({'file_path':path,'file_fan_in':base,'file_fan_out':base+1,'file_cyclomatic_complexity_sum':base+2,'file_cognitive_complexity_sum':base+3})
                    psec.append({'file_path':path,'file_fan_in':base+delta,'file_fan_out':base+1+delta,'file_cyclomatic_complexity_sum':base+2+delta,'file_cognitive_complexity_sum':base+3+delta})
                    a_ck.append({'file_path':path,'file_cbo_mean':base+4,'file_rfc_understand_mean':base+5,'file_lcom_mean':base+6,'file_dit_mean':base%3,'file_noc_mean':base%5})
                    b_ck.append({'file_path':path,'file_cbo_mean':base+4+delta,'file_rfc_understand_mean':base+5+delta,'file_lcom_mean':base+6+delta,'file_dit_mean':base%3+delta*.1,'file_noc_mean':base%5+delta*.1})
                source.append({'Key':key,'Intro Hash':intro,'Payment Hash':payment,'indicator':'viomod',
                              'Intro_Affected_Files':';'.join(f'src/C{j}.java' for j in range(3)),
                              'Is Self-Fixed (Intro=Payment)':str(sf),'GitHub Base URL':f'https://github.com/apache/{repo_name}',
                              'Intro_FileDetails_JSON':json.dumps(isec),'Payment_FileDetails_JSON':json.dumps(psec)})
                ckrows.append({'Key':key,'Intro Hash':intro,'Payment Hash':payment,
                               'Intro_CK_FileSummary_JSON':json.dumps(a_ck),'Payment_CK_FileSummary_JSON':json.dumps(b_ck),
                               'Intro_CK_Status':'success','Payment_CK_Status':'success','Payment_Path_Map_JSON':'{}'})
        pd.DataFrame(source).to_csv(final/'TEST-UNDERSTAND-METRICS-CC-COMPLEXITY-FINAL.csv',index=False)
        pd.DataFrame(ckrows).to_csv(ck/'TEST-UNDERSTAND-CK-METRICS.csv',index=False)
        cmd=[sys.executable,str(ROOT/'run_full_rq2.py'),'--mode','full','--final-dir',str(final),'--ck-dir',str(ck),
            '--repos-root',str(repos),'--outdir',str(out),'--permutations','199','--issue-bootstraps','199',
            '--file-bootstraps','199','--contrast-bootstraps','199']
        print('Synthetic end-to-end extraction + analysis...',flush=True)
        proc=subprocess.run(cmd,text=True,timeout=65)
        if proc.returncode:
            raise RuntimeError('E2E failed')
        required=['extraction/rq3_file_level_nine_metrics.csv','RQ3_PRIMARY_issue_spearman_project_permutation.csv',
          'RQ3_SENSITIVITY_file_spearman_issue_cluster_bootstrap.csv',
          'RQ2_DIRECT_SF_NSF_correlation_difference.csv','Figure3_RQ2_file_commits_vs_nine_metric_deltas.pdf']
        for name in required: assert (out/name).is_file(),name
        result=pd.read_csv(out/required[0]);assert len(result)==72
        print('PASS: full synthetic Git -> FINAL+CK pairing -> CSV -> Table IV+V -> SF/NSF contrast -> Figure 3')

if __name__=='__main__':main()
