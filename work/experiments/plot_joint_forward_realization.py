"""Actual joint-forward credit coverage and all eight previous strict cases."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();base=root/'results/joint_forward_realization'
    a=read(base/'audit/summary.json');assert a['passed'];s=read(base/'development/summary.json');old=read(root/'results/baseline_history/development/summary.json')
    cases=read(base/'analysis/old_cases.json');assert len(cases)==8
    rows=[dict(positive=539,nonpositive=161),next(r for r in old['summaries'] if r['method']=='adam60_history_full')['effective_counts'],*[r['counts'] for r in s['summaries']]]
    labels=['ALM\nreference','Executed Adam\ncomplete history','Constructed\njoint probes','Adam history\n+ joint probes']
    fig,axes=plt.subplots(1,2,figsize=(12.8,6),layout='constrained');bottom=np.zeros(4);x=np.arange(4)
    for status,color,title in [('positive','#148a7c','Exact positive'),('nonpositive','#bdc7d3','Exact nonpositive'),('unknown','#dc9b43','Unknown')]:
        values=np.array([r.get(status,0) for r in rows]);axes[0].bar(x,values,bottom=bottom,color=color,label=title)
        for i,num in enumerate(values):
            if num>=8:axes[0].text(i,bottom[i]+num/2,str(num),ha='center',va='center',color='white' if status=='positive' else '#233448',fontsize=10)
        bottom+=values
    axes[0].set(xticks=x,xticklabels=labels,ylim=(0,725),ylabel='Common task-regions (700 total)',title='Complete comparison: 16 old tasks');axes[0].tick_params(axis='x',labelsize=9)
    axes[0].legend(loc='upper center',bbox_to_anchor=(.5,-.15),ncol=3,frameon=False,fontsize=8)
    x=np.arange(8)
    for offset,method,color,label in [(-.18,'constructed_joint','#327db6','New probes alone'),(.18,'old_plus_constructed_joint','#148a7c','Old history + new')]:
        values=[r['methods'][method]['exact_lower_float'] for r in cases]
        assert all(v>0 for v in values);axes[1].bar(x+offset,values,width=.34,color=color,label=label)
    axes[1].axhline(0,color='#233448',lw=.8);axes[1].set(xticks=x,xticklabels=[r['index'] for r in cases],xlabel='Region index (all eight from task 5920012)',ylabel='Exact positive lower bound (decimal display)',title='All eight now have realizable positive certificates',ylim=(0,.033))
    axes[1].legend(loc='upper center',bbox_to_anchor=(.5,-.15),ncol=2,frameon=False,fontsize=8)
    fig.suptitle('Shared-parameter realization succeeds: actual joint forwards, not per-observation fictitious credits\nCredit-capacity diagnostic only; no new query-risk or online speed claim',fontsize=11)
    out=base/'analysis';path=out/'joint_forward_realization.png';assert not path.exists();fig.savefig(path,dpi=175);plt.close(fig)
    result=dict(passed=True,source_sha256=sha(Path(__file__)),main_sha256=sha(base/'development/summary.json'),audit_sha256=sha(base/'audit/summary.json'),cases_sha256=sha(out/'old_cases.json'),figure_sha256=sha(path))
    (out/'figure_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)

if __name__=='__main__':main()
