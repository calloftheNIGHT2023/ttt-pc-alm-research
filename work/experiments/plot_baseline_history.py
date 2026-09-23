"""Unchanged trajectories: checkpoint capacity versus complete-history capacity."""
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
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();base=root/'results/baseline_history'
    assert read(base/'audit/summary.json')['passed'];s=read(base/'development/summary.json');old=read(root/'results/exact_credit_hull/development/summary.json');cases=read(base/'analysis/strict_cases.json')
    methods=['adam60','pc','nodual'];labels=['Adam','Ordinary PC','No dual'];fig,axes=plt.subplots(1,2,figsize=(13.3,6.5),layout='constrained');x=np.arange(3)
    before=[next(r for r in old['summaries'] if r['method']==m+'_native')['counts'].get('positive',0) for m in methods]
    after=[next(r for r in s['summaries'] if r['method']==m+'_history_full')['effective_counts'].get('positive',0) for m in methods]
    for values,offset,color,label in [(before,-.18,'#8ea0b7','Original checkpoints'),(after,.18,'#168d80','Complete executed history')]:
        axes[0].bar(x+offset,values,width=.35,color=color,label=label)
        for i,n in enumerate(values):axes[0].text(i+offset,n+4,str(n),ha='center',fontsize=10)
    axes[0].axhline(539,color='#cc8538',lw=1.5,ls='--',label='ALM reference: 539');axes[0].set(xticks=x,xticklabels=labels,ylim=(0,700),ylabel='Exactly certified exclusions / 700 regions',title='No extra update steps or observations')
    axes[0].legend(loc='upper center',bbox_to_anchor=(.5,-.1),fontsize=8,frameon=False);axes[0].grid(axis='y',alpha=.2)
    bottom=np.zeros(3)
    for status,color,label in [('positive','#3f72a6','Baseline now has an exact positive certificate'),('nonpositive','#168d80','ALM strict separation persists'),('unknown','#dfa049','Unknown')]:
        vals=np.array([sum(r['comparator']==m+'_native' and r['full_status']==status for r in cases) for m in methods]);axes[1].bar(x,vals,bottom=bottom,color=color,label=label)
        for i,num in enumerate(vals):
            if num:axes[1].text(i,bottom[i]+num/2,str(num),ha='center',va='center',color='white',fontsize=12)
        bottom+=vals
    axes[1].set(xticks=x,xticklabels=labels,ylim=(0,12),ylabel='The 16 previous strict method-cases',title='All prior cases retained, including recoveries')
    axes[1].legend(loc='upper center',bbox_to_anchor=(.5,-.1),fontsize=8,frameon=False);axes[1].grid(axis='y',alpha=.2)
    fig.suptitle('Full-history capacity diagnostic: 16 tasks, 700 common regions, four expanded libraries\nExact positive / exact nonpositive / unknown remain separate. These are not online query results.',fontsize=11)
    out=base/'analysis';path=out/'baseline_history.png';assert not path.exists();fig.savefig(path,dpi=175);plt.close(fig)
    audit=dict(passed=True,source_sha256=sha(Path(__file__)),main_sha256=sha(base/'development/summary.json'),audit_sha256=sha(base/'audit/summary.json'),
        cases_sha256=sha(out/'strict_cases.json'),figure_sha256=sha(path));(out/'figure_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8');print(json.dumps(audit),flush=True)


if __name__=='__main__':main()
