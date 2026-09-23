"""Layout-only revision: keep legends outside all data marks."""
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
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();base=root/'results/exact_credit_hull'
    s=read(base/'development/summary.json');assert read(base/'audit/summary.json')['passed']
    labels=['ALM native','ALM residual','Adam native','Adam residual','Ordinary PC','No dual','Full history','Uniform history','Recent history','Strong union']
    rows=s['summaries'];y=np.arange(len(rows));fig,axes=plt.subplots(1,2,figsize=(14.4,6.8),layout='constrained');left=np.zeros(len(rows))
    for status,color,label in [('positive','#138c80','Exact positive certificate'),('nonpositive','#adb6c1','Exact common nonpositive witness'),('unknown','#dfa049','Unknown')]:
        values=np.array([r['counts'].get(status,0) for r in rows]);axes[0].barh(y,values,left=left,color=color,label=label)
        for j,num in enumerate(values):
            if num>=25:axes[0].text(left[j]+num/2,j,str(num),ha='center',va='center',fontsize=9,color='white' if status=='positive' else '#243746')
        left+=values
    axes[0].set(yticks=y,yticklabels=labels,xlim=(0,700),xlabel='All 700 common task-region pairs',title='What each fixed credit hull can certify');axes[0].invert_yaxis()
    axes[0].legend(fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.12),ncol=1,frameon=False)
    methods=['adam60_native','pc_native','nodual_native','strong_union','alm_residual'];counts=[]
    for method in methods:counts.append(next(r for r in s['comparisons'] if r['comparator']==method)['status_pairs'].get('positive__nonpositive',0))
    axes[1].bar(np.arange(5),counts,color=['#3b6ba5','#bd863f','#786e8d','#92a0af','#7fa79e'])
    for j,c in enumerate(counts):axes[1].text(j,c+.22,str(c),ha='center',fontsize=12)
    axes[1].set(xticks=np.arange(5),xticklabels=['Adam','PC','No dual','Strong\nunion','Same ALM\nresidual'],ylim=(0,12.8),ylabel='ALM positive AND comparator exactly nonpositive',title='Strict separations, not finite-step misses')
    axes[1].grid(axis='y',alpha=.2);axes[1].text(.03,.93,'16 method-cases = 12 distinct regions in 3 tasks\nZero reverse strict separations in this pool',transform=axes[1].transAxes,va='top',fontsize=9)
    fig.suptitle('7000 offline hull checks: exact dual lower bounds vs exact common primal witnesses\nFixed stored libraries only; full baseline histories and online task benefit remain to be tested.',fontsize=11)
    out=base/'analysis';path=out/'exact_credit_hull_v2.png';assert not path.exists();fig.savefig(path,dpi=175);plt.close(fig)
    result=dict(passed=True,source_sha256=sha(Path(__file__)),main_sha256=sha(base/'development/summary.json'),audit_sha256=sha(base/'audit/summary.json'),
        figure_sha256=sha(path),supersedes_layout_only='exact_credit_hull.png');(out/'figure_v2_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
