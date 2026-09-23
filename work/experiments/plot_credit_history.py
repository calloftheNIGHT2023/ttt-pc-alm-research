"""Separates positive-ray information from finite optimizer representation."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();base=root/'results/credit_history'
    s=json.loads((base/'development/summary.json').read_text());a=json.loads((base/'audit/summary.json').read_text());assert a['passed']
    ex=json.loads((base/'audit/expansions.json').read_text());ss=s['summaries'];names=['ALM native (768)','Full history (1024)','Uniform history (768)','Recent history (768)'];y=np.arange(4)
    fig,axes=plt.subplots(1,2,figsize=(13.1,5.4),layout='constrained');old=np.array([r['old_count'] for r in ss]);new=np.array([r['positive'] for r in ss])
    axes[0].barh(y,old,color='#8b9aa9',label='Individual-credit certificates');axes[0].barh(y,new,left=old,color='#158c87',label='New joint-credit certificates')
    for i,v in enumerate(old+new):axes[0].text(v+3,i,str(v),va='center')
    axes[0].set(yticks=y,yticklabels=names,xlim=(0,330),xlabel='Exact exclusions / same 700 task-region pairs',title='Uniform initialization, same 128-response cap')
    axes[0].invert_yaxis();axes[0].grid(axis='x',alpha=.2);axes[0].set_axisbelow(True);axes[0].legend(fontsize=8,loc='lower right')
    for kind,label,color in [('individual','Native individual proof','#346e9d'),('joint','Native joint proof','#ce913e')]:
        rows=[r for r in ex if r['kind']==kind];xx=[r['original_value'] for r in rows];yy=[r['full_history_value'] for r in rows]
        axes[1].scatter(xx,yy,s=13,alpha=.65,label=label,color=color)
    lo=min(min(r['original_value'],r['full_history_value']) for r in ex)*.6;hi=max(max(r['original_value'],r['full_history_value']) for r in ex)*1.8
    axes[1].plot([lo,hi],[lo,hi],color='#7f8995',ls=':',label='Equal margin');axes[1].axhline(.001001,color='#c6565c',ls='--',lw=1,label='Fixed target delta')
    axes[1].set(xscale='log',yscale='log',xlim=(lo,hi),ylim=(lo,hi),xlabel='Original exact D(a)',ylabel='Constructed normalized history D(a / Z)',title=f'All {len(ex)} native proofs expand to positive history mixtures')
    axes[1].grid(alpha=.2);axes[1].legend(fontsize=8)
    fig.suptitle('Read-only observer preserves all trajectories. Expanded weights used ONLY for offline attribution.\nInformation is available in full history; optimizer path, normalization and finite resources still differ.',fontsize=10.5)
    out=base/'analysis';out.mkdir(parents=True,exist_ok=True);path=out/'credit_history.png';assert not path.exists();fig.savefig(path,dpi=175);plt.close(fig)
    result=dict(passed=True,source_sha256=sha(Path(__file__)),main_sha256=sha(base/'development/summary.json'),audit_sha256=sha(base/'audit/summary.json'),
        expansions_sha256=sha(base/'audit/expansions.json'),figure_sha256=sha(path))
    (out/'figure_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
