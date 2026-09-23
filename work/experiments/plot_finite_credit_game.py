"""Static scientific comparison of the frozen finite run and its ceiling."""
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
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve()
    source=root/'results/finite_credit_game';s=json.loads((source/'development/summary.json').read_text());a=json.loads((source/'audit/summary.json').read_text());assert a['passed']
    order=['alm_native','alm_residual','adam60_native','adam60_residual','pc_native','nodual_native']
    names=['ALM native','Same-ALM residual','Adam native','Adam residual','Ordinary PC','No dual']
    ss={r['method']:r for r in s['summaries']};aa={r['method']:r for r in a['summaries']}
    y=np.arange(6);fig,axes=plt.subplots(1,2,figsize=(12.2,5.1),layout='constrained')
    ceiling=np.array([aa[k]['joint_new'] for k in order]);finite=np.array([aa[k]['finite_new'] for k in order])
    axes[0].barh(y,ceiling,color='#e2e6ec',label='Full-hull LP ceiling (post-run comparison)')
    axes[0].barh(y,finite,color='#168d87',label='128-step local game, exact accepted')
    for i in range(6):axes[0].text(ceiling[i]+5,i,f'{finite[i]} / {ceiling[i]}',va='center',fontsize=9)
    axes[0].set(yticks=y,yticklabels=names,xlim=(0,555),xlabel='Additional certificates, beyond retained old proofs',title='28 exact new certificates; 22 strict synergies')
    axes[0].invert_yaxis();axes[0].legend(fontsize=8,loc='lower right')
    prefixes=[1,4,16,64,128]
    for method,label,color in [('alm_native','ALM native','#236a9b'),('alm_residual','Same-ALM residual','#d78330')]:
        axes[1].plot(prefixes,[ss[method]['prefix_proofs'][str(k)] for k in prefixes],marker='o',label=label,color=color)
    axes[1].axhline(0,color='#9aa3ad',linestyle=':',label='Four other libraries: zero throughout')
    axes[1].set(xscale='log',xticks=prefixes,xticklabels=[str(k) for k in prefixes],ylim=(-1,23),xlabel='Fixed prefixes of one path',ylabel='Cumulative new exact certificates',title='No step-size or prefix re-selection')
    axes[1].legend(fontsize=8,loc='upper left')
    for ax in axes:ax.grid(alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('96 old development libraries | 3,266 unproved regions | No LP/BP inside the finite solver',fontsize=11)
    out=source/'analysis';out.mkdir(parents=True,exist_ok=True);path=out/'finite_credit_game.png';assert not path.exists()
    fig.savefig(path,dpi=180);plt.close(fig)
    result=dict(passed=True,source_sha256=sha(Path(__file__)),main_sha256=sha(source/'development/summary.json'),
        audit_sha256=sha(source/'audit/summary.json'),figure_sha256=sha(path))
    (out/'figure_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
