"""Coverage, cost and fixed-grid envelope from same-machine interleaved replay."""
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
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();base=root/'results/credit_moment_step'
    ts=json.loads((base/'timing/summary.json').read_text());a=json.loads((base/'audit/summary.json').read_text());assert ts['passed'] and a['passed']
    order=['alm_native','alm_residual','adam60_native','adam60_residual','pc_native','nodual_native']
    names=['ALM native','Same-ALM residual','Adam native','Adam residual','Ordinary PC','No dual'];ss={r['method']:r for r in ts['summaries']}
    variants=['fixed','projection','moment'];labels=['Fixed step (222)','Exact root (224)','Analytic moment (226)'];colors=['#a0abba','#d19b4c','#138d87']
    y=np.arange(6);fig,axes=plt.subplots(1,3,figsize=(17.5,6),layout='constrained')
    for i,(key,label,color) in enumerate(zip(variants,labels,colors)):
        yy=y+(i-1)*.24
        vals=np.array([ss[k]['full_128_new'][key] for k in order]);axes[0].barh(yy,vals,.22,color=color,label=label)
        for j,v in enumerate(vals):axes[0].text(v+2,yy[j],str(v),va='center',fontsize=8)
        vals=np.array([ss[k]['mean_seconds'][key]*1000 for k in order]);axes[1].barh(yy,vals,.22,color=color,label=label)
        for j,v in enumerate(vals):axes[1].text(v+8,yy[j],f'{v:.0f}',va='center',fontsize=8)
        vals=np.array([ss[k]['moment_mean_count_per_repeat'] if key=='moment' else ss[k]['prefix_envelope_mean_count_per_repeat'][key] for k in order])
        axes[2].barh(yy,vals,.22,color=color,label=label)
        for j,v in enumerate(vals):axes[2].text(v+2,yy[j],f'{v:g}',va='center',fontsize=8)
    axes[0].set(title='128 responses: 28 / 524 / 453',xlabel='New exact certificates (all 16 tasks)',xlim=(0,250))
    axes[1].set(title='576 same-machine interleaved solves',xlabel='Mean milliseconds / bank (2 repetitions)',xlim=(0,max(r['mean_seconds']['projection'] for r in ss.values())*1250))
    axes[2].set(title='At each moment-run time budget',xlabel='Proof count at affordable fixed-grid prefix',xlim=(0,250))
    for ax in axes:
        ax.set(yticks=y,yticklabels=names);ax.invert_yaxis();ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True);ax.legend(fontsize=8,loc='lower right')
    fig.suptitle('Frozen banks; no LP answers supplied. Right: diagnostic prefix envelope, NOT a hard-deadline online test.\nAcross-learner candidate pools differ; no claim of PC-ALM-specific query benefit.',fontsize=11)
    out=base/'analysis';out.mkdir(parents=True,exist_ok=True);path=out/'credit_moment_step.png';assert not path.exists();fig.savefig(path,dpi=170);plt.close(fig)
    result=dict(passed=True,source_sha256=sha(Path(__file__)),timing_sha256=sha(base/'timing/summary.json'),audit_sha256=sha(base/'audit/summary.json'),figure_sha256=sha(path))
    (out/'figure_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
