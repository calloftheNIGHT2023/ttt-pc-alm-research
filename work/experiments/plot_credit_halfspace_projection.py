"""Evidence gain and charged root-update cost, not an end-to-end claim."""
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
    source=root/'results/credit_halfspace_projection';s=json.loads((source/'development/summary.json').read_text());a=json.loads((source/'audit/summary.json').read_text());assert a['passed']
    old=root/'results/finite_credit_game/development/summary.json';prior=json.loads(old.read_text())
    order=['alm_native','alm_residual','adam60_native','adam60_residual','pc_native','nodual_native']
    names=['ALM native','Same-ALM residual','Adam native','Adam residual','Ordinary PC','No dual']
    ss={r['method']:r for r in s['summaries']};bb={r['method']:r for r in prior['summaries']}
    y=np.arange(6);fig,axes=plt.subplots(1,2,figsize=(12.8,5.4),layout='constrained')
    previous=np.array([bb[k]['new_proofs'] for k in order]);current=np.array([ss[k]['new_proofs'] for k in order])
    axes[0].barh(y-.17,previous,.32,color='#9ba8b8',label='Fixed-step exponential weights (222)')
    axes[0].barh(y+.17,current,.32,color='#168d87',label='KL halfspace projection (224)')
    for i,(b,c) in enumerate(zip(previous,current)):
        axes[0].text(b+2,i-.17,str(b),va='center',fontsize=9);axes[0].text(c+2,i+.17,str(c),va='center',fontsize=9)
    axes[0].set(yticks=y,yticklabels=names,xlim=(0,245),xlabel='New exact certificates beyond old retained proofs',title=f'28 → 524 new proofs; {a["checks"]["strict_synergy"]} strict synergies')
    axes[0].invert_yaxis();axes[0].legend(fontsize=8,loc='lower right')
    keys=['mean_oracle_seconds','mean_update_seconds','mean_exact_seconds'];labels=['Local best responses','Credit / scalar-root updates','Exact certificate checks'];colors=['#236a9b','#d99445','#168d87']
    bottom=np.zeros(6)
    for key,label,color in zip(keys,labels,colors):
        values=np.array([ss[k][key] for k in order]);axes[1].barh(y,values,left=bottom,color=color,label=label);bottom+=values
    total=np.array([ss[k]['mean_total_seconds'] for k in order]);axes[1].barh(y,total-bottom,left=bottom,color='#d3d9e0',label='Other charged loop work')
    for i,value in enumerate(total):axes[1].text(value+.018,i,f'{value:.3f}s',va='center',fontsize=9)
    axes[1].set(yticks=y,yticklabels=names,xlim=(0,1.95),xlabel='Seconds / task (new solver only, old screening separate)',title='Cost is dominated by repeated scalar-root updates')
    axes[1].invert_yaxis();axes[1].legend(fontsize=8,loc='lower right')
    for ax in axes:ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('Same 96 development banks and 128-response cap | No LP answers supplied | Not independent task superiority',fontsize=10.5)
    out=source/'analysis';out.mkdir(parents=True,exist_ok=True);path=out/'credit_halfspace_projection.png';assert not path.exists()
    fig.savefig(path,dpi=180);plt.close(fig)
    result=dict(passed=True,source_sha256=sha(Path(__file__)),main_sha256=sha(source/'development/summary.json'),
        audit_sha256=sha(source/'audit/summary.json'),baseline_sha256=sha(old),figure_sha256=sha(path))
    (out/'figure_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
