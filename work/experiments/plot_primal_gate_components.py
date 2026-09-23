"""Frozen-pool results, including all unsuccessful gate implementations."""
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
    families=['primal_upper_gate','directional_primal_gate','factorized_primal_gate','bounded_error_primal_gate']
    paths=[root/f'results/{f}/component/summary.json' for f in families];all_results=[json.loads(p.read_text()) for p in paths]
    assert all(r['passed'] and r['preserved_proofs']==732 for r in all_results)
    inp=root/'results/bounded_error_primal_gate/component';audit=json.loads((inp/'integer_audit.json').read_text());assert audit['passed']
    ss=all_results[-1]['summaries'];labels=['ALM credit','ALM residual','Adam credit','Adam residual','PC credit','No-dual credit']
    fig,axes=plt.subplots(1,3,figsize=(15,5.2),layout='constrained',gridspec_kw=dict(width_ratios=[1.35,1.1,1]))
    y=np.arange(6);h=.34
    base=np.array([s['baseline_seconds']*1000 for s in ss]);gate=np.array([s['gated_seconds']*1000 for s in ss])
    axes[0].barh(y-h/2,base,h,color='#8b99aa',label='Full credit screen')
    axes[0].barh(y+h/2,gate,h,color='#2b9e91',label='Upper gate + remaining screen')
    axes[0].set(yticks=y,yticklabels=labels,xlabel='Cold bank + screening time (ms)',title='All gate costs included');axes[0].invert_yaxis()
    for i,value in enumerate(gate):axes[0].text(value+.12,i+h/2,f'{value:.2f}',va='center',fontsize=8)
    axes[0].set_xlim(0,max(gate)*1.22);axes[0].legend(loc='lower right',fontsize=8)
    rates=np.array([100*s['skipped']/s['regions'] for s in ss]);axes[1].barh(y,rates,color='#2b9e91')
    axes[1].set(yticks=y,yticklabels=labels,xlim=(0,112),xlabel='Regions bypassing credit computation (%)',title='Geometry still sees every skipped region');axes[1].invert_yaxis()
    for i,rate in enumerate(rates):axes[1].text(rate+1.5,i,f'{ss[i]["skipped"]}/{ss[i]["regions"]}',va='center',fontsize=8)
    ratios=[]
    for r in all_results:
        s=next(s for s in r['summaries'] if s['method']=='alm_native');ratios.append(s['gated_seconds']/s['baseline_seconds'])
    axes[2].barh(range(4),ratios,color=['#b2b8c0','#b2b8c0','#b2b8c0','#2b9e91'])
    axes[2].set(yticks=range(4),yticklabels=['3 shared witnesses','Directional intervals','Layer-factorized intervals','Bounded-error upper'],
        xscale='log',xlabel='Gated / same-run ungated ALM time',title='Preserved implementation history');axes[2].invert_yaxis()
    axes[2].axvline(1,color='#444',ls='--',lw=1)
    for i,ratio in enumerate(ratios):axes[2].text(ratio*1.08,i,f'{ratio:.3f}x',va='center',fontsize=8)
    axes[2].set_xlim(.65,max(ratios)*1.7)
    for ax in axes:ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True)
    fig.suptitle('96 frozen pools | 2,437 / 3,998 regions skipped | 732 proofs preserved | component result only',fontsize=12)
    path=inp/'primal_gate_components.png';fig.savefig(path,dpi=180);plt.close(fig)
    result=dict(passed=True,source_sha256=sha(Path(__file__)),input_sha256={str(p.relative_to(root)):sha(p) for p in paths+[inp/'integer_audit.json']},figure_sha256=sha(path))
    (inp/'figure_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
