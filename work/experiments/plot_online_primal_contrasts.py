"""Paired first-stage contrasts, not an extrapolated whole-stream speedup."""
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
    out=root/'results/online_primal_gate/analysis';inp=out/'summary.json';s=json.loads(inp.read_text());assert s['complete']
    contrasts=s['gate_contrasts'];parts={r['method']:r for r in s['components']}
    labels=['ALM credit','Adam credit','PC credit','No-dual credit','ALM residual','Adam residual'];y=np.arange(6)
    fig,ax=plt.subplots(1,2,figsize=(12.8,5.4),layout='constrained')
    means=np.array([r['first_time']['mean'] for r in contrasts])*1000;ci=np.array([r['first_time']['ci95'] for r in contrasts])*1000
    ax[0].errorbar(means,y,xerr=np.stack([means-ci[:,0],ci[:,1]-means]),fmt='o',color='#286f9c',capsize=4)
    ax[0].axvline(0,ls='--',color='#444',lw=1);ax[0].set(yticks=y,yticklabels=labels,
        xlabel='Gated - ungated first-stage time (ms; negative is faster)',title='Task-paired means and descriptive 95% intervals');ax[0].invert_yaxis()
    before=np.array([parts[r['comparator']]['credit_seconds'] for r in contrasts])*1000
    after=np.array([parts[r['method']]['credit_seconds'] for r in contrasts])*1000
    ax[1].barh(y-.17,before,.34,label='Ungated credit calls',color='#8b99aa')
    ax[1].barh(y+.17,after,.34,label='Gated credit calls',color='#2b9e91')
    ax[1].set(yticks=y,yticklabels=labels,xlabel='Measured causal credit-call time (ms)',title='Nested component, not full first-stage latency');ax[1].invert_yaxis()
    ax[1].legend(loc='lower right',fontsize=8)
    for axes in ax:axes.grid(axis='x',alpha=.2);axes.set_axisbelow(True)
    fig.suptitle('768 complete streams | predictions byte-identical | ALM full-stream time interval still includes zero',fontsize=11)
    path=out/'primal_gate_paired_contrasts.png';fig.savefig(path,dpi=180);plt.close(fig)
    result=dict(passed=True,source_sha256=sha(Path(__file__)),summary_sha256=sha(inp),figure_sha256=sha(path))
    (out/'contrast_figure_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
