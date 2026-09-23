"""All methods/budgets plus paired task-level uncertainty, no cherry-pick."""
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
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();base=root/'results/credit_wall_budget'
    s=json.loads((base/'development/summary.json').read_text());assert json.loads((base/'audit/summary.json').read_text())['passed']
    names={'alm_native':'ALM native','alm_residual':'ALM residual','adam60_native':'Adam native','adam60_residual':'Adam residual','pc_native':'Ordinary PC',
        'nodual_native':'No dual','history_full':'Full history','history_uniform12':'Uniform history','history_recent12':'Recent history'}
    colors={'alm_native':'#128880','alm_residual':'#69afa8','adam60_native':'#395f9e','adam60_residual':'#96abc8','pc_native':'#c78333',
        'nodual_native':'#76616a','history_full':'#8a4f9c','history_uniform12':'#c36b8c','history_recent12':'#677e42'}
    blocks=s['summaries'];x=np.array([r['budget_seconds']*1000 for r in blocks]);fig,axes=plt.subplots(1,2,figsize=(13.8,6),layout='constrained')
    for method,label in names.items():
        yy=[next(r for r in block['summaries'] if r['method']==method)['count_mean_over_repeats'] for block in blocks]
        axes[0].plot(x,yy,marker='o',ms=4,lw=2.5 if method=='alm_native' else 1.3,ls='--' if method.startswith('history') else '-',color=colors[method],label=label)
    axes[0].set(xscale='log',xticks=x,xticklabels=[str(int(v)) for v in x],xlabel='Cooperative budget (ms); atomic overruns reported separately',
        ylabel='Exact exclusions completed by cutoff (two-repeat mean)',title='All nine credit libraries, same 700 task-region pairs')
    axes[0].legend(fontsize=8,ncol=2,loc='lower right');axes[0].grid(alpha=.2)
    comparators=['history_full','history_uniform12','adam60_native','pc_native'];offsets=np.arange(4)*.18-.27
    for j,method in enumerate(comparators):
        vals=[next(r for r in b['comparisons'] if r['comparator']==method) for b in blocks]
        yy=np.array([v['native_minus_comparator_mean_per_task'] for v in vals]);ci=np.array([v['paired_bootstrap95'] for v in vals])
        axes[1].errorbar(np.arange(4)+offsets[j],yy,yerr=np.array([yy-ci[:,0],ci[:,1]-yy]),fmt='o',ms=4,capsize=3,color=colors[method],label='vs '+names[method])
    axes[1].axhline(0,color='#5f6d79',ls='--',lw=1);axes[1].set(xticks=np.arange(4),xticklabels=[str(int(v)) for v in x],
        xlabel='Budget (ms)',ylabel='ALM minus comparator: certificates / task',title='Paired 95% bootstrap intervals across 16 tasks')
    axes[1].grid(axis='y',alpha=.2);axes[1].legend(fontsize=8,loc='best')
    fig.suptitle('1152 timed solves; uniform starts, no expanded solution weights. Late certificates are excluded.\nComponent evidence only: actual overrun/guard costs remain charged; no independent query-superiority claim.',fontsize=10.5)
    out=base/'analysis';out.mkdir(parents=True,exist_ok=True);path=out/'credit_wall_budget.png';assert not path.exists();fig.savefig(path,dpi=175);plt.close(fig)
    result=dict(passed=True,source_sha256=sha(Path(__file__)),main_sha256=sha(base/'development/summary.json'),audit_sha256=sha(base/'audit/summary.json'),figure_sha256=sha(path))
    (out/'figure_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
