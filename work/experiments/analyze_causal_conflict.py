"""Causal-bank independent coverage analysis, without tuning any runtime."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);a=p.parse_args();root=a.project/'results/causal_conflict/diagnostic';summary=json.loads((root/'summary.json').read_text());chron=json.loads((root/'chronology.json').read_text());lookup={(r['seed'],r['method']):r for r in chron if r['gate_c20']};rows=[]
    seeds=sorted({r['seed'] for r in chron});labels={};first=[];remaining=[];cards=[]
    for seed in seeds:
        local=lookup[seed,'alm_local_k24'];bp=lookup[seed,'alm_bp_c20_k24'];res=lookup[seed,'alm_residual_k24']
        ll={h['pattern'] for h in local['hits'] if h['beyond_c20_clause_visits']};bb={h['pattern'] for h in bp['hits'] if h['beyond_c20_full_visits']};rr={h['pattern'] for h in res['hits'] if h['beyond_c20_full_visits']};unique=ll-bb-rr
        rows.append(dict(seed=seed,local_unique_beyond_bp_and_residual_full=len(unique),patterns=sorted(unique)))
        for proof in local['proofs']:
            labels[proof['label']]=labels.get(proof['label'],0)+1;first.append(proof['accepted_event']);remaining.append(len(local['events'])-proof['accepted_event']);cards.append(proof['cardinality'])
    result=dict(rows=rows,local_unique_patterns=sum(r['local_unique_beyond_bp_and_residual_full'] for r in rows),tasks_with_unique=sum(r['local_unique_beyond_bp_and_residual_full']>0 for r in rows),
        accepted_labels=labels,accepted_event_range=[min(first),max(first)],remaining_callbacks_range=[min(remaining),max(remaining)],mean_local_clause_cardinality=float(np.mean(cards)),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),input_sha256=hashlib.sha256((root/'chronology.json').read_bytes()).hexdigest(),
        scope='unique local future clause hits excluding all BP/residual future full-bound hits; no claim of universal credit uniqueness or saved steps')
    (root/'independent_coverage.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    selected=[r for r in summary['summary'] if r['gate_c20']];labels=['ALM local','ALM residual','Same ALM / BP','No dual','PC80','Adam16 / BP','Adam60 / BP'];y=np.arange(len(selected));fig,axes=plt.subplots(1,2,figsize=(11,5.2),sharey=True)
    for ax,key,title in zip(axes,['clause_beyond_c20','unique_clause_beyond_c20'],['Future visits detected beyond C20','Distinct modes detected beyond C20']):
        vals=[r[key] for r in selected];ax.barh(y,vals,color=['#177E89' if i==0 else '#748299' for i in y]);ax.set_title(title,fontsize=11);ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
    axes[0].set_yticks(y,labels);axes[0].invert_yaxis();fig.suptitle('Read-only chronological audit: unchanged original trajectories',fontsize=13);fig.text(.5,.025,'Different solvers have different step counts. Repeated hits are not saved steps or independent tasks.',ha='center',fontsize=9);fig.tight_layout(rect=[0,.05,1,.95]);fig.savefig(root/'causal_conflict_results.png',dpi=170);plt.close(fig);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
