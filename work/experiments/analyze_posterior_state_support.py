"""Audit and summarize frozen support-only proposal diagnostics; no query data."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    protocol=json.loads((a.input/'protocol.json').read_text());rows=json.loads((a.input/'diagnostics.json').read_text());captures=json.loads((a.input/'captures.json').read_text())
    assert len(rows)==432 and len(captures)==192
    for name,h in protocol['source_sha256'].items():assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()==h,name
    for c in captures:
        assert c['original_anchor_and_geometry_exact']
        assert hashlib.sha256((a.input/c['state_file']).read_bytes()).hexdigest()==c['state_sha256']
        with np.load(a.input/c['state_file']) as z:
            assert len(z['posterior'])==c['posterior_samples']
            assert z['posterior'].nbytes+z['anchor'].nbytes==c['numeric_persistent_state_bytes']
    names=[c['name'] for c in protocol['configs']];pools=protocol['pools'];summary=[];paired=[]
    for method in names:
        for n in [8,16,24]:
            for pool in pools:
                group=[r for r in rows if (r['method'],r['n_context'],r['pool'])==(method,n,pool)]
                assert len(group)==16
                summary.append(dict(method=method,n_context=n,pool=pool,
                    positive_tasks=sum(r['positive_regions']>0 for r in group),
                    mean_positive_regions=float(np.mean([r['positive_regions'] for r in group])),
                    mean_prior_mass=float(np.mean([r['discovered_prior_mass'] for r in group])),
                    mean_best_pool_mse=float(np.mean([r['best_pool_mse'] for r in group])),
                    median_best_pool_max_error=float(np.median([r['best_pool_max_error'] for r in group])),
                    mean_selected_distinct_regions=float(np.mean([r['selected_distinct_regions'] for r in group]))))
            for other in ['prior256','prior768']:
                pairs=[]
                for seed in protocol['seeds']:
                    pair={r['pool']:r for r in rows if (r['method'],r['n_context'],r['seed'])==(method,n,seed)}
                    left=pair['posterior_mix'];right=pair[other]
                    lm={r['pattern']:r['volume'] for r in left['positive_geometry']};rm={r['pattern']:r['volume'] for r in right['positive_geometry']}
                    assert all(lm[k]==rm[k] for k in lm.keys()&rm.keys())
                    union={**lm,**rm};total=sum(union.values())
                    pairs.append(dict(seed=seed,mass_difference=left['discovered_prior_mass']-right['discovered_prior_mass'],
                        posterior_union_fraction=sum(lm.values())/total if total else None,
                        prior_union_fraction=sum(rm.values())/total if total else None,
                        posterior_regions=len(lm),prior_regions=len(rm),
                        best_pool_mse_difference=left['best_pool_mse']-right['best_pool_mse']))
                valid=[r for r in pairs if r['posterior_union_fraction'] is not None]
                paired.append(dict(method=method,n_context=n,comparison='posterior_mix minus '+other,
                    positive_mass_better=sum(r['mass_difference']>0 for r in pairs),
                    positive_mass_equal=sum(r['mass_difference']==0 for r in pairs),
                    positive_mass_worse=sum(r['mass_difference']<0 for r in pairs),
                    mean_posterior_fraction_of_pair_union=float(np.mean([r['posterior_union_fraction'] for r in valid])) if valid else None,
                    mean_prior_fraction_of_pair_union=float(np.mean([r['prior_union_fraction'] for r in valid])) if valid else None,
                    pair_union_nonempty_tasks=len(valid),pairs=pairs))
    output=dict(audit=dict(source_hashes=len(protocol['source_sha256']),state_hashes=len(captures),exact_historical_captures=True,
        posterior_sample_counts=sorted(set(c['posterior_samples'] for c in captures)),diagnostics=len(rows)),summary=summary,paired=paired,
        scope='Support only, own frozen historical trajectories, reused geometry cost not an online timing comparison; pair union is not full posterior coverage.')
    a.out.mkdir(parents=True,exist_ok=True);(a.out/'summary.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    fig,axes=plt.subplots(1,3,figsize=(13,4),sharey=True)
    colors=['#9aa6b2','#d38a37','#167b79']
    for ax,method in zip(axes,names):
        for k,(pool,color) in enumerate(zip(pools,colors)):
            vals=[next(r for r in summary if (r['method'],r['n_context'],r['pool'])==(method,n,pool))['positive_tasks'] for n in [8,16,24]]
            ax.bar(np.arange(3)+(k-1)*.24,vals,width=.23,label=pool,color=color)
        ax.set_title(method);ax.set_xticks(np.arange(3),['8','16','24']);ax.set_xlabel('Observed support size');ax.set_ylim(0,17);ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Tasks with a positive-volume feasible cell / 16');axes[-1].legend(fontsize=8)
    fig.suptitle('Previous posterior samples as proposals — frozen-history support diagnostic')
    fig.tight_layout();fig.savefig(a.out/'support_state_reuse.png',dpi=160);plt.close(fig)
    print(json.dumps({**output['audit'],'summary':summary,'paired_summary':[{k:v for k,v in r.items() if k!='pairs'} for r in paired]},indent=2))


if __name__=='__main__':main()
