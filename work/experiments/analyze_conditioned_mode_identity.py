"""Observation-only mechanism test: identical nonempty pools imply identical readout.

The proof is deterministic: shared halfspace geometry depends only on X,v and
the forward pattern; sorting, geometry, random seed and particle count match.
No claim that different pools must predict differently, or that pool union
across all controls is an equal-cost deployable baseline.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
from run_multiplier_fixed_point_screen import sha,dump

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();src=Path(__file__).parent
    base=root/'results/matched_budget_confirmation';inp=base/'conditioned_confirmation';audit=base/'conditioned_prediction_audit';out=base/'mode_identity';out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    assert json.loads((audit/'summary.json').read_text())['passed'];p=json.loads((inp/'protocol.json').read_text());hashes=dict(json.loads((audit/'protocol.json').read_text())['source_sha256']);hashes[Path(__file__).name]=sha(Path(__file__))
    for n,h in hashes.items():assert sha(src/n)==h,n
    before=json.loads((inp/'before_query_manifest.json').read_text());assert sha(inp/'rows.json')==before['rows_sha256']
    dump(out/'protocol.json',dict(source_sha256=hashes,prediction_audit_sha256=sha(audit/'summary.json'),query_targets_accessed=False,
        mathematical_prediction='same nonempty feasible mode set + same observations + same deterministic geometry + same fixed MC => bytewise same final predictor',
        scope='all prespecified optimizer comparisons; cross-control union is capacity diagnosis, not an equal-cost baseline'))
    rows=json.loads((inp/'rows.json').read_text());lookup={(r['seed'],r['method']):r for r in rows};methods=[c['name'] for c in p['configs'] if lookup[p['seeds'][0],c['name']]['readout']=='mode'];primary=p['primary'];comparisons=[];unionrows=[];counts=Counter()
    def array(row):
        path=inp/row['file'];assert sha(path)==row['sha256']==before['prediction_files'][row['file']]
        with np.load(path) as z:return {k:z[k].copy() for k in ['prediction','point_prediction','points','selected_b']}
    for seed in p['seeds']:
        main=lookup[seed,primary];assert not main['metadata']['execution_failed'];pa=array(main);pool=set(main['metadata']['positive_modes']);control_union=set()
        for name in methods:
            if name==primary:continue
            row=lookup[seed,name];aa=array(row);failed=row['metadata']['execution_failed'];other=set(row['metadata'].get('positive_modes',[]));control_union|=other
            equal=pa['prediction'].tobytes()==aa['prediction'].tobytes();same=pool==other
            if same and pool and not failed:
                assert equal and pa['points'].tobytes()==aa['points'].tobytes(),(seed,name);counts['same_nonempty_pool_predictor_identities']+=1
            comparisons.append(dict(seed=seed,control=name,primary_modes=len(pool),control_modes=len(other),same_pool=same,both_nonempty=bool(pool and other),same_prediction=equal,
                primary_only=sorted(pool-other),control_only=sorted(other-pool),control_failed=failed,
                same_selected_parameters=pa['selected_b'].tobytes()==aa['selected_b'].tobytes(),same_point_prediction=pa['point_prediction'].tobytes()==aa['point_prediction'].tobytes()))
        unionrows.append(dict(seed=seed,primary_modes=len(pool),control_union_modes=len(control_union),primary_only_vs_control_union=sorted(pool-control_union),control_union_only=sorted(control_union-pool)))
    summary_by_control=[]
    for name in methods:
        if name==primary:continue
        rr=[r for r in comparisons if r['control']==name]
        summary_by_control.append(dict(control=name,tasks=len(rr),identical_nonempty_pools=sum(r['same_pool'] and r['both_nonempty'] for r in rr),identical_predictions=sum(r['same_prediction'] for r in rr),
            tasks_with_primary_only_modes=sum(bool(r['primary_only']) for r in rr),tasks_with_control_only_modes=sum(bool(r['control_only']) for r in rr),
            primary_only_modes=sum(len(r['primary_only']) for r in rr),control_only_modes=sum(len(r['control_only']) for r in rr),different_point_but_same_readout=sum(r['same_prediction'] and not r['same_point_prediction'] for r in rr)))
    dump(out/'comparisons.json',comparisons);dump(out/'control_union.json',unionrows);dump(out/'by_control.json',summary_by_control)
    ans=dict(passed=True,counts=counts,optimizer_methods=len(methods),pairwise_comparisons=len(comparisons),tasks_with_primary_exclusive_to_all_controls=sum(bool(r['primary_only_vs_control_union']) for r in unionrows),
        primary_modes_exclusive_to_all_controls=sum(len(r['primary_only_vs_control_union']) for r in unionrows),protocol_sha256=sha(out/'protocol.json'),comparisons_sha256=sha(out/'comparisons.json'),control_union_sha256=sha(out/'control_union.json'),by_control_sha256=sha(out/'by_control.json'),query_targets_accessed=False)
    dump(out/'summary.json',ans);print(json.dumps(ans),flush=True)

if __name__=='__main__':main()
