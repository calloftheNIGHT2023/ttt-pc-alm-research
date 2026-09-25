"""357 exact support-language, certificate, causal-source and inclusion audit."""
from collections import Counter
from fractions import Fraction as F
from itertools import product
from pathlib import Path
import time
import numpy as np
import budget_reinvestment_suite_v1 as io
import branch_image_chain_v1 as reference
import online_credit_branch_search_v1 as geometry
from test_support_language_chain_v1 import possible
from certified_branch_solver import exact_certificate
from run_cross_region_credit_v1 import CHANNELS,pool
from audit_stasis_escape_geometry_v1 import inequalities
from audit_post_escape_geometry_v1 import certificates
from posterior_confirmation_pipeline import discovery_box


def run(root,out):
    begin=time.perf_counter();source=root/'results/cross_region_credit/development_v1';s=io.complete(source)
    protocol=io.read(source/'protocol.json');configs=protocol['configs'];counts=Counter();totals={c['name']:Counter() for c in configs}
    rows=[];classifications=Counter()
    with discovery_box(.12):
        for task in io.read(source/'tasks.json'):
            d=source/str(task['seed'])
            for n,h in task['files'].items():assert io.sha(d/n)==h
            inp=io.read(d/'input.json');x=np.array(inp['x_observed']);v=np.array(inp['v_observed'])
            with np.load(d/'pool.npz',allow_pickle=False) as z:regs=z['regions']
            language=[[path for path in product(range(4),repeat=4) if possible(x[i],v[i],path)] for i in range(4)]
            assert list(map(len,language))==inp['pool']['single_counts'];counts['independent_single_paths']+=1024
            assert np.prod(list(map(len,language)))==inp['pool']['enumerated']
            rerun,_=pool(x,v,np.array(inp['original']),inp['visited']);assert regs.tobytes()==rerun.tobytes()
            counts['complete_pool_replays']+=1
            proofsets={};steps={};timing={};taskcounts={}
            for cfg in configs:
                name=cfg['name'];meta=io.read(d/(name+'.json'))
                with np.load(d/(name+'.npz'),allow_pickle=False) as z:first=z['first_step'];a=z['proof_credit']
                by={p['index']:p for p in meta['proofs']};assert len(by)==len(meta['proofs'])==int((first>0).sum())
                for p in meta['proofs']:
                    i=p['index'];assert p['step']==first[i] and p['mode']==regs[i].tobytes().hex()
                    value=reference.fixed_value(x,v,a[i],regs[i]);assert (p['structural_empty'] and value is None) or value==F(p['lower'])>0
                    if p['kind']=='transfer':
                        origin=by[p['source_index']];assert origin['kind']=='local' and origin['step']<=p['step'] and p['source_index']!=i
                        assert a[i].tobytes()==a[p['source_index']].tobytes();counts['causal_transfers']+=1
                        match=[q for q in meta['directions'] if q['source_index']==p['source_index']]
                        assert len(match)==1 and match[0]['used_step']==p['step']
                        assert np.array(match[0]['credit']).tobytes()==a[i].tobytes()
                    counts['independent_fraction_proofs']+=1
                steps[name]=first;proofsets[name]=set(by);timing[name]=meta['total_seconds']
                t=totals[name];t['rejected']+=len(by);t['seconds']+=meta['total_seconds']
                for field in ['local_response_pairs','transfer_response_pairs','total_response_pairs','exact_calls']:
                    t[field]+=meta[field]
                t['transfer_rejections']+=sum(p['kind']=='transfer' for p in meta['proofs'])
                t['directions']+=len(meta['directions'])
                for p in [1,4,16,64,128]:t['prefix_'+str(p)]+=int(((first>0)&(first<=p)).sum())
                t['maximum_named_array_bytes']=max(t['maximum_named_array_bytes'],meta['named_array_bytes_subtotal_max'])
                taskcounts[name]=len(by)
            for c in CHANNELS:
                off=steps[c+'_independent'];on=steps[c+'_reuse'];mask=off>0
                assert np.all((on[mask]>0)&(on[mask]<=off[mask]));counts['dominance_regions']+=int(mask.sum())
                totals[c+'_reuse']['additional_rejections']+=len(proofsets[c+'_reuse']-proofsets[c+'_independent'])
                totals[c+'_reuse']['seconds_difference']+=timing[c+'_reuse']-timing[c+'_independent']
            zero=steps['zero_independent'];long=steps['zero_independent256'];mask=zero>0
            assert np.array_equal(zero[mask],long[mask]);counts['extended_prefix_regions']+=int(mask.sum())
            geonotes={};geo_count=Counter()
            for i,reg in enumerate(regs):
                key=reg.tobytes().hex();poly,note=geometry.explicit_geometry(x,v,key)
                aa,rhs=inequalities(x,v,key);counts['geometry_certificates']+=certificates(aa,rhs,note)
                if any(i in indices for indices in proofsets.values()):assert note['classification']=='infeasible'
                geonotes[key]=note;geo_count[note['classification']]+=1
            with np.load(d/'pdhg60.npz',allow_pickle=False) as z:pmask=z['rejected']
            pm=io.read(d/'pdhg60.json')
            for index,p in pm['metadata']['certificates'].items():
                i=int(index);assert exact_certificate(x,v,regs[i],np.array(p['p']),np.array(p['a']))['positive']
                assert geonotes[regs[i].tobytes().hex()]['classification']=='infeasible';counts['pdhg_proofs']+=1
            assert int(pmask.sum())==len(pm['metadata']['certificates'])
            io.save(out/(str(task['seed'])+'_geometry.json'),geonotes)
            classifications.update(geo_count)
            rows.append(dict(seed=task['seed'],regions=len(regs),classifications=dict(geo_count),rejected=taskcounts,
                             seconds=timing,pool_seconds=inp['pool']['total_seconds'],pdhg_rejected=int(pmask.sum()),pdhg_seconds=pm['seconds']))
            print(dict(audited_tasks=len(rows),proofs=counts['independent_fraction_proofs'],seconds=time.perf_counter()-begin),flush=True)
    io.save(out/'rows.json',rows)
    summary=dict(passed=True,tasks=32,regions=s['regions'],counts=dict(counts),classifications=dict(classifications),
        totals={k:dict(v) for k,v in totals.items()},source_summary_sha256=io.sha(source/'summary.json'),
        seconds=time.perf_counter()-begin,query_targets_accessed=False,query_risk_evaluated=False,
        outputs_sha256={'rows.json':io.sha(out/'rows.json')})
    io.save(out/'summary.json',summary);print(summary,flush=True)


if __name__=='__main__':
    root=Path(__file__).resolve().parents[2];out=root/'results/cross_region_credit/audit_v1'
    out.mkdir(parents=True,exist_ok=False);run(root,out)
