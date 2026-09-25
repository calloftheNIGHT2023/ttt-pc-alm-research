"""Independent 309 archive audit: forward replay, masks, sets, and certificates."""
import argparse
from collections import Counter
from fractions import Fraction as F
from pathlib import Path
import time

import numpy as np
import scipy.optimize as opt
import layer_credit_interaction_v1 as candidate
from evaluate_complete_credit_mode_geometry_v1 import read, save, sha, FAMILIES
import complete_credit_mode_geometry_v1 as geometry


def exact_forward(b, x):
    value = [F(q) for q in x]; codes = []
    for bias in b:
        z = [q+F(bias) for q in value]
        codes.extend(sum(t>=c for c in [F(0), F(1,2), F(1)]) for t in z)
        value = [max(F(0), min(2*t, 2-2*t)) for t in z]
    return value, bytes(codes).hex()


def run(root,out):
    begin=time.perf_counter()
    source=root/'results/layer_credit_interaction/development_v1'
    evaluated=root/'results/layer_credit_interaction/geometry_v1'
    counts=Counter()
    for folder in [source,evaluated]:
        summary=read(folder/'summary.json');assert summary['passed']
        for name,digest in summary['outputs_sha256'].items():assert sha(folder/name)==digest
        for name,digest in read(folder/'protocol.json')['source_sha256'].items():
            assert sha(root/'work/experiments'/name)==digest
    all_sets={}; zero={}; guard_example=None
    discrepancies=[]
    for meta in read(source/'rows.json'):
        p=read(source/meta['file']);state=candidate.decode(p['state']);family=p['family']
        key=(p['seed'],*p['location']);seed=p['seed']
        assert p['seed']==meta['seed'] and p['location']==meta['location']
        mask_names={''.join(map(str,m)) for m in candidate.masks(4)}
        assert {r['mask'] for r in p['proposals']}==mask_names
        assert len(p['proposals'])==16
        all_sets.setdefault(seed,{})
        for r in p['proposals']:
            exact=r['exact']
            value, mode=exact_forward(exact['b'],state['x'])
            assert value==[F(q) for q in exact['forward']] and mode==r['exact_mode']
            counts['exact_forward_replays']+=1
            # Independent operation-order floating forward path, no mode_list call.
            prev=np.array(state['x'],dtype=float); cc=[]
            for b in r['float_b']:
                z=prev+b
                cc.extend(int(z0>=0)+int(z0>=.5)+int(z0>=1) for z0 in z)
                prev=np.maximum(0,1-np.abs(2*z-1))
            assert bytes(cc).hex()==r['float_mode']
            counts['floating_forward_replays']+=1
            for backend in ['exact','float']:
                for suffix in [r['mask'],'union']:
                    name=family+'_'+backend+'_'+suffix
                    all_sets[seed].setdefault(name,set()).add(r[backend+'_mode'])
            for name in ['b','h']:
                actual=np.array(exact[name],dtype=float)
                assert float(np.max(abs(actual-np.array(r['float_'+name]))))==r['gaps'][name]
                counts['floating_gap_rechecks']+=1
            if r['float_value_discrepancy'] or r['float_mode_discrepancy']:
                discrepancies.append(dict(seed=seed,location=p['location'],family=family,mask=r['mask'],
                                          gaps=r['gaps'],mode_difference=r['float_mode_discrepancy']))
            if r['mask']=='0000':
                if key in zero:assert zero[key]==exact
                else:zero[key]=exact
        if guard_example is None:guard_example=(state,p['proposals'])
    # Runtime check only of candidate; evaluator global LP is intentionally separate.
    originals=[]
    def prohibited(*args,**kwargs):raise AssertionError('Global optimizer/BP called by local candidate')
    try:
        for obj,name in [(candidate.cold.bp,'evaluate'),(candidate.cold.bp,'refine'),(opt,'linprog'),(opt,'minimize')]:
            originals.append((obj,name,getattr(obj,name)));setattr(obj,name,prohibited)
        output,_=candidate.combine(guard_example[0])
        from exact_quadratic_events_v1 import encode
        assert encode(output)==guard_example[1]
        counts['guarded_full_interventions']=len(output)
    finally:
        for obj,name,value in originals:setattr(obj,name,value)
    tasks=read(evaluated/'tasks.json')
    assert len(tasks)==64 and sum(t['selected_states'] for t in tasks)==131
    recomputed={}
    for t in tasks:
        data=read(evaluated/t['file']);seed=t['seed']
        assert sha(evaluated/t['file'])==t['sha256']
        x=np.array(data['x_observed']);v=np.array(data['v_observed'])
        positive={m for m,g in data['geometry'].items() if g['positive_volume_certified']}
        unknown={m for m,g in data['geometry'].items() if g['classification']=='unresolved'}
        old=set(data['original_positive_modes'])
        for mode,result in data['geometry'].items():
            _,aa,rr,_,_=geometry.matrices(x,v,mode)
            counts['exact_certificate_rechecks']+=geometry.verify_certificates(aa,rr,result)
        for name,m in data['methods'].items():
            proposed=all_sets.get(seed,{}).get(name,set())
            assert proposed==set(m['proposal_modes'])
            pp=proposed&positive;new=pp-old
            assert pp==set(m['positive_modes']) and new==set(m['new_positive_modes'])
            assert proposed&unknown==set(m['unknown_modes'])
            counts['method_set_rechecks']+=1
            agg=recomputed.setdefault(name,Counter())
            agg['tasks_with_new_positive']+=bool(new);agg['task_positive_pairs']+=len(new)
            agg['unknown_pairs']+=len(proposed&unknown)
            agg['volume_missing_pairs']+=len(m['positive_modes_without_numerical_volume'])
            agg['numeric_new_volume']+=sum(data['geometry'][mm]['volume'] for mm in sorted(new)
                                            if data['geometry'][mm]['volume'] is not None)
    actual=read(evaluated/'aggregate.json')
    assert set(actual)==set(recomputed)
    for n,a in actual.items():
        for field,value in a.items():
            if field=='numeric_new_volume':
                assert abs(value-recomputed[n][field])<=1e-25+abs(value)*1e-14
            else:assert value==recomputed[n][field]
            counts['aggregate_numbers_rechecked']+=1
    save(out/'floating_discrepancies.json',discrepancies)
    result=dict(passed=True,counts=dict(counts),float_discrepancies=len(discrepancies),
        source_sha256=sha(Path(__file__)),input_summaries_sha256={p.name:sha(p/'summary.json') for p in [source,evaluated]},
        discrepancies_sha256=sha(out/'floating_discrepancies.json'),
        query_targets_accessed=False,independent_task_gain_established=False,seconds=time.perf_counter()-begin)
    save(out/'summary.json',result);print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    run(Path(__file__).resolve().parents[2],args.out)
