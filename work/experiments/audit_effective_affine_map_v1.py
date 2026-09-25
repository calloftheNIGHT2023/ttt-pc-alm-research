"""315 independent selected-formula probes and exact full local-optimum comparison."""
import argparse
from collections import Counter,defaultdict
from fractions import Fraction as F
import gzip
import json
from pathlib import Path
import time
import traceback
import numpy as np
from effective_affine_map_v1 import restore,canonical,pack,evaluate,digest
from selected_policy_scalar_reference_v1 import scalar
from complete_credit_amplitude_events_v2 import rational_state
from complete_credit_rational_reference_v1 import direct_step,tent
from evaluate_complete_credit_mode_geometry_v1 import read,save,sha


def mode(b,x):
    value=list(map(F,x));codes=[]
    for bias in b:
        zz=[a+bias for a in value];codes.extend(sum(z>=k for k in [F(0),F(1,2),F(1)]) for z in zz)
        value=[tent(z) for z in zz]
    return codes


def run(root,out):
    begin=time.perf_counter();source=root/'results/effective_affine_map/development_v1'
    original=root/'results/solver_policy_recurrence/development_v1'
    ss=read(source/'summary.json');assert ss['passed']
    for n,dig in ss['outputs_sha256'].items():assert sha(source/n)==dig
    tests=root/'results/effective_affine_map/reference_tests_v1/summary.json';ts=read(tests);assert ts['passed']
    for n,dig in ts['source_sha256'].items():assert sha(root/'work/experiments'/n)==dig
    for n,dig in read(source/'protocol.json')['source_sha256'].items():assert sha(root/'work/experiments'/n)==dig
    save(out/'protocol.json',dict(source_sha256={n:sha(root/'work/experiments'/n) for n in [Path(__file__).name,'selected_policy_scalar_reference_v1.py','complete_credit_rational_reference_v1.py']},
        support_summary_sha256=sha(source/'summary.json'),tests_summary_sha256=sha(tests),
        query_targets_accessed=False,resources_matched=False,
        exact_optimizer_disagreement_is_not_automatically_reconstruction_error=True))
    schema=read(original/'policy_schema.json')
    oldrows={(r['seed'],r['family']):r for r in read(original/'rows.json')}
    byseed=defaultdict(list)
    for r in read(source/'rows.json'):byseed[r['seed']].append(r)
    counts=Counter();max_gap=0.;optimizer_cases=[];formula_cases=[];files={}
    for seed,rows in sorted(byseed.items()):
        with gzip.open(source/rows[0]['map_bank_file'],'rt',encoding='utf-8') as stream:bank=json.load(stream)
        maps=[]
        for i,m in enumerate(bank['maps']):
            assert i==m['id'];values=restore(m['rows']);assert digest(values)==m['sha256'];maps.append(values)
        assert len({canonical(m) for m in maps})==len(maps)
        seen=set();block_codes=[{}, {}, {}]
        for row in rows:
            old=oldrows[seed,row['family']]
            with np.load(original/old['file'],allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
            with np.load(source/row['file'],allow_pickle=False) as z:assigned={k:z[k] for k in z.files}
            x,v=arrays['x_observed'],arrays['v_observed'];d=arrays['b'].shape[-1];n=len(x);cuts=[0,d,d+d*n,d+2*d*n]
            for t in range(64):
                for i,location in enumerate(arrays['locations']):
                    b,h,u=arrays['b'][t,i],arrays['h'][t,:,i],arrays['u'][t,:,i]
                    p=pack(b,h,u);idx=int(assigned['map_ids'][t,i]);m=maps[idx]
                    expected=evaluate(m,p)
                    policies={g:arrays['policy_'+g][t,i] for g in schema}
                    fixed=scalar(policies,schema,p,x,v,old['method']);assert fixed==expected
                    counts['exact_selected_formula_current_vectors']+=1
                    key=(old['method'],np.concatenate(list(policies.values())).tobytes())
                    if key not in seen:
                        rng=np.random.default_rng(np.random.SeedSequence([315097,seed,t,i,len(seen)]))
                        for _ in range(2):
                            q=[a+F(int(rng.integers(-100,101)),127) for a in p]
                            assert evaluate(m,q)==scalar(policies,schema,q,x,v,old['method'])
                            counts['exact_off_policy_formula_vectors']+=1
                        seen.add(key)
                    for j in range(3):
                        code=canonical(m[cuts[j]:cuts[j+1]]);bid=int(assigned['block_map_ids'][t,i,j])
                        if bid in block_codes[j]:assert block_codes[j][bid]==code
                        else:block_codes[j][bid]=code
                        counts['block_id_content_checks']+=1
                    actual=np.r_[arrays['b'][t+1,i],arrays['h'][t+1,:,i].ravel(),arrays['u'][t+1,:,i].ravel()]
                    error=abs(np.array(expected,float)-actual)
                    gaps=np.array([np.max(error[cuts[j]:cuts[j+1]]) for j in range(3)])
                    assert np.array_equal(gaps,assigned['gaps'][t,i]);counts['stored_gap_vectors']+=1
                    assert max(gaps)<=1e-10
                    refstate=rational_state(b,h,u,x,v);rr=direct_step(refstate,1)
                    ur=[]
                    for j in range(d):
                        prev=refstate['x'] if j==0 else rr['h'][j-1]
                        ur.append([refstate['direction'][j][ii]+(F(1,2)*(rr['h'][j][ii]-tent(prev[ii]+rr['b'][j])) if old['method']=='alm' else F(0)) for ii in range(n)])
                    exact=rr['b']+[q for ar in rr['h'] for q in ar]+[q for ar in ur for q in ar]
                    exactgap=float(max(abs(a-bb) for a,bb in zip(exact,expected)));max_gap=max(max_gap,exactgap)
                    counts['independent_exact_optimizer_steps']+=1
                    if exactgap>1e-10:
                        optimizer_cases.append(dict(seed=seed,family=row['family'],location=location.tolist(),step=t+1,
                            selected_vs_ideal_exact_max_gap=exactgap,selected_map_id=idx,
                            exact_b=rr['b'],selected_b=expected[:d],exact_activity_ties=rr['activity_ties'],exact_bias_ties=rr['bias_ties']))
                    fmode=mode(expected[:d],x);imode=mode(rr['b'],x);actualmode=arrays['forward_policy'][t,i].tolist()
                    counts['selected_formula_vs_float_forward_mode_differences']+=fmode!=actualmode
                    counts['ideal_optimizer_vs_float_forward_mode_differences']+=imode!=actualmode
                    counts['selected_vs_ideal_forward_mode_differences']+=fmode!=imode
            print(dict(seed=seed,family=row['family'],steps=counts['independent_exact_optimizer_steps'],
                optimizer_value_disagreements=len(optimizer_cases),max_exact_gap=max_gap),flush=True)
        assert len(seen)==bank['full_policy_cache_entries']
        for j in range(3):assert len(block_codes[j])==len(set(block_codes[j].values()))==bank['unique_block_maps'][j]
    assert counts['independent_exact_optimizer_steps']==25152
    save(out/'exact_optimizer_disagreements.json',optimizer_cases);files['exact_optimizer_disagreements.json']=sha(out/'exact_optimizer_disagreements.json')
    files['protocol.json']=sha(out/'protocol.json')
    final=dict(passed=True,selected_formula_reconstruction_passed=True,counts=dict(counts),
        exact_optimizer_value_disagreements=len(optimizer_cases),max_selected_vs_ideal_exact_gap=max_gap,
        query_targets_accessed=False,resources_matched=False,independent_task_gain_established=False,
        future_guard_validity_established=False,seconds=time.perf_counter()-begin,outputs_sha256=files)
    save(out/'summary.json',final);print(final,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    try:run(Path(__file__).resolve().parents[2],args.out)
    except Exception:
        save(args.out/'failure.json',dict(traceback=traceback.format_exc()));raise
