"""323 exhaustive OLD reference coverage: triggers, frozen shells, or ranking."""
import argparse
from collections import Counter,defaultdict
from fractions import Fraction as F
from pathlib import Path
import math
import time
import traceback
import numpy as np
from branch_image_chain_v1 import fixed_value
from evaluate_complete_credit_mode_geometry_v1 import read,save,sha

CHANNELS=['dual','dual_plus_residual','residual','bp','random_sign','zero']
CATEGORIES=['no_trigger','outside_window','ranked_out','selected']


def run(root,out):
    begin=time.perf_counter();source=root/'results/branch_image_chain';risk=read(source/'conditional_risk_v1/summary.json')
    assert risk['passed'] and read(source/'audit_v1/summary.json')['passed'] and read(source/'geometry_audit_v1/summary.json')['passed']
    for n,digest in risk['outputs_sha256'].items():assert sha(source/'conditional_risk_v1'/n)==digest
    rows=read(source/'development_v1/rows.json');byseed=defaultdict(list)
    for row in rows:
        assert sha(source/'development_v1'/row['file'])==row['sha256'];byseed[row['seed']].append(read(source/'development_v1'/row['file']))
    mom=root/'results/confirmation_conditional_risk/moments';ms=read(mom/'summary.json');assert sha(mom/'tasks.json')==ms['tasks_sha256']
    mtasks={t['seed']:t for t in read(mom/'tasks.json')};design=root/'outputs/ttt-pc-alm-research/323_chain_search_opportunity_protocol_v1.md'
    save(out/'protocol.json',dict(source_sha256={n:sha(root/'work/experiments'/n) for n in [Path(__file__).name,'branch_image_chain_v1.py']},
         design_sha256=sha(design),risk_sha256=sha(source/'conditional_risk_v1/summary.json'),tasks=64,channels=CHANNELS,
         categories=CATEGORIES,posthoc_old_task_diagnosis=True,query_targets_accessed=False,resources_matched=False))
    taskrows=[];details=[];comparisons=[];counts=Counter()
    for taskrow in read(source/'geometry_v1/tasks.json'):
        seed=taskrow['seed'];geo=read(source/'geometry_v1'/taskrow['file']);mt=mtasks[seed]
        assert sha(mom/mt['file'])==mt['sha256']
        with np.load(mom/mt['file'],allow_pickle=False) as z:vol=z['volumes'];means=z['means']
        weights=vol/math.fsum(map(float,vol));keys=mt['keys'];prior=set(geo['prior_positive_modes']);assert prior<=set(keys)
        selected=np.array([k in prior for k in keys]);mass=float(weights[selected].sum());assert mass>0
        full=np.einsum('k,kbq->bq',weights,means);mu=np.einsum('k,kbq->bq',weights*selected/mass,means)
        records=byseed[seed];x=np.array(geo['x_observed']);v=np.array(geo['v_observed'])
        prepared=[]
        for rec in records:
            original=np.array(list(bytes.fromhex(rec['original_mode']))).reshape(4,4)
            prepared.append((rec,original,{n:np.array(rec['credits'][n]) for n in CHANNELS}))
        perchannel={n:{c:[] for c in CATEGORIES} for n in CHANNELS}
        for mode,weight in zip(keys,weights):
            if mode in prior:continue
            target=np.array(list(bytes.fromhex(mode))).reshape(4,4);found_by={n:mode in geo['methods'][n]['positive_modes'] for n in CHANNELS}
            in_window={n:False for n in CHANNELS}
            for rec,original,credits in prepared:
                distance=int(np.sum(target!=original));assert distance>0
                for name in CHANNELS:
                    result=rec['results'][name];minimum=result['minimum_nonexcluded_hamming']
                    if minimum is None or not minimum<=distance<=minimum+2:continue
                    in_window[name]=True;value=fixed_value(x,v,credits[name],target)
                    assert value is not None and value<=0;counts['feasible_target_nonpositive_bounds']+=1
                    shell=[p for p in result['proposals'] if p['hamming']==distance]
                    present=any(p['mode']==mode for p in shell)
                    if not present:
                        assert len(shell)==8
                        worst=shell[-1];assert (value,mode)>(F(worst['lower']),worst['mode'])
                        counts['independent_ranking_exclusions']+=1
                    else:counts['selected_target_occurrences']+=1
                    comparisons.append(dict(seed=seed,location=rec['location'],channel=name,mode=mode,distance=distance,
                                            lower=str(value),selected_here=present,
                                            last_retained_lower=shell[-1]['lower'],last_retained_mode=shell[-1]['mode']))
            for name in CHANNELS:
                category='selected' if found_by[name] else 'no_trigger' if not records else 'ranked_out' if in_window[name] else 'outside_window'
                if found_by[name]:assert in_window[name]
                perchannel[name][category].append(float(weight))
                details.append(dict(seed=seed,mode=mode,channel=name,posterior_mass=float(weight),category=category))
        risks={}
        for grid in [257,129]:
            idx=np.arange(257) if grid==257 else np.arange(0,257,2);r=mu[:,idx]-full[:,idx]
            risks[str(grid)]=[float(np.mean(r[a]*r[b])) for a,b in [(0,1),(2,3)]]
        taskrows.append(dict(seed=seed,triggers=len(records),prior_mass=mass,missing_mass=math.fsum(float(w) for k,w in zip(keys,weights) if k not in prior),
                             categories={n:{c:dict(modes=len(ws),mass=math.fsum(ws)) for c,ws in data.items()} for n,data in perchannel.items()},
                             pool_excess_pairs=risks))
        if records:print(dict(seed=seed,trigger_states=len(records),missing_modes=sum(k not in prior for k in keys),seconds=time.perf_counter()-begin),flush=True)
    aggregate={n:{c:dict(mode_pairs=sum(t['categories'][n][c]['modes'] for t in taskrows),
                         mean_mass=math.fsum(t['categories'][n][c]['mass'] for t in taskrows)/64) for c in CATEGORIES} for n in CHANNELS}
    opportunity={}
    for grid in ['257','129']:
        opportunity[grid]={}
        for label,predicate in [('all',lambda t:True),('no_trigger',lambda t:not t['triggers']),('with_trigger',lambda t:bool(t['triggers']))]:
            chosen=[t for t in taskrows if predicate(t)]
            pairs=[math.fsum(t['pool_excess_pairs'][grid][i] for t in chosen)/64 for i in range(2)]
            opportunity[grid][label]=dict(tasks=len(chosen),pair_contributions=pairs,mean_contribution=math.fsum(pairs)/2)
    missing=math.fsum(t['missing_mass'] for t in taskrows)/64
    for name in CHANNELS:assert abs(math.fsum(v['mean_mass'] for v in aggregate[name].values())-missing)<1e-14
    files={}
    for name,value in [('tasks.json',taskrows),('targets.json',details),('ranking_witnesses.json',comparisons),('aggregate.json',aggregate),('risk_opportunity.json',opportunity)]:
        save(out/name,value);files[name]=sha(out/name)
    files['protocol.json']=sha(out/'protocol.json')
    result=dict(passed=True,counts=dict(counts),aggregate=aggregate,risk_opportunity=opportunity,mean_missing_mass=missing,
                seconds=time.perf_counter()-begin,outputs_sha256=files,query_targets_accessed=False,
                posthoc_old_task_diagnosis=True,resources_matched=False,independent_task_gain_established=False)
    save(out/'summary.json',result);print({k:v for k,v in result.items() if k!='outputs_sha256'},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    try:run(Path(__file__).resolve().parents[2],args.out)
    except Exception:save(args.out/'failure.json',dict(traceback=traceback.format_exc()));raise
