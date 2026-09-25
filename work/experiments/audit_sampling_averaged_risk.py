"""Frozen expected-sampling risk audit, no teacher or query answer access."""
import argparse,hashlib,json,itertools
from pathlib import Path
import numpy as np


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def ci(delta):
    index=np.random.default_rng(201441).integers(0,len(delta),(20000,len(delta)))
    return np.quantile(delta[index].mean(1),[.025,.975]).tolist()


def decompose(mu,second,weights,mask,q,pairs):
    full=np.einsum('k,rkq->rq',weights,mu);p=weights*mask;p=p/p.sum()
    mean=np.einsum('k,rkq->rq',p,mu);moment=np.einsum('k,rkq->rq',p,second)
    estimates=[]
    for a,b in pairs:
        bias=(mean[a]-full[a])*(mean[b]-full[b]);variance=(moment[a]+moment[b])/2-mean[a]*mean[b]
        estimates.append(dict(truncation=float(np.trapezoid(bias,x=q)),sampling=float(np.trapezoid(variance,x=q)/512),
            coarse_truncation=float(np.trapezoid(bias[::2],x=q[::2])),coarse_sampling=float(np.trapezoid(variance[::2],x=q[::2])/512)))
    return {k:float(np.mean([row[k] for row in estimates])) for k in estimates[0]}


def verify():
    # Exhaustive two-point posterior: ordered N=2 draw probabilities prove
    # the decomposition without a Monte Carlo approximation.
    values=np.array([.1,.8]);prob=np.array([.25,.75]);target=.4;n=2;mean=prob@values;variance=prob@((values-mean)**2)
    direct=sum(prob[a]*prob[b]*((values[a]+values[b])/2-target)**2 for a,b in itertools.product(range(2),repeat=2))
    formula=(mean-target)**2+variance/n;assert abs(direct-formula)<1e-15
    return dict(passed=True,exact_finite_distribution_risk=direct,formula_risk=formula)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--project',type=Path,required=True);args=parser.parse_args();root=args.project/'results'
    online=root/'typed_routing_memory/development';old_analysis=root/'typed_routing_memory/analysis';curve=root/'posterior_state_reuse/conditional_risk';out=root/'sampling_averaged_risk/diagnostic'
    parent=json.loads((online/'protocol.json').read_text());reference=json.loads((curve/'protocol.json').read_text());hashes=parent['source_sha256'].copy()
    for name,value in reference['source_sha256'].items():
        if name in hashes:assert hashes[name]==value,name
        hashes[name]=value
    for name,value in hashes.items():assert sha(Path(__file__).with_name(name))==value,name
    hashes[Path(__file__).name]=sha(Path(__file__))
    methods=[cfg['name'] for cfg in parent['configs']];seeds=list(range(parent['seed0'],parent['seed0']+parent['count']))
    protocol=dict(source_sha256=hashes,parent_protocol_sha256=sha(online/'protocol.json'),reference_protocol_sha256=sha(curve/'protocol.json'),
        conditional_analysis_sha256=sha(old_analysis/'summary.json'),conditional_rows_sha256=sha(old_analysis/'conditional_risks.json'),
        methods=methods,seeds=seeds,primary_pairs=[[0,1],[2,3]],secondary_pairs=list(itertools.combinations(range(4),2)),verification=verify(),
        scope='same numerical complete posterior volumes and 1025-point grid; expected random 512-particle risk, not a new predictor or fresh confirmation')
    out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists();(out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    saved={(r['seed'],r['method']):r for r in json.loads((old_analysis/'conditional_risks.json').read_text())}
    states={(r['seed'],r['method']):r for r in json.loads((online/'episodes.json').read_text()) if r['n_context']==4}
    rows=[];curve_audits=[];replays=0
    for seed in seeds:
        audit=json.loads((curve/f'audit_{seed}.json').read_text());path=curve/audit['curve_file'];assert sha(path)==audit['curve_sha256'];curve_audits.append(audit)
        with np.load(path) as z:q=z['q'];weights=z['weights'];patterns=z['patterns'];mu=z['region_means'];second=z['region_second_moments']
        assert mu.shape==second.shape and mu.shape[0]==4 and len(q)==1025
        for name in methods:
            state=states[seed,name];path=online/state['state_file'];assert sha(path)==state['state_sha256']
            mask=np.array([k in set(state['positive_mode_keys']) for k in patterns]);assert mask.any()
            main=decompose(mu,second,weights,mask,q,protocol['primary_pairs']);six=decompose(mu,second,weights,mask,q,protocol['secondary_pairs'])
            expected=main['truncation']+main['sampling'];old=saved[seed,name]
            assert abs(main['truncation']-old['ideal_truncation'])<1e-15;replays+=1
            rows.append(dict(seed=seed,method=name,state_sha256=state['state_sha256'],mass=float(weights[mask].sum()),fixed_state_excess=old['conditional_excess'],
                ideal_truncation=main['truncation'],expected_sampling=main['sampling'],expected_total_excess=expected,
                fixed_minus_sampling_expectation=old['conditional_excess']-expected,
                bias_fraction=main['truncation']/expected if expected>0 else None,
                coarse_grid_difference=(main['truncation']+main['sampling'])-(main['coarse_truncation']+main['coarse_sampling']),
                six_pair_expected_total=six['truncation']+six['sampling']))
    assert len(rows)==replays==320
    summary=[];paired=[];primary='alm_tied_forward_full'
    for name in methods:
        rr=[r for r in rows if r['method']==name];fields=['fixed_state_excess','ideal_truncation','expected_sampling','expected_total_excess','fixed_minus_sampling_expectation','six_pair_expected_total','mass']
        record=dict(method=name,**{key:float(np.mean([r[key] for r in rr])) for key in fields},maximum_abs_coarse_grid_change=max(abs(r['coarse_grid_difference']) for r in rr))
        record['aggregate_bias_fraction']=record['ideal_truncation']/record['expected_total_excess'];summary.append(record)
        if name!=primary:
            delta=np.array([next(r['expected_total_excess'] for r in rows if r['seed']==s and r['method']==primary)-next(r['expected_total_excess'] for r in rows if r['seed']==s and r['method']==name) for s in seeds])
            paired.append(dict(comparator=name,expected_excess_delta=float(delta.mean()),descriptive_ci95=ci(delta),better_tasks=int((delta<-1e-14).sum()),equal_tasks=int((np.abs(delta)<=1e-14).sum()),worse_tasks=int((delta>1e-14).sum())))
    result=dict(complete=True,source_hashes=len(hashes),context_methods=len(rows),original_truncation_replays=replays,curve_hashes=len(curve_audits),summary=summary,paired=paired,scope=protocol['scope'])
    (out/'rows.json').write_text(json.dumps(rows,indent=2),encoding='utf-8');(out/'curve_audits.json').write_text(json.dumps(curve_audits,indent=2),encoding='utf-8');(out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
