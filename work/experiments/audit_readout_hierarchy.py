"""Frozen between/within-region attribution and a last-layer gain ceiling.

Evaluator-only access to numerical reference moments, never teacher answers.
This does not construct a new online predictor or select a credit direction.
"""
import argparse, hashlib, itertools, json
from pathlib import Path
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def components(mu, second, weights, mask, q, pairs):
    p = weights * mask
    p /= p.sum()
    mean = np.einsum('k,rkq->rq', p, mu)
    result = []
    for a, b in pairs:
        mean2 = np.einsum('k,kq->q', p, mu[a] * mu[b])
        moment2 = np.einsum('k,kq->q', p, (second[a] + second[b]) / 2)
        within = moment2 - mean2
        between = mean2 - mean[a] * mean[b]
        result.append(dict(within=float(np.trapezoid(within, x=q) / 512),
                           between=float(np.trapezoid(between, x=q) / 512),
                           coarse_within=float(np.trapezoid(within[::2], x=q[::2]) / 512),
                           coarse_between=float(np.trapezoid(between[::2], x=q[::2]) / 512)))
    return {k: float(np.mean([r[k] for r in result])) for k in result[0]}


def verify():
    values = np.array([[.1, .3], [.5, .9]])
    p = np.array([.3, .7])
    means = values.mean(1)
    within = p @ values.var(1)
    between = p @ (means * means) - (p @ means) ** 2
    direct = np.sum(p[:, None] / 2 * (values - p @ means) ** 2)
    assert abs(direct - within - between) < 1e-15
    # Integrating the last bias on one affine query branch attains the
    # epsilon^2/3 single-particle upper bound when the fiber width is epsilon.
    eps = .001
    assert abs(4 * eps**2 / 12 - eps**2 / 3) < 1e-20
    return dict(passed=True,finite_distribution_variance=float(direct),
                decomposition=float(within+between),last_layer_ceiling=eps**2/(3*512))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, required=True)
    args = parser.parse_args()
    root = args.project.resolve()
    results = root/'results'
    prior = results/'sampling_averaged_risk/diagnostic'
    curve = results/'posterior_state_reuse/conditional_risk'
    online = results/'typed_routing_memory/development'
    out = results/'readout_hierarchy/diagnostic'
    parent = json.loads((prior/'protocol.json').read_text())
    hashes = dict(parent['source_sha256'])
    for name, expected in hashes.items():
        assert sha(Path(__file__).with_name(name)) == expected, name
    hashes[Path(__file__).name] = sha(Path(__file__))
    protocol = dict(source_sha256=hashes,parent_protocol_sha256=sha(prior/'protocol.json'),
                    parent_summary_sha256=sha(prior/'summary.json'),parent_rows_sha256=sha(prior/'rows.json'),
                    methods=parent['methods'],seeds=parent['seeds'],pairs=parent['primary_pairs'],
                    six_pairs=list(itertools.combinations(range(4),2)),particles=512,eps=.001,
                    verification=verify(),scope='frozen old-task moments; no new predictor, query labels or confirmation data')
    out.mkdir(parents=True,exist_ok=True)
    assert not (out/'protocol.json').exists()
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    expected = {(r['seed'],r['method']):r for r in json.loads((prior/'rows.json').read_text())}
    states = {(r['seed'],r['method']):r for r in json.loads((online/'episodes.json').read_text()) if r['n_context']==4}
    rows=[];contexts=[]
    for seed in protocol['seeds']:
        audit=json.loads((curve/f'audit_{seed}.json').read_text())
        path=curve/audit['curve_file']
        assert sha(path)==audit['curve_sha256']
        with np.load(path) as z:
            weights=z['weights'];mu=z['region_means'];second=z['region_second_moments'];q=z['q'];patterns=z['patterns'];v=z['v']
        condition=bool(np.any(v>protocol['eps']))
        contexts.append(dict(seed=seed,curve_sha256=audit['curve_sha256'],last_layer_bound_applicable=condition,
                             maximum_observed_value=float(v.max()),ceiling=protocol['eps']**2/(3*512) if condition else None))
        for name in protocol['methods']:
            state=states[seed,name]
            assert sha(online/state['state_file'])==state['state_sha256']
            mask=np.array([p in set(state['positive_mode_keys']) for p in patterns])
            main=components(mu,second,weights,mask,q,protocol['pairs'])
            secondary=components(mu,second,weights,mask,q,protocol['six_pairs'])
            total=main['within']+main['between']
            assert abs(total-expected[seed,name]['expected_sampling'])<1e-15
            rows.append(dict(seed=seed,method=name,**main,six_pair_within=secondary['within'],
                             six_pair_between=secondary['between'],sampling_total=total,
                             prior_sampling=expected[seed,name]['expected_sampling']))
    summary=[]
    for name in protocol['methods']:
        rr=[r for r in rows if r['method']==name]
        item=dict(method=name,**{k:float(np.mean([r[k] for r in rr])) for k in ['within','between','sampling_total','six_pair_within','six_pair_between']})
        item.update(within_fraction=item['within']/item['sampling_total'],
                    maximum_within_grid_change=max(abs(r['within']-r['coarse_within']) for r in rr),
                    maximum_between_grid_change=max(abs(r['between']-r['coarse_between']) for r in rr))
        summary.append(item)
    result=dict(complete=True,rows=len(rows),sum_checks=len(rows),source_hashes=len(hashes),
                last_layer_bound_contexts=sum(c['last_layer_bound_applicable'] for c in contexts),
                last_layer_risk_ceiling=protocol['eps']**2/(3*512),summary=summary,
                raw_negative_component_estimates=sum(r[k]<0 for r in rows for k in ['within','between']),scope=protocol['scope'])
    assert len(rows)==320
    for name,data in [('rows.json',rows),('contexts.json',contexts),('summary.json',result)]:
        (out/name).write_text(json.dumps(data,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
