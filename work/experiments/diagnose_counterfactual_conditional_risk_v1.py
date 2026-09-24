"""295: cached OLD64 conditional-risk decomposition; no fitting or query answers."""
import argparse
from collections import defaultdict
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

SEEDS = list(range(5910000, 5910064))
RULES = ['all_steps', 'changed_forward_mode', 'first_global_forward_mode']
POOLS = ['original_alm', 'bp240'] + RULES
METHODS = ['original_alm', 'counterfactual', 'bp240']
POOL_FOR = dict(original_alm='original_alm', counterfactual='changed_forward_mode', bp240='bp240')
METRICS = ['actual_excess', 'policy_excess', 'read_error', 'cross_term']
PAIRS = [(0, 1), (2, 3)]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def cross(a, b):
    return float(np.mean(a*b))


def scalar_cross(a, b):
    return math.fsum(float(x)*float(y) for x, y in zip(a, b, strict=True))/len(a)


def decomposition(g, ma, mb, ka, kb, dot=cross):
    return dict(actual_excess=dot(g-ma, g-mb), policy_excess=dot(ka-ma, kb-mb),
                read_error=dot(g-ka, g-kb),
                cross_term=dot(ka-ma, g-kb)+dot(kb-mb, g-ka))


def self_test():
    # Exact rational proof of the implemented identity, with unequal batch means.
    for g, ma, mb, ka, kb in [(3, 2, 5, 7, -1), (0, 1, -1, 2, -2), (2, 2, 2, 2, 2)]:
        g, ma, mb, ka, kb = [Fraction(x, 7) for x in (g, ma, mb, ka, kb)]
        actual = (g-ma)*(g-mb)
        rhs = (ka-ma)*(kb-mb)+(g-ka)*(g-kb)+(ka-ma)*(g-kb)+(kb-mb)*(g-ka)
        assert actual == rhs
        assert (g-ma)*(g-mb)-(ka-ma)*(ka-mb) == g*g-ka*ka-(g-ka)*(ma+mb)
    # Negative cross-batch estimates are intentionally preserved.
    assert cross(np.array([1.]), np.array([-1.])) == -1.
    return dict(passed=True, exact_rational_cases=3, negative_estimates_not_clipped=True)


def run(root, out):
    begin = time.perf_counter()
    base = root/'results/confirmation_conditional_risk'
    mom, maudit, ref, raudit = [base/n for n in ['moments', 'moments_audit', 'reference', 'reference_audit']]
    prop = root/'results/counterfactual_branch_proposals/development_v1'
    new = root/'results/counterfactual_credit_branching/full_support_v2'
    diag = root/'results/gradient_flat_split_states/development_v1'
    input_hashes = {}
    source_hashes = {}

    def checked(path, expected=None):
        digest = sha(path)
        if expected is not None:
            assert digest == expected, str(path)
        input_hashes[path.relative_to(root).as_posix()] = digest
        return read(path)

    def sources(protocol, scalar_source=None):
        hashes = protocol.get('source_sha256', {})
        if isinstance(hashes, str):
            assert scalar_source is not None
            hashes = {scalar_source: hashes}
        for name, digest in hashes.items():
            name = name if '/' in name else 'work/experiments/'+name
            assert sha(root/name) == digest, name
            if name in source_hashes:
                assert source_hashes[name] == digest, name
            source_hashes[name] = digest

    def sealed_outputs(folder):
        summary = checked(folder/'summary.json')
        assert summary['passed'], str(folder)
        for name, digest in summary['outputs_sha256'].items():
            path = folder/name
            assert sha(path) == digest, str(path)
            input_hashes[path.relative_to(root).as_posix()] = digest
        sources(read(folder/'protocol.json'), 'counterfactual_branch_proposals_v1.py' if folder == prop else None)
        return summary

    ms, mas, ras = [checked(p/'summary.json') for p in (mom, maudit, raudit)]
    assert ms['passed'] and mas['passed'] and ras['passed']
    assert ms['tasks'] == mas['counts']['task_aggregate_files'] == 64
    assert ras['complete_up_to_certified_zero_volume'] == 64
    assert ms['regional_batches'] == mas['counts']['particle_batches_replayed'] == 6376
    assert ms['particles'] == 13058048
    for s in (ms, mas, ras):
        assert not s['phase_accesses_query_targets']
    mp = checked(mom/'protocol.json', ms['protocol_sha256'])
    maproto = checked(maudit/'protocol.json', mas['protocol_sha256'])
    rap = checked(raudit/'protocol.json', ras['protocol_sha256'])
    assert sha(mom/'summary.json') == maproto['moments_summary_sha256']
    assert sha(raudit/'summary.json') == mp['reference_audit_sha256']
    assert sha(ref/'summary.json') == rap['reference_summary_sha256']
    assert mp['seeds'] == SEEDS and mp['batch_pairs'] == [list(p) for p in PAIRS]
    assert mp['batches'] == 4 and mp['query_points'] == 257
    for protocol in (mp, maproto, rap):
        sources(protocol)
    tasks = checked(mom/'tasks.json', ms['tasks_sha256'])
    coverage = checked(raudit/'coverage.json', ras['coverage_sha256'])
    assert [r['seed'] for r in tasks] == SEEDS
    assert len(coverage) == 64 and all(r['complete_up_to_certified_zero_volume'] for r in coverage)
    refs = {r['seed']: r for r in checked(ref/'coverage.json')}
    assert sorted(refs) == SEEDS
    for folder in (prop, new, diag):
        sealed_outputs(folder)
    proposals = {r['seed']: r for r in checked(prop/'tasks.json')}
    candidates = {r['seed']: r for r in checked(new/'rows.json')}
    drows = {r['seed']: r for r in checked(diag/'tasks.json')}
    old_hashes = checked(diag/'input_hashes.json')
    assert sorted(proposals) == sorted(candidates) == sorted(drows) == SEEDS
    folders = dict(original_alm=root/'results/certificate_activity_attribution/development',
                   bp240=root/'results/probe_continuation_credit/development')
    frozen_names = dict(original_alm='credit_control_probe33', bp240='probe_then_adam240_33')
    lookup = {method: {r['seed']: r for r in checked(folder/'rows.json')
                      if r['method'] == frozen_names[method]} for method, folder in folders.items()}
    design = root/'outputs/ttt-pc-alm-research/295_cached_conditional_risk_protocol.md'
    sources_here = {'work/experiments/'+Path(__file__).name: sha(Path(__file__))}
    source_hashes.update(sources_here)
    save(out/'protocol.json', dict(seeds=SEEDS, methods=METHODS, pools=POOLS, batch_pairs=PAIRS,
        grids=[257,129], metrics=METRICS, source_sha256=source_hashes, design_sha256=sha(design),
        phase_accesses_query_targets=False, fits_or_samples=False, new_confirmation=False,
        scope='OLD64 post-result diagnostic; expensive cached posterior reference, not an online resource benchmark'))
    save(out/'self_test.json', self_test())
    rows, ideal_rows, contrasts, task_info, cache = [], [], [], [], {}
    max_identity = max_scalar = max_direct = max_mean = 0.
    for task in tasks:
        seed = task['seed']
        mpath = mom/task['file']
        assert sha(mpath) == task['sha256']
        input_hashes[mpath.relative_to(root).as_posix()] = task['sha256']
        with np.load(mpath, allow_pickle=False) as z:
            q, volumes, means = z['q'], z['volumes'], z['means']
        keys = task['keys']
        assert means.shape == (len(keys),4,257) and volumes.shape == (len(keys),)
        assert len(set(keys)) == len(keys) and np.all(volumes > 0) and np.isfinite(means).all()
        assert np.array_equal(q, np.linspace(0,1,257))
        r = refs[seed]
        rawref = checked(ref/r['file'], r['sha256'])
        assert proposals[seed]['reference_sha256'] == r['sha256']
        bykey = {p['pattern']:p['volume'] for p in rawref['reference']['positive_regions']}
        assert set(keys) == set(bykey)
        assert np.array_equal(volumes, np.array([bykey[k] for k in keys]))
        weights = volumes/volumes.sum()
        full = np.einsum('k,kbq->bq', weights, means)
        # Independently sum every coordinate over regions, not just risk scalars.
        full_scalar = np.array([[math.fsum(float(w)*float(m) for w,m in zip(weights, means[:,b,j]))
                                 for j in range(257)] for b in range(4)])
        max_mean = max(max_mean, float(np.max(np.abs(full-full_scalar))))
        pools, predictions, selected = {}, {}, {}
        for method, folder in folders.items():
            oldrow = lookup[method][seed]
            path = folder/oldrow['file']
            digest = sha(path)
            assert digest == oldrow['sha256'] == old_hashes[path.relative_to(root).as_posix()]
            input_hashes[path.relative_to(root).as_posix()] = digest
            with np.load(path, allow_pickle=False) as z:
                assert np.array_equal(z['q_observed'], q)
                assert np.array_equal(z['x_observed'], np.array(rawref['x_observed']))
                assert np.array_equal(z['v_observed'], np.array(rawref['v_observed']))
                predictions[method] = z['prediction']
                selected[method] = z['selected_b']
            pools[method] = set(oldrow['metadata']['positive_modes'])
            role = 'alm' if method == 'original_alm' else 'bp'
            assert pools[method] == set(drows[seed]['positive_modes'][role])
        p = proposals[seed]
        assert p['original_sha256'] == lookup['original_alm'][seed]['sha256']
        for rule in RULES:
            pools[rule] = pools['original_alm'] | set(p['rules'][rule]['new_positive_modes'])
        cr = candidates[seed]
        assert cr['source_file_sha256'] == lookup['original_alm'][seed]['sha256']
        assert set(cr['metadata']['positive_modes']) == pools['changed_forward_mode']
        cpath = new/cr['output_file']
        assert sha(cpath) == cr['output_sha256']
        with np.load(cpath, allow_pickle=False) as z:
            predictions['counterfactual'] = z['prediction']
            assert np.array_equal(z['selected_b'], selected['original_alm'])
        assert all(v.shape == (257,) and np.isfinite(v).all() for v in predictions.values())
        mus, masses = {}, {}
        for pool in POOLS:
            assert pools[pool] and pools[pool] <= set(keys), (seed, pool)
            mask = np.array([k in pools[pool] for k in keys])
            mass = float(weights[mask].sum())
            masses[pool] = mass
            w = weights*mask/mass
            mus[pool] = np.einsum('k,kbq->bq', w, means)
            scalar_mu = np.array([[math.fsum(float(wi)*float(mi) for wi,mi in zip(w,means[:,b,j]))
                                   for j in range(257)] for b in range(4)])
            max_mean = max(max_mean,float(np.max(np.abs(mus[pool]-scalar_mu))))
            expected = p['old_alm_coverage'] if pool == 'original_alm' else (
                p['old_bp_coverage'] if pool == 'bp240' else p['rules'][pool]['union_coverage'])
            assert abs(mass-expected) < 1e-12, (seed,pool,mass,expected)
        task_info.append(dict(seed=seed, masses=masses, mode_counts={n:len(pools[n]) for n in POOLS}))
        cache[f'{seed}_full'] = full
        cache[f'{seed}_mus'] = np.array([mus[n] for n in POOLS])
        cache[f'{seed}_predictions'] = np.array([predictions[n] for n in METHODS])
        for grid in (257,129):
            idx = np.arange(257) if grid == 257 else np.arange(0,257,2)
            for a,b in PAIRS:
                fa,fb = full[a,idx],full[b,idx]
                ideal = {}
                for pool in POOLS:
                    ka,kb = mus[pool][a,idx],mus[pool][b,idx]
                    val = cross(ka-fa,kb-fb)
                    max_scalar = max(max_scalar,abs(val-scalar_cross(ka-fa,kb-fb)))
                    ideal[pool] = val
                    ideal_rows.append(dict(seed=seed,grid=grid,pair=[a,b],pool=pool,
                        policy_excess=val,delta_vs_original=val-ideal['original_alm'],mass=masses[pool]))
                values = {}
                for method in METHODS:
                    pool = POOL_FOR[method]
                    g = predictions[method][idx]
                    ka,kb = mus[pool][a,idx],mus[pool][b,idx]
                    value = decomposition(g,fa,fb,ka,kb)
                    scalar = decomposition(g,fa,fb,ka,kb,scalar_cross)
                    gap = abs(value['actual_excess']-math.fsum(value[k] for k in METRICS[1:]))
                    max_identity = max(max_identity,gap)
                    max_scalar = max(max_scalar,max(abs(value[k]-scalar[k]) for k in METRICS))
                    rows.append(dict(seed=seed,method=method,grid=grid,pair=[a,b],**value))
                    values[method] = value
                for method in ['counterfactual','bp240']:
                    g,old = predictions[method][idx],predictions['original_alm'][idx]
                    direct = float(np.mean(g*g-old*old)-np.mean((g-old)*(fa+fb)))
                    delta = {k:values[method][k]-values['original_alm'][k] for k in METRICS}
                    direct_scalar = math.fsum(float(x)*float(x)-float(y)*float(y)
                        -(float(x)-float(y))*(float(u)+float(v)) for x,y,u,v in zip(g,old,fa,fb))/grid
                    max_direct = max(max_direct,abs(direct-delta['actual_excess']),abs(direct-direct_scalar))
                    contrasts.append(dict(seed=seed,grid=grid,pair=[a,b],method=method,
                                          reference='original_alm',direct_delta=direct,**delta))
        if (seed-SEEDS[0]+1)%16 == 0:
            print(dict(tasks=seed-SEEDS[0]+1, query_targets_accessed=False), flush=True)
    assert len(rows)==768 and len(ideal_rows)==1280 and len(contrasts)==512
    assert max(max_identity,max_scalar,max_direct,max_mean) < 1e-12

    def aggregate(records, group, metrics):
        buckets = defaultdict(list)
        for row in records:
            buckets[(row[group],row['grid'])].append(row)
        result = []
        for (name,grid), subset in buckets.items():
            assert len(subset)==128
            row = {group:name, 'grid':grid, 'tasks':64}
            for metric in metrics:
                pair_means = [math.fsum(r[metric] for r in subset if r['pair']==list(pair))/64 for pair in PAIRS]
                row[metric] = dict(mean=math.fsum(pair_means)/2, pair_means=pair_means,
                                   pair_difference=pair_means[0]-pair_means[1])
            result.append(row)
        return result

    aggregates = dict(actual=aggregate(rows,'method',METRICS),
        ideal=aggregate(ideal_rows,'pool',['policy_excess','delta_vs_original','mass']),
        contrasts=aggregate(contrasts,'method',METRICS))
    for name,obj in [('rows',rows),('ideal_rows',ideal_rows),('contrasts',contrasts),
                     ('tasks',task_info),('aggregates',aggregates),('input_hashes',input_hashes)]:
        save(out/f'{name}.json',obj)
    with (out/'cached_means.npz').open('xb') as stream:
        np.savez_compressed(stream,**cache)
    for name,digest in source_hashes.items():
        assert sha(root/name)==digest,name
    for name,digest in input_hashes.items():
        assert sha(root/name)==digest,name
    summary = dict(passed=True,tasks=64,decompositions=len(rows),ideal_estimates=len(ideal_rows),
        contrasts=len(contrasts),query_targets_accessed=False,fits_or_samples=False,new_confirmation=False,
        max_identity_gap=max_identity,max_independent_scalar_gap=max_scalar,max_direct_contrast_gap=max_direct,
        max_independent_mean_gap=max_mean,seconds=time.perf_counter()-begin,
        core_research_goal_complete=False,outputs_sha256={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()})
    save(out/'summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,required=True)
    root=parser.parse_args().project.resolve()
    out=root/'results/counterfactual_conditional_risk/development_v2'
    out.mkdir(parents=True,exist_ok=False)
    try:
        run(root,out)
    except BaseException as exc:
        save(out/'failure.json',dict(error_type=type(exc).__name__,message=str(exc),no_automatic_retry=True))
        raise


if __name__=='__main__':
    main()
