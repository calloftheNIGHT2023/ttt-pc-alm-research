"""300: fixed OLD64 support-only stagnation locations, then sealed pool diagnosis."""
import argparse
from collections import defaultdict
from pathlib import Path
import math
import time
import numpy as np
import cold_stagnation_switch as cold
from posterior_confirmation_pipeline import discovery_box
from diagnose_gradient_flat_split_states_v1 import mode_list
from diagnose_counterfactual_conditional_risk_v1 import read, save, sha

SEEDS = list(range(5910000, 5910064))
ALPHAS = [-1., -.5, 0., .5, 1., 1.5, 2.]
DIRECTIONS = ['dual', 'residual', 'random_sign']
CONTINUATIONS = ['alm_reset', 'alm_keep', 'nodual', 'pc', 'adam']
HORIZONS = [1, 2, 4, 8]
GROUPS = ['dwell', 'parameter_stall', 'union']


def state_at(b, h, u, best, x, v, method='alm'):
    state = cold.Local(b, x, v, method)
    state.h, state.u, state.best = h.copy(), u.copy(), best.copy()
    state.errors, state.moves = cold.base.score(best, x, v, np.zeros(4))
    return state


def modes(bank, x):
    return sorted(set(mode_list(bank.reshape(-1, 4), x))) if bank.size else []


def aggregate(scores):
    groups = defaultdict(list)
    for r in scores:
        groups[r['method'], r['grid']].append(r)
    result = []
    for (method, grid), rows in groups.items():
        assert len(rows) == 64 and len({r['seed'] for r in rows}) == 64
        result.append(dict(method=method, grid=grid, scope=rows[0]['scope'], tasks=64,
            policy_excess=float(np.mean([r['policy_excess'] for r in rows])),
            delta=float(np.mean([r['delta'] for r in rows])),
            pair_delta_means=[float(np.mean([r['pairs'][j]['delta'] for r in rows])) for j in range(2)],
            mass=float(np.mean([r['mass'] for r in rows])),
            shadow_steps=float(np.mean([r['shadow_steps'] for r in rows])),
            new_positive_modes=sum(len(r['new_positive_modes']) for r in rows),
            tasks_with_new_positive_modes=sum(bool(r['new_positive_modes']) for r in rows)))
    return result


def run(root, out):
    started = time.perf_counter()
    census = root / 'results/stagnant_local_states/development_v2'
    assert sha(census/'summary.json') == 'e96bffa8a65d57021c9302d525d4fe01e9c1d9eca2e92169b711e2fd96c8756e'
    cs = read(census/'summary.json')
    assert cs['passed'] and cs['tasks'] == 64
    for name, digest in cs['outputs_sha256'].items():
        assert sha(census/name) == digest, name
    locations = {r['seed']: r for r in read(census/'tasks.json')}
    raw_folder = root / 'results/certificate_activity_attribution/development'
    raw_manifest = read(raw_folder/'before_evaluation_manifest.json')
    assert sha(raw_folder/'rows.json') == raw_manifest['rows_sha256']
    assert sha(raw_folder/'protocol.json') == raw_manifest['protocol_sha256']
    original = {r['seed']: r for r in read(raw_folder/'rows.json') if r['method'] == 'credit_control_probe33'}
    frozen = read(root/'results/round_287_audit_v6.json')
    for name, digest in frozen['source_sha256'].items():
        assert sha(root/'work/experiments'/name) == digest, name
    source_names = ['diagnose_stagnant_trigger_exploration_v1.py', 'cold_stagnation_switch.py',
                    'batched_bp_discovery.py', 'posterior_confirmation_pipeline.py',
                    'diagnose_gradient_flat_split_states_v1.py', 'diagnose_counterfactual_conditional_risk_v1.py']
    sources = {n: sha(root/'work/experiments'/n) for n in source_names}
    docs = root/'outputs/ttt-pc-alm-research'
    save(out/'protocol.json', dict(seeds=SEEDS, groups=GROUPS, alphas=ALPHAS,
        directions=DIRECTIONS, continuations=CONTINUATIONS, horizons=HORIZONS,
        source_sha256=sources, frozen_source_seal_sha256=sha(root/'results/round_287_audit_v6.json'),
        design_sha256=sha(docs/'300_stagnant_trigger_exploration_protocol.md'),
        addendum_sha256=sha(docs/'300_execution_addendum_v1.md'), census_summary_sha256=sha(census/'summary.json'),
        random_seed_prefix=300911, random_seed_fields=['prefix', 'seed', 'phase', 'step', 'origin'],
        parameter_box=.12, new_pool_configs=69, background_configs=46,
        query_targets_accessed=False, reference_only_after_all_proposals_sealed=True,
        new_confirmation=False, resources_matched=False, grids=[257,129], batch_pairs=[[0,1],[2,3]]))
    proposal_rows, files = [], {}
    counts = {group: 0 for group in GROUPS}
    empty_tasks = {group: 0 for group in GROUPS}
    checks = defaultdict(int)
    max_norm_gap = 0.
    zero_u = zero_r = 0

    # Phase 1: only original support-side states and the fixed census are used.
    for seed in SEEDS:
        selected = {g: {tuple(a) for a in locations[seed]['locations'][g]} for g in GROUPS[:2]}
        selected['union'] = selected['dwell'] | selected['parameter_stall']
        loc = sorted(selected['union'])
        masks = {g: np.array([a in selected[g] for a in loc], dtype=bool) for g in GROUPS}
        assert np.array_equal(masks['union'], masks['dwell'] | masks['parameter_stall'])
        for g in GROUPS:
            counts[g] += len(selected[g])
            empty_tasks[g] += int(not selected[g])
        source = raw_folder/original[seed]['file']
        assert sha(source) == original[seed]['sha256'] == locations[seed]['source_sha256']
        with np.load(source, allow_pickle=False) as z:
            x, v = z['x_observed'], z['v_observed']
            b = np.array([z[('prefix' if p == 0 else 'anchor')+'_b'][t-1,o] for p,t,o in loc]).reshape(-1,4)
            best = np.array([z[('prefix' if p == 0 else 'anchor')+'_best'][t-1,o] for p,t,o in loc]).reshape(-1,4)
            expected_one = np.array([z[('prefix' if p == 0 else 'anchor')+'_b'][t,o] for p,t,o in loc]).reshape(-1,4)
            h = np.stack([z[('prefix' if p == 0 else 'anchor')+'_h'][t-1,:,o] for p,t,o in loc],axis=1) if loc else np.empty((4,0,len(x)))
            u = np.stack([z[('prefix' if p == 0 else 'anchor')+'_u'][t-1,:,o] for p,t,o in loc],axis=1) if loc else np.empty_like(h)
        residual = np.array([h[j]-cold.base.g((x if j == 0 else h[j-1])+b[:,j,None]) for j in range(4)])
        unorm = np.sqrt(np.sum(u*u,axis=(0,2)))
        rnorm = np.sqrt(np.sum(residual*residual,axis=(0,2)))
        zero_u += int(np.count_nonzero(unorm == 0))
        zero_r += int(np.count_nonzero(rnorm == 0))
        scale = np.divide(unorm,rnorm,out=np.zeros_like(unorm),where=(rnorm > 0)&(unorm > 0))
        signs = np.stack([np.random.default_rng(np.random.SeedSequence([300911,seed,p,t,o])).choice([-1.,1.],size=(4,len(x)))
                          for p,t,o in loc],axis=1) if loc else np.empty_like(u)
        directions = {'dual':u, 'residual':residual*scale[None,:,None], 'random_sign':u*signs}
        assert np.array_equal(abs(directions['random_sign']),abs(u))
        valid = (rnorm > 0)&(unorm > 0)
        gap = abs(np.sqrt(np.sum(directions['residual']**2,axis=(0,2)))-unorm)/np.maximum(1.,unorm)
        if np.any(valid):
            max_norm_gap = max(max_norm_gap,float(np.max(gap[valid])))
        assert max_norm_gap < 1e-12
        apaths, ahpaths, paths, hpaths, upaths, timings = {}, {}, {}, {}, {}, {}
        with discovery_box(.12):
            for family in DIRECTIONS:
                bb, hh, elapsed = [], [], []
                for alpha in ALPHAS:
                    tick = time.perf_counter()
                    if loc:
                        state = state_at(b,h,alpha*directions[family],best,x,v)
                        state.step()
                        bb.append(state.b.copy())
                        hh.append(state.h.copy())
                    else:
                        bb.append(b.copy()); hh.append(h.copy())
                    elapsed.append(time.perf_counter()-tick)
                apaths[family], ahpaths[family] = np.array(bb), np.array(hh)
                timings['amplitude_'+family] = elapsed
            for family in CONTINUATIONS[:-1]:
                method = 'alm' if family.startswith('alm') else family
                bb, hh, uu, elapsed = [], [], [], []
                if loc:
                    state = state_at(b,h,u if family == 'alm_keep' else np.zeros_like(u),best,x,v,method)
                for _ in range(8):
                    tick = time.perf_counter()
                    if loc:
                        state.step()
                        bb.append(state.b.copy()); hh.append(state.h.copy()); uu.append(state.u.copy())
                    else:
                        bb.append(b.copy()); hh.append(h.copy()); uu.append(u.copy())
                    elapsed.append(time.perf_counter()-tick)
                paths[family], hpaths[family], upaths[family] = np.array(bb), np.array(hh), np.array(uu)
                timings[family] = elapsed
            tick = time.perf_counter()
            if loc:
                _, bp_trace, _ = cold.run_bp(b,x,v,'adam',8,trace=True)
                assert bp_trace['b'].shape == (9,len(b),4)
                assert bp_trace['roles'].tolist() == [True]*8+[False]
                paths['adam'] = bp_trace['b'][1:]
            else:
                paths['adam'] = np.empty((8,0,4))
            timings['adam_full8_trace'] = time.perf_counter()-tick
        for family in DIRECTIONS:
            assert apaths[family][2].tobytes() == paths['alm_reset'][0].tobytes(), (seed,family,'alpha0_b')
            assert ahpaths[family][2].tobytes() == hpaths['alm_reset'][0].tobytes(), (seed,family,'alpha0_h')
            checks['alpha0_bh'] += 1
        assert apaths['dual'][4].tobytes() == expected_one.tobytes(), (seed,'alpha1')
        assert paths['alm_keep'][0].tobytes() == expected_one.tobytes(), (seed,'keep1')
        assert paths['alm_reset'][0].tobytes() == paths['nodual'][0].tobytes(), (seed,'nodual1_b')
        assert hpaths['alm_reset'][0].tobytes() == hpaths['nodual'][0].tobytes(), (seed,'nodual1_h')
        for name in ['alpha1', 'keep1', 'reset_nodual_bh']:
            checks[name] += 1
        methods, individual = {}, {}
        for group, mask in masks.items():
            n = int(mask.sum())
            for family in DIRECTIONS:
                union = set()
                for ai, alpha in enumerate(ALPHAS):
                    values = modes(apaths[family][ai,mask],x)
                    individual[f'{group}__{family}_a{ai}'] = dict(group=group,family=family,alpha=alpha,modes=values,shadow_steps=n)
                    union.update(values)
                assert union == set(modes(apaths[family][:,mask],x))
                checks['amplitude_union_alias'] += 1
                methods[f'{group}__{family}_union'] = dict(group=group,family=family,horizon=None,modes=sorted(union),shadow_steps=7*n)
            for family in CONTINUATIONS:
                for horizon in HORIZONS:
                    methods[f'{group}__{family}_{horizon}'] = dict(group=group,family=family,horizon=horizon,
                        modes=modes(paths[family][:horizon,mask],x),shadow_steps=n*horizon)
        assert len(methods) == 69 and len(individual) == 63
        suffixes = [k.split('__',1)[1] for k in methods if k.startswith('union__')]
        for suffix in suffixes:
            assert set(methods['union__'+suffix]['modes']) == set(methods['dwell__'+suffix]['modes']) | set(methods['parameter_stall__'+suffix]['modes'])
            checks['selection_union_alias'] += 1
        arrays = dict(locations=np.array(loc,dtype=int).reshape(-1,3),x_observed=x,v_observed=v,
                      initial_b=b,initial_h=h,initial_u=u,initial_best=best)
        arrays.update({f'mask_{g}':m for g,m in masks.items()})
        arrays.update({f'amplitude_{f}_b':a for f,a in apaths.items()})
        arrays.update({f'amplitude_{f}_h':a for f,a in ahpaths.items()})
        arrays.update({f'direction_{f}':a for f,a in directions.items()})
        arrays.update({f'continuation_{f}_b':a for f,a in paths.items()})
        arrays.update({f'continuation_{f}_h':a for f,a in hpaths.items()})
        arrays.update({f'continuation_{f}_u':a for f,a in upaths.items()})
        file = f'{seed}_proposals.npz'
        with (out/file).open('xb') as stream:
            np.savez_compressed(stream,**arrays)
        files[file] = sha(out/file)
        proposal_rows.append(dict(seed=seed,locations=loc,selection_counts={g:len(selected[g]) for g in GROUPS},
            original_positive_modes=original[seed]['metadata']['positive_modes'],source_sha256=sha(source),
            methods=methods,individual_alphas=individual,file=file,sha256=files[file],
            union_batched_diagnostic_timings=timings,subset_timings_measured=False))
        if (seed-SEEDS[0]+1)%16 == 0:
            print(dict(phase='support_only',tasks=seed-SEEDS[0]+1,counts=counts),flush=True)
    assert counts['dwell'] == 2068 and counts['parameter_stall'] == 38
    save(out/'proposals.json',proposal_rows)
    save(out/'selftests.json',dict(passed=True,cases=dict(checks),counts=counts,empty_tasks=empty_tasks,
        max_norm_gap=max_norm_gap,zero_multiplier_states=zero_u,zero_residual_states=zero_r))
    for name in ['protocol.json','proposals.json','selftests.json']:
        files[name] = sha(out/name)
    save(out/'before_reference_manifest.json',dict(files_sha256=files,tasks=64,new_configs=69,
        reference_accessed=False,query_targets_accessed=False))
    print(dict(phase='all_proposals_sealed',counts=counts,checks=dict(checks)),flush=True)

    # Phase 2: cached full-posterior regional moments are evaluation only.
    cached = root/'results/counterfactual_conditional_risk/development_v2'
    assert sha(cached/'summary.json') == '95852bf236596935f7ac0dd3c6fe389118b6976b5b2f368e09cc8ac37325a3ec'
    refsummary = read(cached/'summary.json')
    assert refsummary['passed']
    for name,digest in refsummary['outputs_sha256'].items():
        assert sha(cached/name) == digest,name
    for name,digest in read(cached/'input_hashes.json').items():
        assert sha(root/name) == digest,name
    previous = root/'results/dual_amplitude_paths/development_v1'
    assert sha(previous/'summary.json') == '1a90bc14baebe225c2501f8886e1e5e59077031fa894a0080a5f3a27d95a972c'
    prevsummary = read(previous/'summary.json')
    for name,digest in prevsummary['outputs_sha256'].items():
        assert sha(previous/name) == digest,name
    background = [dict(r,method='background__'+r['method'],scope='298_background_not_trigger_matched') for r in read(previous/'scores.json')]
    assert len(background) == 5888
    moments = root/'results/confirmation_conditional_risk/moments'
    task_moments = {r['seed']:r for r in read(moments/'tasks.json')}
    scores, max_scalar_gap = [], 0.
    for row in proposal_rows:
        task = task_moments[row['seed']]
        assert sha(moments/task['file']) == task['sha256']
        with np.load(moments/task['file'],allow_pickle=False) as z:
            vol,means = z['volumes'],z['means']
            assert np.array_equal(z['q257'],np.linspace(0,1,257))
        assert np.all(vol > 0) and means.shape == (len(vol),4,257)
        keys = task['keys']; keyset = set(keys)
        weights = vol/vol.sum()
        full = np.einsum('k,kbq->bq',weights,means)
        old = set(row['original_positive_modes'])
        assert old and old <= keyset
        def mixture(pool):
            mask = np.array([k in pool for k in keys])
            mass = float(weights[mask].sum())
            assert mass > 0
            return mass,np.einsum('k,kbq->bq',weights*mask/mass,means)
        oldmass,oldmu = mixture(old)
        for name,method in row['methods'].items():
            extra = (set(method['modes']) & keyset)-old
            mass,mu = mixture(old|extra)
            for grid in [257,129]:
                idx = np.arange(257) if grid == 257 else np.arange(0,257,2)
                estimates = []
                for a,b in [(0,1),(2,3)]:
                    bias = float(np.mean((mu[a,idx]-full[a,idx])*(mu[b,idx]-full[b,idx])))
                    base = float(np.mean((oldmu[a,idx]-full[a,idx])*(oldmu[b,idx]-full[b,idx])))
                    scalar = math.fsum(float(c)*float(d) for c,d in zip(mu[a,idx]-full[a,idx],mu[b,idx]-full[b,idx]))/grid
                    max_scalar_gap = max(max_scalar_gap,abs(bias-scalar))
                    estimates.append(dict(pair=[a,b],policy_excess=bias,delta=bias-base))
                scores.append(dict(seed=row['seed'],method=name,scope='300_same_stagnation_locations',grid=grid,
                    mass=mass,mass_added=mass-oldmass,new_positive_modes=sorted(extra),shadow_steps=method['shadow_steps'],
                    pairs=estimates,policy_excess=float(np.mean([e['policy_excess'] for e in estimates])),
                    delta=float(np.mean([e['delta'] for e in estimates]))))
                if row['selection_counts'][method['group']] == 0:
                    assert not extra and scores[-1]['delta'] == 0 and method['shadow_steps'] == 0
        if (row['seed']-SEEDS[0]+1)%16 == 0:
            print(dict(phase='evaluation',tasks=row['seed']-SEEDS[0]+1),flush=True)
    assert max_scalar_gap < 1e-12 and len(scores) == 8832
    scores += background
    assert len(scores) == 14720
    aggregates = aggregate(scores)
    assert len(aggregates) == 230
    lookup = {(r['method'],r['grid']):r for r in aggregates}
    max_replay = 0.
    for r in read(previous/'aggregates.json'):
        rr = lookup['background__'+r['method'],r['grid']]
        for metric in ['policy_excess','delta','mass','shadow_steps']:
            max_replay = max(max_replay,abs(rr[metric]-r[metric]))
    assert max_replay == 0
    save(out/'scores.json',scores)
    save(out/'aggregates.json',aggregates)
    for name,digest in files.items():
        assert sha(out/name) == digest,name
    for name,digest in sources.items():
        assert sha(root/'work/experiments'/name) == digest,name
    summary = dict(passed=True,tasks=64,new_configs=69,background_configs=46,grid_rows=len(scores),
        counts=counts,empty_tasks=empty_tasks,selftest_cases=dict(checks),max_norm_gap=max_norm_gap,
        max_scalar_gap=max_scalar_gap,max_background_replay_gap=max_replay,seconds=time.perf_counter()-started,
        query_targets_accessed=False,reference_only_after_all_proposals_sealed=True,numerical_reference_only=True,
        resources_matched=False,new_confirmation=False,core_research_goal_complete=False,
        cached_reference_summary_sha256=sha(cached/'summary.json'),background_summary_sha256=sha(previous/'summary.json'),
        outputs_sha256={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()})
    save(out/'summary.json',summary)
    print({k:v for k,v in summary.items() if k != 'outputs_sha256'},flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,required=True)
    root = parser.parse_args().project.resolve()
    out = root/'results/stagnant_trigger_exploration/development_v1'
    out.mkdir(parents=True,exist_ok=False)
    try:
        run(root,out)
    except BaseException as exc:
        save(out/'failure.json',dict(error_type=type(exc).__name__,message=str(exc),no_automatic_retry=True))
        raise


if __name__ == '__main__':
    main()
