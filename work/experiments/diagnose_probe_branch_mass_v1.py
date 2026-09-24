"""289 supplementary OLD64 numerical posterior mass and discovery trajectories."""
import argparse
from datetime import datetime, timezone
from fractions import Fraction as F
import math
from pathlib import Path
import time

import numpy as np
from diagnose_gradient_flat_split_states_v1 import METHODS, mode_list, read, save, sha


def describe(values):
    a = np.array(values, dtype=float)
    return dict(mean=float(a.mean()), minimum=float(a.min()), median=float(np.median(a)),
                maximum=float(a.max()), positive_tasks=int(np.sum(a > 0)))


def selftests():
    masses = [F(1, 3)]*3
    values = [F(0), F(1, 2), F(1)]
    full = sum(w*v for w, v in zip(masses, values))
    subsets = [{i for i in range(3) if bits & (1 << i)} for bits in range(1, 8)]
    for a in subsets:
        ca = sum(masses[i] for i in a)
        pa = [masses[i]/ca if i in a else F(0) for i in range(3)]
        mu = sum(p*v for p, v in zip(pa, values))
        assert sum(abs(p-w) for p, w in zip(pa, masses))/2 == 1-ca
        assert abs(mu-full) <= 1-ca
        excess = sum(w*((mu-v)**2-(full-v)**2) for w, v in zip(masses, values))
        assert excess == (mu-full)**2
        for b in subsets:
            cb = sum(masses[i] for i in b)
            pb = [masses[i]/cb if i in b else F(0) for i in range(3)]
            assert sum(abs(p-q) for p, q in zip(pa, pb))/2 == 1-sum(masses[i] for i in a & b)/max(ca, cb)
    # Strictly more coverage can worsen actual conditional prediction risk.
    assert (F(1, 2)-full)**2 == 0 < (F(1, 4)-full)**2
    return dict(passed=True, subset_cases=7, pair_cases=49,
                higher_coverage_worse_risk_counterexample=True, query_targets_accessed=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    root = parser.parse_args().project.resolve()
    out = root/'results/gradient_flat_split_states/branch_mass_v1'
    out.mkdir(parents=True, exist_ok=False)
    begin = time.perf_counter()
    tests = selftests()
    save(out/'selftests.json', tests)
    diagnosis = root/'results/gradient_flat_split_states/development_v1'
    ds = read(diagnosis/'summary.json')
    assert ds['passed'] and ds['tasks'] == 64 and ds['origins'] == 2112
    for name, digest in ds['outputs_sha256'].items():
        assert sha(diagnosis/name) == digest
    tasks = read(diagnosis/'tasks.json')
    original_files = read(diagnosis/'input_hashes.json')
    base = root/'results/confirmation_conditional_risk'
    reference, audit, boundary = [base/name for name in ['reference', 'reference_audit', 'boundary_certificates']]
    rs, aus, ap, bs = [read(path) for path in [reference/'summary.json', audit/'summary.json', audit/'protocol.json', boundary/'summary.json']]
    assert rs['passed'] and aus['passed'] and bs['passed'] and bs['unresolved'] == 0
    assert aus['complete_up_to_certified_zero_volume'] == 64
    assert ap['reference_summary_sha256'] == sha(reference/'summary.json')
    assert ap['boundary_summary_sha256'] == sha(boundary/'summary.json')
    assert aus['protocol_sha256'] == sha(audit/'protocol.json')
    assert rs['coverage_sha256'] == sha(reference/'coverage.json')
    assert rs['files_sha256'] == sha(reference/'files.json')
    assert aus['coverage_sha256'] == sha(audit/'coverage.json')
    assert aus['mass_summary_sha256'] == sha(audit/'mass_summary.json')
    assert bs['certificates_sha256'] == sha(boundary/'certificates.json')
    for name, digest in ap['source_sha256'].items():
        assert sha(root/'work/experiments'/name) == digest, name
    audit_rows = {r['seed']: r for r in read(audit/'coverage.json')}
    ref_rows = {r['seed']: r for r in read(reference/'coverage.json')}
    ref_files = read(reference/'files.json')
    expected_coverage = {}
    rows = {}
    parent_paths = [diagnosis/'summary.json', audit/'summary.json', reference/'summary.json',
                    boundary/'summary.json', boundary/'certificates.json']
    for group in sorted({value[0] for value in METHODS.values()}):
        folder = root/'results'/group
        rows[group] = {(r['seed'], r['method']): r for r in read(folder/'development/rows.json')}
        old_audit = read(folder/'audit/summary.json')
        assert old_audit['passed']
        assert old_audit['coverage_sha256'] == sha(folder/'audit/coverage.json')
        expected_coverage[group] = {(r['seed'], r['method']): r for r in read(folder/'audit/coverage.json')}
        parent_paths += [folder/'audit/summary.json', folder/'development/rows.json']
    sources = ['work/experiments/diagnose_probe_branch_mass_v1.py',
               'work/experiments/diagnose_gradient_flat_split_states_v1.py',
               'outputs/ttt-pc-alm-research/289_branch_mass_followup_protocol.md']
    save(out/'protocol.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
         scope='Supplementary OLD64 support-only numerical mass, not query risk or an online oracle',
         seeds=list(range(5910000, 5910064)), methods=METHODS, query_arrays_decoded=False,
         original_reference_direct_complete_tasks=rs['complete_references'],
         reference_complete_after_zero_volume_audit=64, numerical_volumes_not_exact_real=True,
         parent_sha256={str(p.relative_to(root)).replace('\\', '/'): sha(p) for p in parent_paths},
         source_sha256={name: sha(root/name) for name in sources}))
    results = []
    reference_hashes = {}
    max_volume_gap = 0.
    for task in tasks:
        seed = task['seed']
        refrow = ref_rows[seed]
        path = reference/refrow['file']
        assert sha(path) == refrow['sha256'] == ref_files[refrow['file']]
        data = read(path)
        assert data['reference']['enumeration_completed']
        assert audit_rows[seed]['complete_up_to_certified_zero_volume']
        refset = {r['pattern']: r['volume'] for r in data['reference']['positive_regions']}
        assert len(refset) == audit_rows[seed]['positive_regions']
        total = math.fsum(refset.values())
        assert total > 0 and math.isclose(total, audit_rows[seed]['total_numerical_volume'], rel_tol=1e-12)
        reference_hashes[str(path.relative_to(root)).replace('\\', '/')] = sha(path)
        for key, item in data['geometry'].items():
            path = reference/item['file']
            assert sha(path) == item['sha256'] == ref_files[item['file']]
            reference_hashes[str(path.relative_to(root)).replace('\\', '/')] = sha(path)
            with np.load(path, allow_pickle=False) as z:
                vol = float(np.abs(np.linalg.det(z['facets']-z['interior'])).sum()/24*np.prod(z['scale']))
                assert float(z['volume']) == refset[key]
                gap = abs(vol/refset[key]-1)
                assert gap < 1e-8
                max_volume_gap = max(max_volume_gap, gap)
        def mass(keys):
            assert set(keys) <= refset.keys()
            return math.fsum(refset[key] for key in sorted(keys))/total
        pools = {role: set(keys) for role, keys in task['positive_modes'].items()}
        sets = {name: set(keys) for name, keys in task['sets'].items()}
        sets.update(alm_bp_union=pools['alm']|pools['bp'],
                    alm_bp_intersection=pools['alm']&pools['bp'],
                    all_three_union=pools['alm']|pools['bp']|pools['nodual'])
        coverage = {role: mass(keys) for role, keys in pools.items()}
        fractions = {name: mass(keys) for name, keys in sets.items()}
        curves, origins = {}, {}
        for role, (group, method) in METHODS.items():
            row = rows[group][seed, method]
            relative = f'results/{group}/development/{row["file"]}'
            assert sha(root/relative) == row['sha256'] == original_files[relative]
            with np.load(root/relative, allow_pickle=False) as z:
                x, v = z['x_observed'], z['v_observed']
                np.testing.assert_array_equal(x, np.array(data['x_observed']))
                np.testing.assert_array_equal(v, np.array(data['v_observed']))
                if role in ['alm', 'nodual']:
                    traces = [z['prefix_b'][i] for i in range(1, 33)]
                    traces += [z['anchor_b'][i] for i in range(1, 33)]
                    individual = [np.r_[z['prefix_b'][:, i], z['anchor_b'][1:, 0]] if i == 0 else z['prefix_b'][:, i] for i in range(33)]
                elif role == 'bp':
                    traces = [z['bp_b'][i] for i in range(1, 241)]
                    individual = [z['bp_b'][:, i] for i in range(33)]
                else:
                    traces, individual = [], []
            seen = pools['archive'].copy()
            curve = [mass(seen)]
            for trace in traces:
                seen.update(set(mode_list(trace, x)) & refset.keys())
                curve.append(mass(seen))
            assert seen == pools[role], (seed, role)
            previous = expected_coverage[group][seed, method]
            assert set(previous['keys']) == pools[role]
            assert math.isclose(previous['numerical_mass'], coverage[role], rel_tol=1e-12, abs_tol=1e-12)
            curves[role] = curve
            origins[role] = [mass(set(mode_list(trace, x)) & refset.keys()) for trace in individual]
        assert curves['alm'][:2] == curves['nodual'][:2]
        denominator = max(coverage['alm'], coverage['bp'])
        tv = 1-fractions['alm_bp_intersection']/denominator if min(coverage['alm'], coverage['bp']) > 0 else None
        results.append(dict(seed=seed, coverage=coverage, set_mass=fractions,
                            alm_bp_truncated_tv=tv, round_coverage=curves,
                            origin_reachable_mass=origins,
                            bounds={role: (1-value)**2 for role, value in coverage.items()}))
        if len(results) % 16 == 0:
            print(dict(tasks=len(results), total=64, seconds=time.perf_counter()-begin), flush=True)
    save(out/'tasks.json', results)
    save(out/'reference_hashes.json', reference_hashes)
    summary = dict(passed=True, scope='Numerical posterior mass diagnostic, not task advantage', tasks=64,
                   method_coverage={role: describe([r['coverage'][role] for r in results]) for role in METHODS},
                   set_mass={name: describe([r['set_mass'][name] for r in results]) for name in results[0]['set_mass']},
                   alm_bp_tv=describe([r['alm_bp_truncated_tv'] for r in results if r['alm_bp_truncated_tv'] is not None]),
                   undefined_tv_empty_pool_tasks=sum(r['alm_bp_truncated_tv'] is None for r in results),
                   mean_round_coverage={role: np.mean([r['round_coverage'][role] for r in results], axis=0).tolist() for role in METHODS},
                   maximum_relative_volume_reconstruction_gap=max_volume_gap,
                   reference_files_verified=len(reference_hashes), seconds=time.perf_counter()-begin,
                   query_arrays_decoded=False, query_targets_accessed=False, adaptation_rerun=False,
                   protocol_sha256=sha(out/'protocol.json'),
                   outputs_sha256={name: sha(out/name) for name in ['selftests.json', 'tasks.json', 'reference_hashes.json']})
    save(out/'summary.json', summary)
    print({k: v for k, v in summary.items() if k not in ['mean_round_coverage', 'set_mass']}, flush=True)
    print('selected_set_mass', {k: summary['set_mass'][k] for k in ['alm_only_vs_bp', 'bp_only_vs_alm', 'alm_only_vs_both', 'alm_bp_union']}, flush=True)


if __name__ == '__main__':
    main()
