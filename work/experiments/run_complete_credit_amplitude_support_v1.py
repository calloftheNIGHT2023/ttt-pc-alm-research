"""308: frozen old-support full-sweep event enumeration, no reference scoring.

The support phase ends by sealing all proposals. Geometry evaluation belongs
to a separate later script. CLI --first-states is a deterministic smoke run,
not a filtered effectiveness experiment.
"""
import argparse
from collections import Counter
from fractions import Fraction as F
import gzip
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np
import cold_stagnation_switch as cold
from posterior_confirmation_pipeline import discovery_box
from diagnose_gradient_flat_split_states_v1 import mode_list
from dual_amplitude_local_audit_v1 import scalar_step
from exact_quadratic_events_v1 import Root, between, encode
from complete_credit_amplitude_events_v1 import rational_state, sweep, sweep_partition
from complete_credit_rational_reference_v1 import direct_step

SEEDS = list(range(5910000, 5910064))
ALPHAS = [F(-1), F(-1, 2), F(0), F(1, 2), F(1), F(3, 2), F(2)]
FAMILIES = ['dual', 'residual', 'random_sign']
CONTROL_FAMILIES = ['alm_keep', 'alm_reset', 'nodual', 'pc', 'adam',
                    'adam_perturb_001', 'adam_perturb_004']


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def save(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(encode(value), stream, ensure_ascii=False, indent=2, allow_nan=False)


def save_compressed(path, value):
    payload = json.dumps(encode(value), ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8')
    with gzip.open(path, 'xb') as stream:
        stream.write(payload)
    return dict(serialized_json_bytes=len(payload), gzip_bytes=path.stat().st_size,
                scope='archive size, NOT live or peak solver memory')


def hexmode(code):
    return bytes(int(c) for c in code).hex()


def evaluate(values, point):
    return [evaluate(v, point) for v in values] if isinstance(values, list) else values.at(point)


def reference_check(result, state, point):
    ref = direct_step(state, point)
    for key in ['b', 'h', 'forward']:
        assert evaluate(result[key], point) == ref[key], (key, str(point))
    assert result['codes'] == ref['codes'] and result['mode'] == ref['mode']


def verify_partition(result, state):
    started = time.perf_counter()
    checks = Counter()
    for cell in result['cells']:
        for q in [between(cell['low'], cell['witness']), between(cell['witness'], cell['high'])]:
            reference_check(cell, state, q)
            checks['rational_interior_checks'] += 1
    for row in result['boundaries']:
        if isinstance(row['point'], F):
            reference_check(row, state, row['point'])
            checks['rational_boundary_checks'] += 1
        else:
            checks['algebraic_boundaries_not_independently_scalar_checked'] += 1
    return dict(checks=dict(checks), seconds=time.perf_counter() - started)


def inputs(root):
    events = root / 'results/support_boundary_events/development_v1'
    assert sha(events / 'summary.json') == 'cf8f82aca2f716a24e37fdbd5a83068f3ae030d46890c9906cdbaced4ca9de2f'
    selection_summary = read(events / 'summary.json')
    for name, digest in selection_summary['outputs_sha256'].items():
        assert sha(events / name) == digest
    selections = {r['seed']: r for r in read(events / 'selections.json')}
    raw = root / 'results/certificate_activity_attribution/development'
    manifest = read(raw / 'before_evaluation_manifest.json')
    assert sha(raw / 'rows.json') == manifest['rows_sha256']
    assert sha(raw / 'protocol.json') == manifest['protocol_sha256']
    originals = {r['seed']: r for r in read(raw / 'rows.json') if r['method'] == 'credit_control_probe33'}
    archive = root / 'results/observable_trap_continuation/development_v1'
    old_summary = read(archive / 'summary.json')
    assert old_summary['passed'] and old_summary['counts']['forward_stasis'] == 131
    # Hash all outputs without interpreting risk or geometry fields.
    for name, digest in old_summary['outputs_sha256'].items():
        assert sha(archive / name) == digest
    oldrows = {r['seed']: r for r in read(archive / 'proposals.json')}
    frozen = root / 'results/round_287_audit_v6.json'
    for name, digest in read(frozen)['source_sha256'].items():
        assert sha(root / 'work/experiments' / name) == digest
    names = [(events / 'summary.json'), (raw / 'before_evaluation_manifest.json'),
             (archive / 'summary.json'), frozen]
    digests = {str(p.relative_to(root)): sha(p) for p in names}
    result = []
    task_manifest = []
    for seed in SEEDS:
        locs = sorted(tuple(p) for p in selections[seed]['locations']['forward_stasis__all'])
        task_manifest.append(dict(seed=seed, locations=locs, selected_states=len(locs)))
        if not locs:
            continue
        source = raw / originals[seed]['file']
        assert sha(source) == originals[seed]['sha256'] == selections[seed]['source_sha256'] == oldrows[seed]['source_sha256']
        archived = archive / oldrows[seed]['file']
        assert sha(archived) == oldrows[seed]['sha256']
        digests[str(source.relative_to(root))] = sha(source)
        digests[str(archived.relative_to(root))] = sha(archived)
        with np.load(source, allow_pickle=False) as z, np.load(archived, allow_pickle=False) as az:
            x, v = z['x_observed'], z['v_observed']
            assert np.array_equal(x, az['x_observed']) and np.array_equal(v, az['v_observed'])
            old_index = {tuple(p): i for i, p in enumerate(az['locations'].tolist())}
            assert set(locs) == {p for p, i in old_index.items() if az['mask_forward_stasis'][i]}
            for phase, step, origin in locs:
                prefix = 'prefix' if phase == 0 else 'anchor'
                b = z[prefix + '_b'][step - 1, origin].copy()
                h = z[prefix + '_h'][step - 1, :, origin].copy()
                u = z[prefix + '_u'][step - 1, :, origin].copy()
                best = z[prefix + '_best'][step - 1, origin].copy()
                idx = old_index[(phase, step, origin)]
                assert np.array_equal(b, az['initial_b'][idx])
                assert np.array_equal(h, az['initial_h'][:, idx])
                assert np.array_equal(u, az['initial_u'][:, idx])
                assert np.array_equal(best, az['initial_best'][idx])
                residual = np.stack([h[j] - cold.base.g((x if j == 0 else h[j-1]) + b[j]) for j in range(4)])
                unorm = np.sqrt(np.sum(np.sum(u * u, axis=1), axis=0))
                rnorm = np.sqrt(np.sum(np.sum(residual * residual, axis=1), axis=0))
                scale = unorm / rnorm if unorm > 0 and rnorm > 0 else 0.
                signs = np.random.default_rng(np.random.SeedSequence([303911, seed, phase, step, origin, 0])).choice([-1., 1.], size=u.shape)
                directions = dict(dual=u, residual=residual * scale, random_sign=u * signs)
                assert np.array_equal(directions['residual'], az['residual_direction'][:, idx])
                assert np.array_equal(directions['random_sign'], az['alm_random_sign_initial_u'][:, idx])
                assert np.array_equal(directions['dual'], az['alm_keep_initial_u'][:, idx])
                controls = {}
                for family in CONTROL_FAMILIES:
                    trace = az[family + '_b'][:, idx].copy()
                    assert trace.shape == (65, 4)
                    horizons = [1, 2, 4, 8, 64] if family.startswith('adam') else [64]
                    for horizon in horizons:
                        controls[f'{family}_{horizon}'] = dict(
                            modes=sorted(set(mode_list(trace[:horizon + 1], x))),
                            state_steps=horizon, retained_initial_point=True,
                            global_jacobian_rows=horizon if family.startswith('adam') else 0,
                            origin='verified same-location 303 stored trace',
                            timing_not_reusable_for_subselected_batch=True)
                result.append(dict(seed=seed, location=(phase, step, origin), b=b, h=h, u=u, best=best,
                    x=x.copy(), v=v.copy(), directions=directions, controls=controls,
                    expected_b=z[prefix + '_b'][step, origin].copy(),
                    expected_h=z[prefix + '_h'][step, :, origin].copy()))
    assert len(result) == 131 and len(task_manifest) == 64
    return result, task_manifest, digests


def seven_points(item, family, state):
    rows = []
    with discovery_box(.12):
        for alpha in ALPHAS:
            exact = sweep(state, alpha, strict=False)
            reference_check(exact, state, alpha)
            start = time.perf_counter()
            local = cold.Local(item['b'][None], item['x'], item['v'], 'alm')
            local.h = item['h'][:, None].copy()
            local.u = (float(alpha) * item['directions'][family])[:, None].copy()
            local.step()
            seconds = time.perf_counter() - start
            scalar = scalar_step(item['b'], item['h'], item['directions'][family], item['x'], item['v'], float(alpha))
            expected = {k: np.array(evaluate(exact[k], alpha), dtype=float) for k in ['b', 'h']}
            gap_prod = max(float(np.max(abs(local.b[0] - expected['b']))), float(np.max(abs(local.h[:, 0] - expected['h']))))
            gap_scalar = max(float(np.max(abs(scalar[k] - expected[k]))) for k in ['b', 'h'])
            if family == 'dual' and alpha == 1:
                assert np.array_equal(local.b[0], item['expected_b'])
                assert np.array_equal(local.h[:, 0], item['expected_h'])
            rows.append(dict(alpha=alpha, exact_b=exact['b'], exact_h=exact['h'],
                exact_mode=hexmode(exact['mode']), float_b=local.b[0].tolist(), float_h=local.h[:, 0].tolist(),
                float_mode=mode_list(local.b, item['x'])[0], production_seconds=seconds,
                max_production_gap=gap_prod, max_scalar_gap=gap_scalar,
                float_value_discrepancy=max(gap_prod, gap_scalar) > 1e-10,
                float_mode_discrepancy=hexmode(exact['mode']) != mode_list(local.b, item['x'])[0]))
    return rows


def run(root, out, first_states):
    started = time.perf_counter()
    validation = root / 'results/complete_credit_amplitude_events/tests_attempt04'
    vs, vp = read(validation / 'summary.json'), read(validation / 'protocol.json')
    assert vs['exact_tests_passed'] and vs['floating_checks_passed']
    assert len(vs['fixtures']) == 5
    for name, digest in vs['outputs_sha256'].items():
        assert sha(validation / name) == digest
    for name, digest in vp['source_sha256'].items():
        assert sha(root / 'work/experiments' / name) == digest
    items, task_manifest, input_hashes = inputs(root)
    docs = root / 'outputs/ttt-pc-alm-research'
    source_names = list(vp['source_sha256']) + [Path(__file__).name, 'diagnose_gradient_flat_split_states_v1.py']
    save(out / 'protocol.json', dict(source_sha256={n: sha(root / 'work/experiments' / n) for n in source_names},
         input_sha256=input_hashes, validation_summary_sha256=sha(validation / 'summary.json'),
         design_sha256={n: sha(docs / n) for n in ['308_complete_credit_amplitude_events_protocol.md', '308_execution_addendum_v1.md']},
         seeds=SEEDS, tasks=64, full_selected_states=131, families=FAMILIES, alphas=ALPHAS,
         first_states=first_states, smoke_only=first_states is not None,
         max_segments=4096, comparison_limit=1000000, bound=.12, trust=.01, eps=.001,
         query_targets_accessed=False, geometry_or_posterior_accessed=False,
         new_confirmation=False, resources_matched=False, global_bp_in_candidate=False,
         independent_exact_checks_separately_timed=True, task_manifest=task_manifest))
    if first_states is not None:
        items = items[:first_states]
    rows, files = [], {}
    for index, item in enumerate(items):
        seed, location = item['seed'], item['location']
        for family in FAMILIES:
            name = f'{seed}_{location[0]}_{location[1]}_{location[2]}_{family}'
            tick = time.perf_counter()
            state = rational_state(item['b'], item['h'], item['directions'][family], item['x'], item['v'])
            exact = sweep_partition(state)
            audit = verify_partition(exact, state)
            seven = seven_points(item, family, state)
            open_modes = sorted({hexmode(cell['mode']) for cell in exact['cells']})
            boundary_modes = sorted({hexmode(row['mode']) for row in exact['boundaries']})
            fixed_modes = sorted({row['exact_mode'] for row in seven})
            complete_modes = set(open_modes) | set(boundary_modes)
            if exact['coverage_complete']:
                assert set(fixed_modes) <= complete_modes
            payload = dict(state=state, partition=exact, independent_check=audit, seven_points=seven,
                           controls=item['controls'], seed=seed, location=location, family=family)
            filename = name + '.json.gz'
            sizes = save_compressed(out / filename, payload)
            files[filename] = sha(out / filename)
            row = dict(seed=seed, location=location, family=family, file=filename, sha256=files[filename],
                cells=len(exact['cells']), boundaries=len(exact['boundaries']),
                irrational_boundaries=sum(isinstance(v['point'], Root) for v in exact['boundaries']),
                coverage_complete=exact['coverage_complete'], open_intervals_complete=exact['open_intervals_complete'],
                pending_intervals=len(exact['pending']), limit_reason=exact['limit_reason'],
                comparisons=exact['polynomial_root_comparisons'], search_seconds=exact['seconds'],
                independent_checks=audit, full_diagnostic_seconds=time.perf_counter() - tick,
                sizes=sizes, open_modes=open_modes, boundary_modes=boundary_modes,
                fixed_exact_modes=fixed_modes, fixed_float_modes=sorted({r['float_mode'] for r in seven}),
                additional_open_modes_vs_fixed=sorted(set(open_modes) - set(fixed_modes)),
                boundary_only_modes=sorted(set(boundary_modes) - set(open_modes)),
                floating_value_discrepancies=sum(r['float_value_discrepancy'] for r in seven),
                floating_mode_discrepancies=sum(r['float_mode_discrepancy'] for r in seven),
                max_float_value_gap=max(max(r['max_production_gap'], r['max_scalar_gap']) for r in seven))
            rows.append(row)
            print(dict(state=index + 1, total_states=len(items), seed=seed, location=location, family=family,
                       cells=row['cells'], complete=row['coverage_complete'], search_seconds=row['search_seconds'],
                       extra_open_modes=len(row['additional_open_modes_vs_fixed']),
                       float_value_discrepancies=row['floating_value_discrepancies']), flush=True)
    save(out / 'rows.json', rows)
    files.update({name: sha(out / name) for name in ['protocol.json', 'rows.json']})
    save(out / 'before_geometry_manifest.json', dict(files_sha256=files, states=len(items), directions=len(rows),
         all_selected_states_processed=len(items) == 131, query_targets_accessed=False,
         geometry_or_posterior_accessed=False, smoke_only=first_states is not None))
    counts = Counter()
    for row in rows:
        counts['complete_directions'] += row['coverage_complete']
        counts['incomplete_directions'] += not row['coverage_complete']
        counts['cells'] += row['cells']
        counts['boundaries'] += row['boundaries']
        counts['directions_with_extra_open_modes'] += bool(row['additional_open_modes_vs_fixed'])
        counts['floating_value_discrepancies'] += row['floating_value_discrepancies']
        counts['floating_mode_discrepancies'] += row['floating_mode_discrepancies']
    summary = dict(support_execution_passed=True, states=len(items), directions=len(rows), counts=dict(counts),
         all_selected_states_processed=len(items) == 131, smoke_only=first_states is not None,
         geometry_evaluated=False, effectiveness_established=False, resources_matched=False,
         seconds=time.perf_counter() - started, manifest_sha256=sha(out / 'before_geometry_manifest.json'))
    save(out / 'summary.json', summary)
    print(summary, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--first-states', type=int)
    args = parser.parse_args()
    assert args.first_states is None or 1 <= args.first_states <= 131
    root = Path(__file__).resolve().parents[2]
    args.out.mkdir(parents=True, exist_ok=False)
    try:
        run(root, args.out, args.first_states)
    except Exception:
        save(args.out / 'failure.json', dict(traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
