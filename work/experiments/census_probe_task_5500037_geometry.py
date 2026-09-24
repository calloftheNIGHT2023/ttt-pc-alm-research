"""Diagnose the 38th task's final-pool bound; no query truths or gate acceptance."""
import argparse
from collections import Counter
from fractions import Fraction as F
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
import probe_credit_confirmation_suite as suite
import probe_confirmation_path_reference as reference
import conditioned_mode_geometry as conditioned
import shared_mode_readout as shared
from posterior_confirmation_pipeline import discovery_box
from probe_exact_polytope_certificate import observed_constraints
from probe_pool_geometry_gate_v2 import checked_component, exact_pool_bound, independent_pool_bound_check
from probe_exact_geometry_gate import READOUT_BUDGET
from run_probe_credit_confirmation_v2 import exclusive_json, verify_commit
from run_multiplier_fixed_point_screen import sha


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def normalized(values):
    values = [F(t) for t in values]
    assert min(values) >= 0 and sum(values) > 0
    return [t / sum(values) for t in values]


def diagnose_pool(keys, cache, records):
    raw = np.array([cache[key]['volume'] for key in keys])
    raw /= raw.sum()
    p = normalized([float(t) for t in raw])
    volumes = [F(records[key]['certificate']['exact_reference_volume']) for key in keys]
    r = normalized(volumes)
    errors = [F(records[key]['certificate']['exact_nominal_readout_expectation_error_bound']) for key in keys]
    result = exact_pool_bound(p, r, errors)
    result['independent_bound_check'] = independent_pool_bound_check(p, r, errors, result)
    # Flatten the hierarchical draw into a single (mode, cone) mixture.
    # TV of the joint weights avoids an extra triangle inequality; the mapped
    # simplex position coupling is unchanged. This is diagnosis, not a new gate.
    joint_p, joint_r, positions, components = [], [], [], []
    for key, prob, refprob, volume, error in zip(keys, p, r, volumes, errors):
        record = records[key]
        conditional_p = normalized([float(t) for t in cache[key]['simplex_probs']])
        conditional_r = [F(t) / volume for t in record['witness']['cone_volumes']]
        assert len(conditional_p) == len(conditional_r) and sum(conditional_r) == 1
        position = sum(F(2**(4-j)) * F(t) for j,t in enumerate(record['certificate']['exact_maximum_coordinate_displacements']))
        assert position + F(record['certificate']['exact_weight_total_variation']) == error
        joint_p.extend(prob * t for t in conditional_p)
        joint_r.extend(refprob * t for t in conditional_r)
        positions.append(prob * position)
        components.append(dict(pattern=key, stored_weight=float(prob), reference_weight=float(refprob),
            component_tv=float(error-position), position_bound=float(position),
            weighted_component_error=float(prob*error)))
    assert sum(joint_p) == sum(joint_r) == 1
    joint_tv = sum(abs(a-b) for a,b in zip(joint_p, joint_r)) / 2
    position = sum(positions)
    joint_bound = joint_tv + position
    assert joint_bound <= F(result['exact_nominal_readout_error_bound'])
    result.update(components=components, joint_cones=len(joint_p), exact_joint_weight_tv=str(joint_tv),
        joint_weight_tv=float(joint_tv), exact_joint_position_bound=str(position),
        exact_joint_nominal_bound=str(joint_bound), joint_nominal_bound=float(joint_bound),
        joint_bound_within_original_budget=joint_bound <= READOUT_BUDGET,
        diagnostic_only_not_gate_acceptance=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, required=True)
    root = parser.parse_args().project.resolve()
    src = Path(__file__).parent
    base = root / 'results/probe_credit_confirmation'
    inp = base / 'predictions'
    out = base / 'task_5500037_geometry_census_v1'
    assert not out.exists() and not (base / 'evaluation_v5').exists()
    failed = read(base / 'pipeline_execution_v5/failure.json')
    assert failed['no_restart'] and 'paths' in failed['message']
    old = read(base / 'path_audit_v5/protocol.json')
    prior = read(base / 'prediction_audit_v2/summary.json')
    assert prior['passed'] and not prior['query_targets_accessed']
    protocol = read(inp / 'protocol.json')
    seed = old['seeds'][37]
    assert seed == 5500037 and protocol['seeds'][37] == seed
    committed = verify_commit(inp, seed, protocol['methods'], sha(inp / 'protocol.json'))
    commit = read(inp / committed['file'])
    rows = {r['method']:r for r in read(inp / commit['rows_file'])}
    configs = suite.catalogue(root)
    assert configs == protocol['configs']
    hashes = {**old['source_sha256'], Path(__file__).name:sha(Path(__file__))}
    for name, digest in hashes.items():
        assert sha(src / name) == digest
    assert os.environ.get('OPENBLAS_NUM_THREADS') == os.environ.get('OMP_NUM_THREADS') == '1'
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    out.mkdir()
    (out / 'geometries').mkdir()
    exclusive_json(out / 'protocol.json', dict(seed=seed, source_sha256=hashes,
        prediction_commit_sha256=committed['sha256'], original_budget=str(READOUT_BUDGET),
        query_targets_accessed=False, audit_gate_passed=False,
        scope='All mode methods of task38; saved-array replay, same certificates; hierarchical and joint-mixture error bounds are diagnostics only'))
    loaded, checkpoints = suite.resources.legacy.oldfit.meta.load(root)
    assert checkpoints == protocol['checkpoint_manifest']
    records, pools, files, cache, counts = {}, [], {}, {}, Counter()
    start = time.perf_counter()
    try:
        with discovery_box(.12):
            for cfg in configs:
                row = rows[cfg['name']]
                meta = row['metadata']
                if meta['execution_failed'] or meta['method_kind'] != 'mode_pool':
                    continue
                assert sha(inp / row['file']) == row['sha256']
                with np.load(inp / row['file']) as saved:
                    arrays = {k:saved[k].copy() for k in saved.files}
                x, v = arrays['x_observed'], arrays['v_observed']
                def traced(c, xx, vv, qq, ss, ll):
                    return suite.resources.fit(c, xx, vv, qq, ss, ll, trace=True)
                repairs = []
                with conditioned.geometry_scope(repairs):
                    fresh, mm = suite.resources.legacy.guarded_fit(cfg, x, v, arrays['q_observed'], seed, loaded, runner=traced)
                assert not mm['execution_failed'] and repairs == meta['geometry_repair_log']
                for name, value in arrays.items():
                    if name not in ['x_observed', 'v_observed', 'q_observed']:
                        assert value.tobytes() == fresh[name].tobytes(), (cfg['name'], name)
                        counts['fresh_bytewise_arrays'] += 1
                seen, checked = reference.reference(cfg, x, v, fresh, meta)
                counts.update(checked)
                for key in meta['positive_modes']:
                    assert key in seen
                    if key in records:
                        continue
                    with conditioned.geometry_scope([]):
                        poly, proof = shared.build(x, v, seen[key])
                    assert poly is not None and proof['proof']['accepted']
                    rr, bb = observed_constraints(x, v, key)
                    certificate, witness = checked_component(poly, rr, bb)
                    filename = f'geometries/{key}.npz'
                    with (out / filename).open('xb') as stream:
                        np.savez_compressed(stream, **poly)
                    files[filename] = sha(out / filename)
                    record = dict(certificate=certificate, witness=witness, geometry_file=filename,
                        x_observed=x.tolist(), v_observed=v.tolist(), query_targets_accessed=False)
                    proofname = f'geometries/{key}.json'
                    exclusive_json(out / proofname, record)
                    files[proofname] = sha(out / proofname)
                    records[key], cache[key] = record, poly
                    counts['certified_geometries'] += 1
                keys = meta['positive_modes']
                result = diagnose_pool(keys, cache, records) if keys else dict(empty_pool=True)
                pools.append(dict(method=cfg['name'], patterns=keys, **result))
                counts['mode_methods'] += 1
                print(json.dumps(dict(method=cfg['name'], modes=len(keys),
                    hierarchical=result.get('nominal_readout_error_bound'), joint=result.get('joint_nominal_bound'),
                    seconds=time.perf_counter()-start)), flush=True)
        for name, digest in hashes.items():
            assert sha(src / name) == digest
        exclusive_json(out / 'pools.json', pools)
        exclusive_json(out / 'files.json', files)
        summary = dict(diagnosis_complete=True, counts=counts, query_targets_accessed=False, audit_gate_passed=False,
            hierarchical_over_budget=sum(row.get('full_pool_bound_within_original_budget') is False for row in pools),
            joint_over_budget=sum(row.get('joint_bound_within_original_budget') is False for row in pools),
            maximum_hierarchical_bound=max(row.get('nominal_readout_error_bound', 0) for row in pools),
            maximum_joint_bound=max(row.get('joint_nominal_bound', 0) for row in pools),
            seconds=time.perf_counter()-start,
            outputs_sha256={name:sha(out/name) for name in ['protocol.json','pools.json','files.json']})
        exclusive_json(out / 'summary.json', summary)
        print(json.dumps(summary), flush=True)
    except BaseException as error:
        exclusive_json(out / 'failure.json', dict(error_type=type(error).__name__, error=str(error), query_targets_accessed=False))
        raise


if __name__ == '__main__':
    main()
