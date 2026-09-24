"""Sixth frozen task's geometry-budget census, without accepting any gate.

Fresh trace-enabled replay supplies reference-only arrays absent from saved
prediction artifacts. Checks every saved array bytewise, then records geometry
bounds without accepting the fixed-first64 gate or reading query quality.
"""
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
from probe_exact_polytope_certificate import observed_constraints, certify, all_vertices, reference_volume
from probe_exact_geometry_gate import validate, READOUT_BUDGET
from run_probe_credit_confirmation_v2 import exclusive_json, verify_commit
from run_multiplier_fixed_point_screen import sha


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, required=True)
    root = parser.parse_args().project.resolve()
    src = Path(__file__).parent
    base = root / 'results/probe_credit_confirmation'
    inp = base / 'predictions'
    out = base / 'task_5500005_geometry_census_v1'
    assert not out.exists() and not (base / 'evaluation_v3').exists()
    old = read(base / 'path_audit_v4/protocol.json')
    prior = read(base / 'prediction_audit_v2/summary.json')
    assert prior['passed'] and not prior['query_targets_accessed']
    protocol = read(inp / 'protocol.json')
    seed = old['seeds'][5]
    assert seed == 5500005 and protocol['seeds'][5] == seed
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
        scope='All saved method pools of sixth task only; fresh bytewise array replay and original-block references; diagnosis only'))
    loaded, checkpoint_manifest = suite.resources.legacy.oldfit.meta.load(root)
    assert checkpoint_manifest == protocol['checkpoint_manifest']
    records, pools, files, counts = {}, [], {}, Counter()
    start = time.perf_counter()
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
            replay_repairs = []
            with conditioned.geometry_scope(replay_repairs):
                fresh, fresh_meta = suite.resources.legacy.guarded_fit(
                    cfg, x, v, arrays['q_observed'], seed, loaded, runner=traced)
            assert not fresh_meta['execution_failed']
            assert replay_repairs == meta['geometry_repair_log']
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
                repairs = []
                with conditioned.geometry_scope(repairs):
                    poly, proof = shared.build(x, v, seen[key])
                assert poly is not None and proof['proof']['accepted']
                rr, bb = observed_constraints(x, v, key)
                filename = f'geometries/{key}.npz'
                with (out / filename).open('xb') as stream:
                    np.savez_compressed(stream, **poly)
                files[filename] = sha(out / filename)
                try:
                    validate(poly, rr, bb)
                    certificate, witness = certify(poly, rr, bb)
                    assert np.isclose(certificate['reference_volume'], float(poly['volume']), rtol=1e-8, atol=1e-22)
                    upper = F(certificate['exact_nominal_readout_expectation_error_bound'])
                    records[key] = dict(certificate_valid=True, certificate=certificate, witness=witness,
                        old_bound_within_budget=upper <= READOUT_BUDGET, saved_volume=float(poly['volume']),
                        geometry_file=filename, repairs=repairs, query_targets_accessed=False)
                    counts['valid_geometry_certificates'] += 1
                    counts['old_bound_over_budget'] += int(upper > READOUT_BUDGET)
                except (AssertionError, ValueError, KeyError) as error:
                    records[key] = dict(certificate_valid=False, exception_type=type(error).__name__,
                        error=str(error), saved_volume=float(poly['volume']), geometry_file=filename,
                        repairs=repairs, query_targets_accessed=False)
                    exact, _ = all_vertices(rr, bb)
                    raw_vertices = np.unique(poly['facets'].reshape(-1,4), axis=0)
                    physical = [tuple(F(float(c))+F(float(scale))*F(float(t))
                        for c,scale,t in zip(poly['center'],poly['scale'],point)) for point in raw_vertices]
                    distances = [[max(abs(a-b) for a,b in zip(point,vertex)) for vertex in exact] for point in physical]
                    matching = [min(range(len(exact)),key=lambda i:row[i]) for row in distances]
                    refvol, _ = reference_volume(exact,rr,bb)
                    records[key]['matching_diagnosis'] = dict(
                        raw_vertices=len(raw_vertices), exact_vertices=len(exact), unique_nearest_vertices=len(set(matching)),
                        nearest_matching=matching, missing_exact_vertices=sorted(set(range(len(exact)))-set(matching)),
                        exact_vertices_coordinates=[[str(t) for t in row] for row in exact],
                        exact_physical_coordinates=[[str(t) for t in row] for row in physical],
                        maximum_nearest_distance=float(max(min(row) for row in distances)),
                        exact_reference_volume=str(refvol), exact_rows=[list(row) for row in rr],
                        exact_rhs=[str(t) for t in bb])
                    counts['certificate_failures'] += 1
                proofname = f'geometries/{key}.json'
                exclusive_json(out / proofname, records[key])
                files[proofname] = sha(out / proofname)
                counts['geometries'] += 1
                print(json.dumps(dict(pattern=key, geometries=counts['geometries'],
                    certificate_valid=records[key]['certificate_valid'],
                    bound=records[key].get('certificate', {}).get('nominal_readout_expectation_error_bound'),
                    over_budget=counts['old_bound_over_budget'], seconds=time.perf_counter()-start)), flush=True)
            keys = meta['positive_modes']
            if not keys:
                pools.append(dict(method=cfg['name'], empty_pool=True))
            elif all(records[key]['certificate_valid'] for key in keys):
                raw = np.array([records[key]['saved_volume'] for key in keys])
                raw /= raw.sum()
                stored = [F(float(t)) for t in raw]
                total = sum(stored)
                stored = [t / total for t in stored]
                actual = [F(records[key]['certificate']['exact_reference_volume']) for key in keys]
                total = sum(actual)
                actual = [t / total for t in actual]
                tv = sum(abs(a-b) for a,b in zip(stored, actual)) / 2
                bound = tv + sum(w * F(records[key]['certificate']['exact_nominal_readout_expectation_error_bound'])
                                 for key,w in zip(keys, stored))
                pools.append(dict(method=cfg['name'], patterns=keys, exact_mode_weight_tv=str(tv),
                    exact_old_nominal_bound=str(bound), mode_weight_tv=float(tv), old_nominal_bound=float(bound),
                    old_bound_within_budget=bound <= READOUT_BUDGET))
            else:
                pools.append(dict(method=cfg['name'], patterns=keys, unresolved_geometry_certificate=True))
            counts['mode_pool_methods'] += 1
    for name, digest in hashes.items():
        assert sha(src / name) == digest
    exclusive_json(out / 'pools.json', pools)
    exclusive_json(out / 'files.json', files)
    valid = [row for row in records.values() if row['certificate_valid']]
    summary = dict(diagnosis_complete=True, counts=counts, query_targets_accessed=False, audit_gate_passed=False,
        maximum_old_geometry_bound=max((float(F(row['certificate']['exact_nominal_readout_expectation_error_bound']))
                                        for row in valid), default=None),
        pools_over_budget=sum(row.get('old_bound_within_budget') is False for row in pools),
        maximum_mode_weight_tv=max((row.get('mode_weight_tv', 0) for row in pools), default=None),
        seconds=time.perf_counter()-start,
        scope='Conservative-bound census only; an excessive upper bound is not an actual error violation',
        outputs_sha256={name:sha(out/name) for name in ['protocol.json','pools.json','files.json']})
    exclusive_json(out / 'summary.json', summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
