"""Diagnose the first frozen confirmation geometry-budget failure, not a gate.

Replays the original first method and first positive mode from observed supports.
Preserves every frozen source, prediction, failed path attempt and tolerance.
No teacher query values, quality scores, model selection or online change.
"""
import argparse
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
from probe_exact_polytope_certificate import observed_constraints, certify
from probe_exact_geometry_gate import validate, READOUT_BUDGET
from run_probe_credit_confirmation_v2 import exclusive_json
from run_multiplier_fixed_point_screen import sha


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', type=Path, required=True)
    root = ap.parse_args().project.resolve()
    src = Path(__file__).parent
    base = root/'results/probe_credit_confirmation'
    inp = base/'predictions'
    failed = base/'path_audit_v3'
    out = base/'geometry_budget_diagnosis_v1'
    assert not out.exists()
    assert not (inp/'RUNNING.lock').exists()
    assert not (failed/'RUNNING.lock').exists()
    assert not (base/'evaluation_v3').exists()
    assert read(base/'pipeline_execution_v3/paths_exit.json')['returncode'] == 1
    assert read(base/'prediction_audit_v2/summary.json')['passed']
    assert not read(base/'prediction_audit_v2/summary.json')['query_targets_accessed']
    protocol = read(failed/'protocol.json')
    hashes = dict(protocol['source_sha256'])
    hashes[Path(__file__).name] = sha(Path(__file__))
    for name, digest in hashes.items():
        assert sha(src/name) == digest, name
    assert os.environ.get('OPENBLAS_NUM_THREADS') == os.environ.get('OMP_NUM_THREADS') == '1'
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    seed = protocol['seeds'][0]
    assert seed == 5500000
    committed = read(inp/'tasks'/str(seed)/'commit.json')
    rows = {r['method']: r for r in read(inp/committed['rows_file'])}
    cfg = suite.catalogue(root)[0]
    row = rows[cfg['name']]
    assert sha(inp/row['file']) == row['sha256']
    with np.load(inp/row['file']) as z:
        saved = {k: z[k].copy() for k in z.files}
    x, v, q = (saved[k] for k in ['x_observed', 'v_observed', 'q_observed'])
    assert x.shape == v.shape == (4,) and q.shape == (257,)
    out.mkdir()
    exclusive_json(out/'protocol.json', dict(
        seed=seed, method=cfg['name'], source_sha256=hashes,
        failed_path_protocol_sha256=sha(failed/'protocol.json'),
        failed_driver_record_sha256=sha(base/'pipeline_execution_v3/failure.json'),
        original_prediction_sha256=row['sha256'], query_targets_accessed=False,
        audit_gate_passed=False, budget=str(READOUT_BUDGET),
        scope='Reproduce the first positive geometry and decompose its existing exact uniform-readout upper bound; diagnosis only'))
    loaded, checkpoint_manifest = suite.resources.legacy.oldfit.meta.load(root)
    assert checkpoint_manifest == read(inp/'protocol.json')['checkpoint_manifest']
    begin = time.perf_counter()
    with discovery_box(.12):
        def traced(c, xx, vv, qq, ss, ll):
            return suite.resources.fit(c, xx, vv, qq, ss, ll, trace=True)
        repairs = []
        with conditioned.geometry_scope(repairs):
            fresh, meta = suite.resources.legacy.guarded_fit(cfg, x, v, q, seed, loaded, runner=traced)
        assert not meta['execution_failed'] and repairs == row['metadata']['geometry_repair_log']
        replay_arrays = []
        for name, value in saved.items():
            if name not in ['x_observed', 'v_observed', 'q_observed']:
                assert value.tobytes() == fresh[name].tobytes(), name
                replay_arrays.append(name)
        seen, counters = reference.reference(cfg, x, v, fresh, meta)
        chosen = None
        for key, b in sorted(seen.items()):
            feasibility = suite.resources.legacy.cold.base.branch_feasibility(x, v, b)
            if not feasibility['current_branch_feasible']:
                continue
            build_repairs = []
            with conditioned.geometry_scope(build_repairs):
                poly, proof = shared.build(x, v, b)
            if poly is not None:
                chosen = (key, b, poly, proof, build_repairs)
                break
    assert chosen is not None
    key, representative, poly, proof, build_repairs = chosen
    assert key == row['metadata']['positive_modes'][0]
    assert proof['proof']['accepted']
    constraints, rhs = observed_constraints(x, v, key)
    validate(poly, constraints, rhs)
    certificate, witness = certify(poly, constraints, rhs)
    assert np.isclose(certificate['reference_volume'], float(poly['volume']), rtol=1e-8, atol=1e-22)
    bound = F(certificate['exact_nominal_readout_expectation_error_bound'])
    assert bound > READOUT_BUDGET, 'The exact failed condition must be reproduced'
    weight_tv = F(certificate['exact_weight_total_variation'])
    displacement = [F(t) for t in certificate['exact_maximum_coordinate_displacements']]
    position = sum(F(2**(4-j))*d for j, d in enumerate(displacement))
    assert position + weight_tv == bound
    vertices = [[F(t) for t in row] for row in witness['vertices']]
    widths = [max(p[j] for p in vertices)-min(p[j] for p in vertices) for j in range(4)]
    generic_diameter = min(F(1), sum(F(2**(4-j))*w for j, w in enumerate(widths)))
    with (out/'geometry.npz').open('xb') as f:
        np.savez_compressed(f, **poly, x_observed=x, v_observed=v, representative_b=representative)
    exclusive_json(out/'certificate.json', certificate)
    exclusive_json(out/'witness.json', witness)
    exclusive_json(out/'replay.json', dict(
        bytewise_replayed_arrays=replay_arrays, reference_counts=counters,
        original_repair_log=repairs, geometry_repair_log=build_repairs,
        proof=proof['proof'], query_targets_accessed=False))
    diagnosis = dict(
        diagnosis_complete=True, audit_gate_passed=False, query_targets_accessed=False,
        seed=seed, method=cfg['name'], pattern=key,
        failed_condition_reproduced=True, original_predictions_unchanged=True,
        exact_weight_tv=str(weight_tv), weight_tv=float(weight_tv),
        exact_position_bound=str(position), position_bound=float(position),
        exact_total_bound=str(bound), total_bound=float(bound),
        exact_budget=str(READOUT_BUDGET), budget=float(READOUT_BUDGET),
        bound_over_budget=float(bound/READOUT_BUDGET),
        exact_coordinate_widths=[str(w) for w in widths],
        coordinate_widths=[float(w) for w in widths],
        generic_lipschitz_readout_oscillation_bound=float(generic_diameter),
        exact_generic_lipschitz_readout_oscillation_bound=str(generic_diameter),
        reference_volume=certificate['reference_volume'],
        saved_volume_relative_error=certificate['saved_volume_relative_error'],
        exact_geometry_vertices=certificate['exact_vertices'],
        saved_facets=len(poly['facets']), seconds=time.perf_counter()-begin,
        outputs_sha256={name: sha(out/name) for name in [
            'protocol.json', 'geometry.npz', 'certificate.json', 'witness.json', 'replay.json']})
    exclusive_json(out/'summary.json', diagnosis)
    print(json.dumps(diagnosis), flush=True)


if __name__ == '__main__':
    main()
