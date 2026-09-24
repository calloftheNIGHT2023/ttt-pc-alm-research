"""292 OLD64 functional gate, not a quality or matched-resource experiment."""
import argparse
from pathlib import Path
import time
import numpy as np
import counterfactual_credit_branching_v1 as new
import conditioned_mode_geometry as conditioned
from posterior_confirmation_pipeline import discovery_box
from diagnose_gradient_flat_split_states_v1 import read, save, sha


def run(root, out):
    raw = root/'results/certificate_activity_attribution/development'
    old = {r['seed']:r for r in read(raw/'rows.json') if r['method']=='credit_control_probe33'}
    diagnostic = root/'results/gradient_flat_split_states/development_v1'
    inputs = read(diagnostic/'input_hashes.json')
    shadow = root/'results/state_matched_dual_intervention/development_v2'
    proposals = root/'results/counterfactual_branch_proposals/development_v1'
    ps = read(proposals/'summary.json')
    assert ps['passed'] and ps['tasks'] == 64
    for name, digest in ps['outputs_sha256'].items():
        assert sha(proposals/name) == digest
    selections = {r['seed']:r for r in read(proposals/'tasks.json')}
    frozen = read(root/'results/round_287_audit_v6.json')
    for name, digest in frozen['source_sha256'].items():
        assert sha(root/'work/experiments'/name) == digest
    sources = {n: sha(root/'work/experiments'/n) for n in
               ['test_counterfactual_credit_branching_v2.py','counterfactual_credit_branching_v1.py']}
    save(out/'protocol.json', dict(seeds=list(range(5910000,5910064)), source_sha256=sources,
        parent_summary_sha256=sha(proposals/'summary.json'),
        design_sha256=sha(root/'outputs/ttt-pc-alm-research/292_full_support_gate.md'),
        original_bound=.12, query_targets_accessed=False, query_coordinates='public linspace(0,1,257)',
        purpose='Functional only; 288 audit runs concurrently, no fair timing claim',
        frozen_dependency_seal_sha256=sha(root/'results/round_287_audit_v6.json')))
    cfg = next(dict(c) for c in new.original.CONFIGS if c['name']=='credit_control_probe33')
    records = []
    outputs = {'protocol.json':sha(out/'protocol.json')}
    for seed in range(5910000, 5910064):
        path = raw/old[seed]['file']
        assert sha(path) == old[seed]['sha256'] == inputs[str(path.relative_to(root)).replace('\\','/')]
        with np.load(path, allow_pickle=False) as z:
            x, v = z['x_observed'], z['v_observed']
            stored = {n:z[n] for n in z.files if n.startswith(('prefix_','anchor_'))}
            old_best = z['selected_b']
        q = np.linspace(0, 1, 257)
        with discovery_box(.12):
            base_repairs = []
            with conditioned.geometry_scope(base_repairs):
                base, base_meta = new.original.fit(cfg, x, v, q, seed, trace=True)
            zero_repairs = []
            with conditioned.geometry_scope(zero_repairs):
                zero, zero_meta = new.fit(x, v, q, seed, rule='none', trace=True)
            for name, value in base.items():
                assert value.tobytes() == zero[name].tobytes(), (seed, 'none', name)
            assert zero_meta['counterfactual_state_steps'] == 0
            repairs = []
            begin = time.perf_counter()
            with conditioned.geometry_scope(repairs):
                candidate, meta = new.fit(x, v, q, seed, trace=True)
            seconds = time.perf_counter()-begin
        for name, value in stored.items():
            assert candidate[name].tobytes() == value.tobytes() == base[name].tobytes(), (seed, name)
        assert candidate['selected_b'].tobytes() == old_best.tobytes()
        expected = selections[seed]['rules']['changed_forward_mode']
        locations = candidate['counterfactual_locations'].tolist()
        assert locations == expected['locations']
        assert meta['counterfactual_state_steps'] == expected['proposals'] == len(locations)
        with np.load(shadow/f'{seed}.npz', allow_pickle=False) as z:
            bank = [z[('prefix' if phase==0 else 'anchor')+'_shadow_b'][t-1, origin]
                    for phase, t, origin in locations]
        assert candidate['counterfactual_b'].tobytes() == np.array(bank).tobytes()
        assert set(meta['positive_modes']) == set(base_meta['positive_modes'])|set(expected['new_positive_modes'])
        assert set(base_meta['visited_modes']) <= set(meta['visited_modes'])
        file = f'{seed}.npz'
        with (out/file).open('xb') as stream:
            np.savez_compressed(stream, **candidate)
        outputs[file] = sha(out/file)
        records.append(dict(seed=seed, wrapper_identity_arrays=len(base), preserved_original_trace_arrays=len(stored),
            exact_shadow_rows=len(locations), new_positive_modes=expected['new_positive_modes'],
            functional_only_complete_seconds=seconds, metadata=meta, geometry_repairs=repairs,
            source_file_sha256=old[seed]['sha256'], output_file=file, output_sha256=outputs[file]))
        print(dict(seed=seed, passed=True, shadow_rows=len(locations), new_modes=len(expected['new_positive_modes'])), flush=True)
    save(out/'rows.json', records)
    outputs['rows.json'] = sha(out/'rows.json')
    for name, digest in sources.items():
        assert sha(root/'work/experiments'/name) == digest
    result = dict(passed=True, tasks=64, seeds=list(range(5910000,5910064)), functional_only=True,
        exact_shadow_rows=sum(r['exact_shadow_rows'] for r in records),
        original_trajectory_arrays=sum(r['preserved_original_trace_arrays'] for r in records),
        wrapper_identity_arrays=sum(r['wrapper_identity_arrays'] for r in records),
        query_targets_accessed=False, query_quality_evaluated=False, resources_matched=False,
        core_research_goal_complete=False, outputs_sha256=outputs,
        next='Calibrate complete cold calls and matched strong controls before any fresh query experiment')
    save(out/'summary.json', result)
    print(result, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    root = parser.parse_args().project.resolve()
    out = root/'results/counterfactual_credit_branching/full_support_v2'
    out.mkdir(parents=True, exist_ok=False)
    try:
        run(root, out)
    except BaseException as exc:
        save(out/'failure.json', dict(error_type=type(exc).__name__, message=str(exc), no_automatic_retry=True))
        raise


if __name__ == '__main__':
    main()
