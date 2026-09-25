"""386 old-task n24 prior bounds, frozen arithmetic and shared readout tests."""
from pathlib import Path
import os
import traceback
import numpy as np
import n24_optimizer_controls_v1 as model
import run_independent_hybrid_memory as old
from run_prefix_obstruction_v3 import sha, save


def main(root, out):
    x, v = old.observations(5920000); q = np.linspace(0., 1., 257)
    sources = [Path(__file__), Path(model.__file__), Path(model.old.__file__), Path(model.old.bp.__file__)]
    save(out/'protocol.json', dict(old_seed=5920000, source_sha256={str(p.relative_to(root)):sha(p) for p in sources},
        new_query_targets_accessed=False, objective_scope='Common band feasibility; finite-step surrogates differ'))
    initial_bound = model.old.base.BOUND; bp_bound = model.old.bp.base.BOUND; rows = []
    for family, steps in [('adam', 240), ('gauss_newton', 40), ('pc', 128), ('alm', 128), ('nodual', 128)]:
        point, pm = model.fit(x, v, q, seed=5920000, family=family, steps=steps, restarts=64, readout='point')
        union, um = model.fit(x, v, q, seed=5920000, family=family, steps=steps, restarts=64, readout='posterior_union')
        for k in ['starts', 'best_bank', 'selected_point', 'point_prediction']:
            np.testing.assert_array_equal(point[k], union[k])
        assert model.old.base.BOUND == initial_bound and model.old.bp.base.BOUND == bp_bound
        assert pm['global_bp_used'] == (family in ['adam', 'gauss_newton'])
        assert np.max(abs(point['best_bank'])) <= .12+1e-14
        np.testing.assert_array_equal(point['prediction'], np.clip(model.old.base.forward(q, point['selected_point']), 0., 1.))
        # The adapter must equal direct frozen arithmetic with the same explicit bounds.
        with model.prior_bounds():
            if family in ['adam', 'gauss_newton']:
                reference, _, _ = model.old.run_bp(model.starts(64), x, v, family, steps, trace=False)
            else:
                reference, _, _ = model.old.run_local(model.starts(64), x, v, family, steps, trace=False)
        np.testing.assert_array_equal(point['best_bank'], reference)
        rm = um['readout_metadata']
        assert rm is not None and rm['conditional_on_output_union'] and not rm['full_candidate_set_resolved']
        if rm['readout_available']:
            np.testing.assert_array_equal(union['prediction'], union['readout_prediction'])
            for p in union['readout_points']:
                assert np.max(abs(model.old.base.forward(x, p)-v)) <= .001+1e-7
        directory = out/family; directory.mkdir()
        np.savez_compressed(directory/'point.npz', **point); np.savez_compressed(directory/'union.npz', **union)
        save(directory/'point.json', pm); save(directory/'union.json', um)
        rows.append(dict(family=family, steps=steps, restarts=64, shared_geometry_available=rm['readout_available'],
            point_diagnostic_seconds=pm['total_seconds'], union_diagnostic_seconds=um['total_seconds'],
            files={p.name:sha(p) for p in directory.iterdir() if p.is_file()}))
    # Restoration must also hold through a failed scoped body.
    try:
        with model.prior_bounds(): raise RuntimeError('intentional scope exit')
    except RuntimeError: pass
    assert model.old.base.BOUND == initial_bound and model.old.bp.base.BOUND == bp_bound
    save(out/'rows.json', rows)
    result = dict(passed=True, families=5, point_and_union_calls=10, frozen_bank_replays=5,
        paired_discovery_arrays=20, restored_prior_scope_on_success_and_error=True,
        global_bp_forbidden_in_local_families=True, no_query_targets=True, no_risk_comparison=True,
        outputs_sha256={p:sha(out/p) for p in ['protocol.json', 'rows.json']})
    save(out/'summary.json', result); print(result, flush=True)


if __name__ == '__main__':
    assert os.environ.get('OPENBLAS_NUM_THREADS') == os.environ.get('OMP_NUM_THREADS') == '1'
    root = Path(__file__).resolve().parents[2]; out = root/'results/n24_optimizer_controls/preflight_v1'
    out.mkdir(parents=True, exist_ok=False)
    try: main(root, out)
    except Exception:
        save(out/'failure.json', dict(traceback=traceback.format_exc(), automatic_retry=False)); raise
