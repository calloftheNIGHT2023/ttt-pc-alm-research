"""386 old-input compatibility tests for strong n24 controls, not new task scores."""
from pathlib import Path
import os
import time
import traceback
import numpy as np
import torch
import band_conditioned_meta as meta
import independent_hybrid_memory as regression
import run_independent_hybrid_memory as observations
from run_prefix_obstruction_v3 import read, sha, save


def main(root, out):
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    begin = time.perf_counter(); loaded, manifest = meta.load(root)
    load_seconds = time.perf_counter()-begin
    x, v = observations.observations(5920000); q = np.linspace(0., 1., 257)
    x_saved = x.copy(); v_saved = v.copy(); rows = []; aa = dict(x=x, v=v, q=q)
    for name, model in loaded.items():
        for mode in ['cold24', 'trajectory4_8_16_24']:
            tick = time.perf_counter(); state = None; state_bytes = []; exposures = 0
            for n in ([24] if mode == 'cold24' else [4, 8, 16, 24]):
                predict, state, note = meta.fit(model, x[:n], v[:n], state)
                state_bytes.append(note['persistent_state_bytes']); exposures += n
            prediction = predict(q); seconds = time.perf_counter()-tick
            assert prediction.shape == q.shape and np.isfinite(prediction).all()
            assert np.all((prediction >= 0) & (prediction <= 1))
            np.testing.assert_allclose(predict(q[::-1])[::-1], prediction, rtol=1e-12, atol=1e-12)
            aa[name+'_'+mode+'_prediction'] = prediction
            aa.update({name+'_'+mode+'_'+k:value for k, value in meta.arrays(state).items()})
            rows.append(dict(method=name, mode=mode, diagnostic_seconds=seconds, support_exposures=exposures,
                state_bytes_at_stages=state_bytes, shared_model_bytes=note['shared_model_bytes'],
                repeated_support_processing_charged=True, query_targets_accessed=False))
        if 'ridge' in name or name == 'prior256_fixed':
            np.testing.assert_array_equal(aa[name+'_cold24_prediction'], aa[name+'_trajectory4_8_16_24_prediction'])
    # Validate streaming RLS against its exact same-penalty batch problem at all four stages.
    rls = None; rls_errors = []
    for n in [4, 8, 16, 24]:
        predict, rls, note = regression.regression(x[:n], v[:n], 'residual_linear_rls', rls)
        phi = np.column_stack([np.ones(n), x[:n]])
        target = v[:n]-regression.base.forward(x[:n], np.zeros(4))
        expected = np.linalg.solve(phi.T@phi+1e-4*np.eye(2), phi.T@target)
        error = float(np.max(abs(expected-rls['w']))); rls_errors.append(error)
        np.testing.assert_allclose(rls['w'], expected, rtol=1e-8, atol=1e-8)
    aa['rls_w'] = rls['w']; aa['rls_p'] = rls['p']; aa['rls_prediction'] = predict(q)
    for name in ['linear_ls', 'residual_linear_ls', 'prior4096_ridge', 'rbf_loocv']:
        tick = time.perf_counter(); predict, _, note = regression.regression(x, v, name)
        prediction = predict(q); seconds = time.perf_counter()-tick
        assert prediction.shape == q.shape and np.isfinite(prediction).all()
        np.testing.assert_allclose(predict(q[::-1])[::-1], prediction, rtol=1e-10, atol=1e-10)
        aa[name+'_prediction'] = prediction
        rows.append(dict(method=name, diagnostic_seconds=seconds, metadata=note, query_targets_accessed=False))
    np.testing.assert_array_equal(x, x_saved); np.testing.assert_array_equal(v, v_saved)
    # Same task family as the branch model, tested on definition-only synthetic inputs.
    b = np.array([[.03, -.08, .11, -.02]])
    teacher_gap = float(np.max(abs(meta.models.teacher(torch.as_tensor(q[None]), torch.as_tensor(b)).numpy()[0]-regression.base.forward(q, b[0]))))
    assert teacher_gap < 1e-14
    sources = ['test_n24_task_baselines_v1.py', 'band_conditioned_meta.py', 'matched_shifted_meta_models.py',
        'independent_hybrid_memory.py', 'run_independent_hybrid_memory.py']
    save(out/'protocol.json', dict(old_seed=5920000, new_query_risk_experiment=False,
        source_sha256={s:sha(Path(__file__).with_name(s)) for s in sources},
        inherited_meta_protocol_sha256=sha(root/'results/scalar_matched_batched/meta_prior/protocol.json'),
        inherited_meta_training_summary_sha256=sha(root/'results/scalar_matched_batched/meta_prior/training_summary.json')))
    np.savez_compressed(out/'arrays.npz', **aa); save(out/'models.json', manifest); save(out/'rows.json', rows)
    result = dict(passed=True, methods=len(loaded)+4+1, meta_modes=2, meta_predictions=10,
        diagnostic_model_load_seconds=load_seconds, rls_batch_max_errors=rls_errors, teacher_max_error=teacher_gap,
        new_query_targets_accessed=False, no_new_training=True, no_risk_ranking=True,
        cold_and_training_trajectory_both_preserved=True,
        outputs_sha256={s:sha(out/s) for s in ['protocol.json', 'arrays.npz', 'models.json', 'rows.json']})
    save(out/'summary.json', result); print(result, flush=True)


if __name__ == '__main__':
    assert os.environ.get('OPENBLAS_NUM_THREADS') == os.environ.get('OMP_NUM_THREADS') == '1'
    root = Path(__file__).resolve().parents[2]; out = root/'results/n24_task_baselines/preflight_v1'
    out.mkdir(parents=True, exist_ok=False)
    try: main(root, out)
    except Exception:
        save(out/'failure.json', dict(traceback=traceback.format_exc(), automatic_retry=False)); raise
