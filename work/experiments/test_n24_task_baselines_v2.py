"""Preserve v1 negative-stride test failure; only repair NumPy/Torch boundary."""
from pathlib import Path
import os
import traceback
import test_n24_task_baselines_v1 as test
import n24_meta_input_adapter_v1 as adapter
from run_prefix_obstruction_v3 import read, sha, save


if __name__ == '__main__':
    assert os.environ.get('OPENBLAS_NUM_THREADS') == os.environ.get('OMP_NUM_THREADS') == '1'
    root = Path(__file__).resolve().parents[2]; out = root/'results/n24_task_baselines/preflight_v2'
    out.mkdir(parents=True, exist_ok=False)
    original = test.meta
    try:
        failure = root/'results/n24_task_baselines/preflight_v1/failure.json'
        assert 'negative' in read(failure)['traceback']
        test.meta = adapter
        test.main(root, out)
        note = dict(passed=True, source_sha256={str(p.relative_to(root)):sha(p) for p in [Path(__file__), Path(adapter.__file__)]},
            original_failure_sha256=sha(failure), test_summary_sha256=sha(out/'summary.json'),
            repair='Contiguous NumPy boundary only; original checkpoints, source models, and all checks retained',
            no_model_or_checkpoint_changes=True)
        save(out/'adapter_verification.json', note); print(note, flush=True)
    except Exception:
        save(out/'failure.json', dict(traceback=traceback.format_exc(), automatic_retry=False)); raise
    finally:
        test.meta = original
