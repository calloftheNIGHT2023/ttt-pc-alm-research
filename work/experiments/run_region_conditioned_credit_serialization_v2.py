"""I/O-only repair after v1 failed serializing PDHG NumPy proof arrays.

The frozen solver/test/runner remain byte-identical. Preserve failed attempt1;
record this adapter in the new protocol before any repeated numerical work.
"""
from pathlib import Path
import traceback
import numpy as np
import budget_reinvestment_suite_v1 as io
import run_region_conditioned_credit_v1 as original


def lists(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k):lists(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)):
        return [lists(v) for v in value]
    return value


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[2]
    failed = root/'results/region_conditioned_credit/development_attempt1'
    assert (failed/'failure.json').exists()
    note = dict(adapter_file='work/experiments/'+Path(__file__).name,
                adapter_sha256=io.sha(Path(__file__)),
                failure_file=str((failed/'failure.json').relative_to(root)),
                failure_sha256=io.sha(failed/'failure.json'),
                change='Serialize NumPy proof arrays/scalars as JSON lists/numbers; no numeric algorithm change')
    save_original = io.save
    def save(path, value):
        if path.name == 'protocol.json':
            value = dict(value, serialization_repair=note)
        return save_original(path, lists(value))
    io.save = save
    out = root/'results/region_conditioned_credit/development_v1'
    out.mkdir(parents=True, exist_ok=False)
    try:
        original.run(root, out)
    except Exception:
        io.save(out/'failure.json', dict(traceback=traceback.format_exc(), automatic_retry=False)); raise
    finally:
        io.save = save_original
