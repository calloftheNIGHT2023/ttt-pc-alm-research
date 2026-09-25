"""387 run future proof-audit paths on already sealed old-task session outputs."""
from collections import Counter
from pathlib import Path
import os
import traceback
import numpy as np
import deadline_risk_io_v1 as io
import deadline_risk_registry_v1 as registry
import audit_new_task_deadline_risk_v1 as auditor


def main(root, out):
    pre = root/io.BASE/'preflight_v1'; summary = io.read(pre/'summary.json'); assert summary['passed']
    for p, h in summary['outputs_sha256'].items(): assert io.sha(pre/p) == h
    import audit_contiguous_regional_repeat_v1 as whole
    counts = Counter(); records = []
    for cfg in registry.configs():
        directory = pre/cfg['name']/'a0'; a = io.load_arrays(directory/'state.npz'); m = io.read(directory/'state.json')
        outputs = io.load_arrays(directory/'outputs.npz'); meta = io.read(directory/'metadata.json'); events = io.read(directory/'events.json')
        eligible = [(j, e) for j, e in enumerate(events) if e['received_seconds'] <= meta['budget_seconds'] and e['kind'] in ['fallback', 'final']]
        j, last = eligible[-1]; assert last['kind'] == meta['selected'] == 'final'
        np.testing.assert_array_equal(outputs['prediction'], outputs[f'event_{j}_prediction'])
        assert all(e['generation'] == meta['generation'] for e in events)
        assert all(0 <= e['worker_seconds'] <= e['received_seconds']+1e-3 for e in events)
        cc = Counter()
        if cfg['kind'] == 'regional':
            cc.update(whole.audit_canonical({k:value for k, value in a.items() if k.startswith(('search_', 'readout_'))}, m))
        elif cfg['kind'] == 'optimizer' and cfg['readout'] == 'posterior_union':
            cc.update(auditor.readout_check({k[8:]:value for k, value in a.items() if k.startswith('readout_')}, m['readout_metadata']))
        counts.update(cc); counts['receipt_checks'] += 1; records.append(dict(method=cfg['name'], checks=dict(cc)))
    # Analytic deadline chooser is independent of the implementation's helper.
    for budget, times, expected in [(0., [.1,.2], -1), (.1,[.1,.2],0), (.15,[.1,.2],0), (.2,[.1,.2],1)]:
        accepted = [i for i, t in enumerate(times) if t <= budget]; assert (accepted[-1] if accepted else -1) == expected
        counts['analytic_deadline_boundaries'] += 1
    io.save(out/'checks.json', records)
    result = dict(passed=True, checks=dict(counts), query_targets_accessed=False,
        preflight_summary_sha256=io.sha(pre/'summary.json'),
        source_sha256={str(p.relative_to(root)):io.sha(p) for p in [Path(__file__), Path(auditor.__file__)]},
        outputs_sha256={'checks.json':io.sha(out/'checks.json')})
    io.save(out/'summary.json', result); print(result, flush=True)


if __name__ == '__main__':
    assert os.environ.get('OPENBLAS_NUM_THREADS') == os.environ.get('OMP_NUM_THREADS') == '1'
    root = Path(__file__).resolve().parents[2]; out = root/io.BASE/'audit_preflight_v1'; out.mkdir(parents=True, exist_ok=False)
    try: main(root, out)
    except Exception:
        io.save(out/'failure.json', dict(traceback=traceback.format_exc(), automatic_retry=False)); raise
