"""382 all 544 actual sealed candidate sets, common uncached geometry/readout."""
from collections import Counter
from pathlib import Path
import os
import time
import traceback
import numpy as np
import candidate_set_readout_v1 as model
import run_retired_region_join_v1 as previous
from run_prefix_obstruction_v3 import read, sha, save, complete

BASE = 'results/candidate_set_readout'
DESIGN = 'outputs/ttt-pc-alm-research/382_common_readout_protocol_v1.md'


def hashes(root):
    h = dict(complete(root/BASE/'prototype_preflight_v1')['source_sha256'])
    for p, value in h.items(): assert sha(root/p) == value, p
    for p in [DESIGN, 'work/experiments/run_candidate_set_readout_v1.py', 'work/experiments/audit_candidate_set_readout_v1.py']:
        h[p] = sha(root/p)
    return h


def run(root, out):
    begin = time.perf_counter(); h = hashes(root); source = root/previous.BASE/'development_v1'
    complete(source); audit = complete(root/previous.BASE/'audit_v1')
    assert audit['development_summary_sha256'] == sha(source/'summary.json')
    assert read(source/'protocol.json')['source_sha256'] == previous.hashes(root)
    inputs = read(source/'rows.json'); assert len(inputs) == 544
    save(out/'protocol.json', dict(source_sha256=h, search_rows_sha256=sha(source/'rows.json'), search_audit_sha256=sha(root/previous.BASE/'audit_v1/summary.json'),
        primary=previous.model.PRIMARY, methods=[c[0] for c in previous.model.CONFIGS], settings=previous.model.SETTINGS,
        query_targets_accessed=False, all_inputs_old_development=True, source_rows=inputs,
        cross_method_geometry_cache=False, staged_sum_not_contiguous_timing=True, new_blind_confirmation=False))
    q = np.linspace(0., 1., 257); rows = []
    for seed in sorted({r['seed'] for r in inputs}):
        jobs = [r for r in inputs if r['seed'] == seed]; assert len(jobs) == 34
        for j in np.random.default_rng(np.random.SeedSequence([382929, seed])).permutation(34):
            row = jobs[int(j)]; src = root/row['directory']
            for f, value in row['files'].items(): assert sha(src/f) == value
            observed = previous.load(src/'arrays.npz'); x, v, regions = observed['original_x'], observed['original_v'], observed['regions']
            before = x.tobytes(), v.tobytes(), regions.tobytes(), q.tobytes()
            a, m = model.fit(x, v, q, regions, seed=seed, search_completed=bool(row['completed']))
            assert before == (x.tobytes(), v.tobytes(), regions.tobytes(), q.tobytes())
            directory = out/f"{seed}_{row['schedule']}_{row['ordering']}_{row['method']}"; directory.mkdir()
            tick = time.perf_counter(); np.savez_compressed(directory/'arrays.npz', **a); save(directory/'metadata.json', m)
            disk_seconds = time.perf_counter()-tick
            result = {k: row[k] for k in ['seed', 'method', 'schedule', 'ordering', 'completed', 'stop_reason', 'remaining']}
            result.update(search_seconds=row['seconds'], readout_seconds=m['charged_readout_seconds'],
                staged_total_seconds=row['seconds']+m['charged_readout_seconds'], geometry_seconds=m['geometry_seconds'],
                sampling_seconds=m['sampling_seconds'], reading_seconds=m['reading_seconds'], disk_seconds=disk_seconds,
                readout_available=m['readout_available'], full_candidate_set_resolved=m['full_candidate_set_resolved'],
                positive_modes=len(m['positive_modes']), unresolved_modes=len(m['unresolved_modes']),
                classification_counts=m['classification_counts'], geometry_numeric_bytes=m['geometry_numeric_bytes_subtotal'],
                returned_array_bytes=m['returned_array_bytes'], search_named_bytes=row['named_bytes'],
                source_directory=row['directory'], directory=str(directory.relative_to(root)),
                files={f: sha(directory/f) for f in ['arrays.npz', 'metadata.json']})
            rows.append(result)
            if len(rows) % 4 == 0:
                print(dict(stage='common_readout', calls=len(rows), resolved=sum(r['full_candidate_set_resolved'] for r in rows),
                           seconds=time.perf_counter()-begin), flush=True)
        save(out/f'sealed_seed_{seed}.json', rows[-34:])
    assert len(rows) == 544 and hashes(root) == h; save(out/'rows.json', rows)
    result = dict(passed=True, calls=len(rows), resolved=sum(r['full_candidate_set_resolved'] for r in rows),
        unavailable=sum(not r['readout_available'] for r in rows), seconds=time.perf_counter()-begin,
        query_targets_accessed=False, all_readouts_sealed_before_audit=True, core_research_goal_complete=False,
        outputs_sha256={f: sha(out/f) for f in ['protocol.json', 'rows.json']})
    save(out/'summary.json', result); print(dict(stage='readout_complete', **result), flush=True)


if __name__ == '__main__':
    assert os.environ.get('OPENBLAS_NUM_THREADS') == os.environ.get('OMP_NUM_THREADS') == '1'
    root = Path(__file__).resolve().parents[2]; out = root/BASE/'development_v1'; out.mkdir(parents=True, exist_ok=False)
    try: run(root, out)
    except Exception:
        save(out/'failure.json', dict(traceback=traceback.format_exc(), automatic_retry=False)); raise
