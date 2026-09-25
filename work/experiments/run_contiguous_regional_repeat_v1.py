"""384 frozen contiguous support-to-prediction repeats, no archived fit reuse."""
import argparse
from collections import Counter
from pathlib import Path
import hashlib
import os
import time
import traceback
import numpy as np
import continuous_regional_prediction_v1 as model
import run_candidate_set_readout_v1 as previous
import run_regional_prefix_join_v1 as legacy
from run_prefix_obstruction_v3 import read, sha, save, complete

BASE = 'results/contiguous_regional_repeat'
DESIGN = 'outputs/ttt-pc-alm-research/384_contiguous_repeat_protocol_v1.md'
NAMES = ['regional_active256', 'regional_active512', 'regional_active1024', 'regional_passive1024']+[
    f'pdhg_{initial}{steps}' for initial in ['cold', 'box'] for steps in [256, 512, 1024]]
SETTINGS = [('bfs', 'observed'), ('dfs', 'farthest_x')]
PRIMARY = 'regional_active1024'


def hashes(root):
    parent = complete(root/previous.BASE/'audit_v1')
    assert parent['development_summary_sha256'] == sha(root/previous.BASE/'development_v1/summary.json')
    h = dict(read(root/previous.BASE/'development_v1/protocol.json')['source_sha256'])
    assert previous.hashes(root) == h
    for p in [DESIGN]+['work/experiments/'+n for n in ['continuous_regional_prediction_v1.py',
        'run_contiguous_regional_repeat_v1.py', 'audit_contiguous_regional_repeat_v1.py']]: h[p] = sha(root/p)
    return h


def digests(arrays):
    return {k: dict(shape=list(a.shape), dtype=a.dtype.str, sha256=hashlib.sha256(a.tobytes()).hexdigest()) for k, a in sorted(arrays.items())}


def untimed(value):
    if isinstance(value, dict): return {k: untimed(v) for k, v in value.items() if not k.endswith('seconds')}
    if isinstance(value, list): return [untimed(v) for v in value]
    return value


def split(a):
    assert all(k.startswith(('search_', 'readout_')) for k in a)
    return ({k[7:]: value for k, value in a.items() if k.startswith('search_')},
            {k[8:]: value for k, value in a.items() if k.startswith('readout_')})


def preflight(root, out):
    from audit_contiguous_regional_repeat_v1 import audit_canonical
    begin = time.perf_counter(); h = hashes(root); counts = Counter(); records = []
    c = previous.previous.cases(root)[0]; q = np.linspace(0., 1., 257)
    searches = {(r['seed'], r['method'], r['schedule'], r['ordering']): r for r in read(root/previous.previous.BASE/'development_v1/rows.json')}
    readings = {(r['seed'], r['method'], r['schedule'], r['ordering']): r for r in read(root/previous.BASE/'development_v1/rows.json')}
    for name in NAMES:
        for schedule, order in SETTINGS:
            key = c['seed'], name, schedule, order
            a, m = model.fit(c['x'], c['v'], q, seed=c['seed'], name=name, schedule=schedule, order_name=order)
            sa, ra = split(a)
            for source, current in [(searches[key], sa), (readings[key], ra)]:
                directory = root/source['directory']
                for f, value in source['files'].items(): assert sha(directory/f) == value
                counts['staged_arrays_bitwise_equal'] += previous.previous.same(previous.previous.load(directory/'arrays.npz'), current)
            assert m['completed'] and m['full_candidate_set_resolved'] and m['readout_available']
            counts.update(audit_canonical(a, m))
            directory = out/f'{schedule}_{order}_{name}'; directory.mkdir()
            np.savez_compressed(directory/'arrays.npz', **a); save(directory/'metadata.json', m); save(directory/'array_hashes.json', digests(a))
            records.append(dict(method=name, schedule=schedule, ordering=order, diagnostic_seconds=m['contiguous_seconds']))
    assert hashes(root) == h; save(out/'records.json', records)
    files = [str(p.relative_to(out)).replace('\\', '/') for p in out.rglob('*') if p.is_file()]
    result = dict(passed=True, calls=len(records), checks=dict(counts), source_sha256=h, seconds=time.perf_counter()-begin,
        costs_are_diagnostic_not_formal=True, outputs_sha256={f: sha(out/f) for f in files})
    save(out/'summary.json', result); print(dict(stage='preflight_complete', passed=True, calls=len(records), checks=dict(counts), seconds=result['seconds']), flush=True)


def run(root, out):
    begin = time.perf_counter(); h = hashes(root); assert complete(root/BASE/'preflight_v1')['source_sha256'] == h
    cc = previous.previous.cases(root); assert len(cc) == 16
    protocol = dict(source_sha256=h, primary=PRIMARY, methods=NAMES, settings=SETTINGS, repetitions=3,
        cases=[{k: v for k, v in c.items() if k not in ['x', 'v']} for c in cc],
        staged_source_sha256=sha(root/previous.BASE/'development_v1/rows.json'), query_targets_accessed=False,
        fresh_search_every_call=True, old_development_tasks=True, canonical_array_deduplication=True,
        primary_comparators={'bfs_observed': 'pdhg_cold1024', 'dfs_farthest_x': 'pdhg_box1024'},
        thread_counts=dict(blas=1, omp=1), core_research_goal_complete=False)
    save(out/'protocol.json', protocol)
    warm = legacy.old_join.load_cases(root, 'preflight')[0]; q = np.linspace(0., 1., 257); tick = time.perf_counter()
    for name in NAMES: model.fit(warm['x'], warm['v'], q, seed=384071, name=name, schedule='bfs')
    warmup_seconds = time.perf_counter()-tick; save(out/'warmup.json', dict(calls=len(NAMES), seconds=warmup_seconds, query_targets_accessed=False))
    rows = []
    for c in cc:
        jobs = [(name, schedule, order, rep) for name in NAMES for schedule, order in SETTINGS for rep in range(3)]
        canonical = {}
        for j in np.random.default_rng(np.random.SeedSequence([384929, c['seed']])).permutation(len(jobs)):
            name, schedule, order, rep = jobs[int(j)]; key = name, schedule, order
            a, m = model.fit(c['x'], c['v'], q, seed=c['seed'], name=name, schedule=schedule, order_name=order)
            directory = out/f"{c['seed']}_{schedule}_{order}_{name}_r{rep}"; directory.mkdir()
            digest = digests(a); actual = untimed(m); tick = time.perf_counter()
            if key not in canonical:
                np.savez_compressed(directory/'arrays.npz', **a)
                array_file = str((directory/'arrays.npz').relative_to(root)); array_sha = sha(root/array_file)
                canonical[key] = dict(digest=digest, metadata=actual, array_file=array_file, array_sha256=array_sha)
                first = True
            else:
                reference = canonical[key]
                # Preserve a diagnostic on mismatch instead of silently accepting a changed repeat.
                if digest != reference['digest'] or actual != reference['metadata']:
                    np.savez_compressed(directory/'mismatch_arrays.npz', **a); save(directory/'mismatch_metadata.json', m)
                    raise AssertionError(('Non-time repeat mismatch', c['seed'], key, rep))
                array_file = reference['array_file']; array_sha = reference['array_sha256']; first = False
            save(directory/'array_hashes.json', digest); save(directory/'metadata.json', m)
            disk_seconds = time.perf_counter()-tick
            row = dict(seed=c['seed'], method=name, schedule=schedule, ordering=order, repetition=rep, canonical=first,
                completed=m['completed'], resolved=m['full_candidate_set_resolved'], seconds=m['contiguous_seconds'],
                search_seconds=m['search_seconds'], readout_seconds=m['readout_seconds'],
                returned_array_bytes=m['returned_array_bytes'], disk_seconds=disk_seconds,
                directory=str(directory.relative_to(root)), array_file=array_file, array_sha256=array_sha,
                files={f: sha(directory/f) for f in ['array_hashes.json', 'metadata.json']})
            rows.append(row)
            if len(rows) % 20 == 0:
                print(dict(stage='contiguous_repeat', calls=len(rows), completed=sum(r['resolved'] for r in rows), seconds=time.perf_counter()-begin), flush=True)
        save(out/f"sealed_seed_{c['seed']}.json", rows[-60:])
    assert len(rows) == 960 and sum(r['canonical'] for r in rows) == 320 and hashes(root) == h
    save(out/'rows.json', rows)
    result = dict(passed=True, calls=len(rows), canonical_arrays=320, resolved=sum(r['resolved'] for r in rows),
        seconds=time.perf_counter()-begin, query_targets_accessed=False, core_research_goal_complete=False,
        all_outputs_sealed_before_audit=True, outputs_sha256={f: sha(out/f) for f in ['protocol.json', 'warmup.json', 'rows.json']})
    save(out/'summary.json', result); print(dict(stage='contiguous_complete', **result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--stage', choices=['preflight', 'run'], required=True); args = p.parse_args()
    assert os.environ.get('OPENBLAS_NUM_THREADS') == os.environ.get('OMP_NUM_THREADS') == '1'
    root = Path(__file__).resolve().parents[2]; out = root/BASE/('preflight_v1' if args.stage == 'preflight' else 'development_v1')
    out.mkdir(parents=True, exist_ok=False)
    try: globals()[args.stage](root, out)
    except Exception:
        save(out/'failure.json', dict(traceback=traceback.format_exc(), automatic_retry=False)); raise
