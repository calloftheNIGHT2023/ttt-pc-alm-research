"""422 archived-coordinate-box component comparison; not continuous timing."""
from collections import Counter
from fractions import Fraction as F
from pathlib import Path
import pickle
import time
import traceback
import numpy as np
import piecewise_frontier_enclosure_v1 as model
import candidate_set_readout_v1 as serializer
import deadline_risk_io_v1 as io

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'results/piecewise_frontier'
OUT = BASE/'component_v1'
DESIGN = 'outputs/ttt-pc-alm-research/422_piecewise_frontier_readout_protocol_v1.md'


def original(boxes, q):
    begin = time.perf_counter(); bounds = []; crossings = 0
    for box in boxes:
        one = []
        for point in q:
            interval, count = model.reference.enclose(F(float(point)), box['inverse'], box['intervals'])
            one.append(interval); crossings += count
        bounds.append(one)
    return dict(bounds=bounds, crossings=crossings, seconds=time.perf_counter()-begin)


def compiled(boxes, q):
    begin = time.perf_counter(); programs = [model.compile_box(b['inverse'], b['intervals']) for b in boxes]
    compile_seconds = time.perf_counter()-begin; tick = time.perf_counter()
    outputs = [model.evaluate_many(p, [F(float(v)) for v in q]) for p in programs]
    read_seconds = time.perf_counter()-tick
    return dict(bounds=[p['intervals'] for p in outputs], crossings=sum(p['crossing_operations'] for p in outputs),
        compile_seconds=compile_seconds, read_seconds=read_seconds, seconds=time.perf_counter()-begin,
        programs=programs, routes=dict(sum((Counter(p['routes']) for p in outputs), Counter())))


def main():
    begin = time.perf_counter(); gate = io.read(BASE/'preflight_v1/summary.json')
    assert gate['passed']; io.verify_hashes(ROOT, gate['source_sha256'])
    source = ROOT/'results/continuous_frontier/preflight_v1'; checked = io.read(source/'summary.json')
    assert checked['passed']; io.verify_hashes(ROOT, checked['source_sha256'])
    for name, digest in checked['outputs_sha256'].items():
        assert io.sha(source/name) == digest
    rows = [r for r in io.read(source/'direct_rows.json') if r['branch'] == 'frontier']
    sources = {p.relative_to(ROOT).as_posix(): io.sha(p) for p in
               sorted((ROOT/'work/experiments').glob('*.py'))+[ROOT/DESIGN]}
    io.save(OUT/'protocol.json', dict(source_sha256=sources, preflight_sha256=io.sha(BASE/'preflight_v1/summary.json'),
        archived_source_summary_sha256=io.sha(source/'summary.json'), query_targets_accessed=False,
        archived_coordinate_boxes_used=True, common_box_preparation_excluded_from_both=True,
        compile_cost_included=True, hard_deadline_enforced=False, independent_confirmation=False,
        expected_contexts=len(rows), queries=257))
    report = []; counts = Counter()
    for i, row in enumerate(rows):
        folder = ROOT/row['directory']; a = io.load_arrays(folder/'arrays.npz'); m = io.read(folder/'metadata.json')
        assert m['branch'] == 'frontier' and len(a['q']) == 257
        boxes = [dict(inverse=tuple(tuple(map(F, r)) for r in b['inverse']),
                      intervals=tuple(tuple(map(F, r)) for r in b['intervals'])) for b in m['range_readout']['coordinate_boxes']]
        identity = tuple(tuple(F(int(i == j)) for j in range(4)) for i in range(4))
        boxes.append(dict(inverse=identity, intervals=[(-model.reference.core.BOUND, model.reference.core.BOUND)]*4))
        if i % 2:
            new = compiled(boxes, a['q']); old = original(boxes, a['q'])
        else:
            old = original(boxes, a['q']); new = compiled(boxes, a['q'])
        assert new['bounds'] == old['bounds'] and new['crossings'] == old['crossings']
        assert serializer.serializable(old['bounds'][:-1]) == m['range_readout']['cell_query_ranges']
        result = dict(method=row['method'], n=row['n'], source_directory=row['directory'],
            cells=len(boxes)-1, queries=len(a['q']), original_seconds=old['seconds'],
            compile_seconds=new['compile_seconds'], compiled_read_seconds=new['read_seconds'],
            compile_plus_read_seconds=new['seconds'], compiled_first=bool(i % 2),
            total_segments=sum(len(p['segments']) for p in new['programs']),
            capped_boxes=sum(not p['compiled'] for p in new['programs']),
            routes=new['routes'], program_pickle_bytes=len(pickle.dumps(new['programs'], protocol=5)),
            program_bytes_are_not_peak_RSS=True)
        report.append(result); counts['contexts'] += 1; counts['exact_box_query_matches'] += len(boxes)*len(a['q'])
        io.save(OUT/f'case_{i}_programs.json', serializer.serializable(new['programs']))
        print(result, flush=True)
    io.verify_hashes(ROOT, sources); io.save(OUT/'rows.json', report)
    summary = dict(passed=True, counts=dict(counts), query_targets_accessed=False, task_risk_compared=False,
        archived_search_used=True, continuous_speedup_established=False, hard_deadline_enforced=False,
        seconds=time.perf_counter()-begin,
        outputs_sha256={p.name: io.sha(p) for p in OUT.iterdir() if p.is_file()})
    io.save(OUT/'summary.json', summary); print(summary, flush=True)


if __name__ == '__main__':
    assert io.read(ROOT/'results/frontier_deadline/evaluation_v1/summary.json')['passed']
    OUT.mkdir(parents=True, exist_ok=False)
    try:
        main()
    except Exception:
        io.save(OUT/'failure.json', dict(traceback=traceback.format_exc(), automatic_retry=False))
        raise
