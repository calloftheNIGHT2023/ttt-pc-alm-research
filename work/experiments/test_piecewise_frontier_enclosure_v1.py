"""422 exact point equivalence, boundary behavior and independent containment."""
from collections import Counter
from fractions import Fraction as F
from pathlib import Path
import time
import traceback
import numpy as np
import piecewise_frontier_enclosure_v1 as model
from test_frontier_range_certificate_v1 import line_extrema
import deadline_risk_io_v1 as io

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'results/piecewise_frontier/preflight_v1'
DESIGN = 'outputs/ttt-pc-alm-research/422_piecewise_frontier_readout_protocol_v1.md'


def main():
    begin = time.perf_counter(); rng = np.random.default_rng(422731); counts = Counter()
    sources = {p.relative_to(ROOT).as_posix(): io.sha(p) for p in
               sorted((ROOT/'work/experiments').glob('*.py'))+[ROOT/DESIGN]}
    io.save(OUT/'protocol.json', dict(source_sha256=sources, query_targets_accessed=False,
        original_enclosure='frontier_range_certificate_v1.enclose', max_segments=4096))
    identity = tuple(tuple(F(int(i == j)) for j in range(4)) for i in range(4))
    inverses = [identity]
    while len(inverses) < 17:
        matrix = rng.integers(-3, 4, (4, 4)).tolist()
        if model.reference.core.exact_rank(matrix) == 4:
            inverses.append(model.reference.inverse(matrix))
    records = []
    for ii, inv in enumerate(inverses):
        centers = [F(int(n), 100) for n in rng.integers(-10, 11, 4)]
        boxes = [[(-F(.12), F(.12))]*4, [(c-F(1, 1000), c+F(1, 1000)) for c in centers],
                 [(c, c) for c in centers]]
        for bi, box in enumerate(boxes):
            compiled = model.compile_box(inv, box)
            assert compiled['compiled']
            queries = set(F(i, 256) for i in range(257)) | set(compiled['knots']) | {F(-1, 4), F(5, 4)}
            for a, b in zip(compiled['knots'], compiled['knots'][1:]):
                queries.update([(3*a+b)/4, (a+b)/2, (a+3*b)/4])
            routes = Counter()
            for q in sorted(queries):
                actual, crossing, route = model.evaluate(compiled, q)
                expected, reference_crossing = model.reference.enclose(q, inv, box)
                assert actual == expected and crossing == reference_crossing, (ii, bi, q, actual, expected)
                routes[route] += 1; counts['exact_point_and_crossing_matches'] += 1
            assert routes['compiled'] > 0 and routes['boundary_reference'] == len(compiled['knots'])
            assert routes['outside_reference'] == 2
            for _ in range(8):
                q = F(int(rng.integers(0, 1001)), 1000)
                z = [a+(b-a)*F(int(rng.integers(0, 1001)), 1000) for a, b in box]
                theta = [sum((a*t for a, t in zip(row, z)), F(0)) for row in inv]
                bounds, _, _ = model.evaluate(compiled, q)
                truth = model.reference.core.mathcore.forward(q, theta)
                assert bounds[0] <= truth <= bounds[1]
                if bi == 2:
                    assert bounds[0] == truth == bounds[1]
                    counts['zero_width_exact_outputs'] += 1
                counts['exact_box_point_containments'] += 1
            records.append(dict(matrix_index=ii, box_index=bi, segments=len(compiled['segments']),
                routes=dict(routes), layer_counts=compiled['layer_segment_counts']))
            counts['compiled_boxes'] += 1
        constants = [F(int(n), 100) for n in rng.integers(-3, 4, 3)]
        box = [(F(-1, 20), F(1, 20))]+[(c, c) for c in constants]
        compiled = model.compile_box(inv, box)
        for j in range(8):
            q = F(j, 7); exact = line_extrema(q, inv, box[0][0], box[0][1], constants)
            bounds, _, _ = model.evaluate(compiled, q)
            assert bounds[0] <= exact[0] <= exact[1] <= bounds[1]
            counts['independent_line_extrema_containments'] += 1
    wide = [(F(0), F(1))]+[(F(0), F(0))]*3
    compiled = model.compile_box(identity, wide)
    bounds, _, _ = model.evaluate(compiled, F(0))
    assert bounds == (0, 1); counts['interior_peak_not_vertex_only'] += 1
    capped = model.compile_box(identity, [(-F(.12), F(.12))]*4, max_segments=1)
    assert not capped['compiled'] and capped['segments'] == [] and capped['fallback'] == 'whole_original_enclosure'
    for q in map(lambda j: F(j, 256), range(257)):
        bounds, crossings, route = model.evaluate(capped, q)
        assert route == 'cap_reference' and (bounds, crossings) == model.reference.enclose(q, identity, capped['intervals'])
        counts['whole_box_cap_fallback_matches'] += 1
    io.verify_hashes(ROOT, sources); io.save(OUT/'boxes.json', records)
    summary = dict(passed=True, counts=dict(counts), seconds=time.perf_counter()-begin,
        source_sha256=sources, query_targets_accessed=False, task_risk_compared=False,
        exact_network_extrema_claim=False, outputs_sha256={f: io.sha(OUT/f) for f in ['protocol.json', 'boxes.json']})
    io.save(OUT/'summary.json', summary)
    print({k: v for k, v in summary.items() if k != 'source_sha256'}, flush=True)


if __name__ == '__main__':
    # This guard prevents accidental concurrent numerical work during run419.
    assert io.read(ROOT/'results/frontier_deadline/evaluation_v1/summary.json')['passed']
    OUT.mkdir(parents=True, exist_ok=False)
    try:
        main()
    except Exception:
        io.save(OUT/'failure.json', dict(traceback=traceback.format_exc(), automatic_retry=False))
        raise
