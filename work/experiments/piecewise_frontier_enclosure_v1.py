"""422 compile the frozen scalar query enclosure, not exact network extrema.

All roots/coefficients are Fraction. Original point evaluation is retained at
every partition boundary, outside the domain, or after a segment-cap fallback.
"""
from bisect import bisect_right
from collections import Counter
from fractions import Fraction as F
import time
import frontier_range_certificate_v1 as reference

ZERO = (F(0), F(0))
ONE = (F(0), F(1))


def value(line, q):
    return line[0]*q+line[1]


def add(*lines):
    return tuple(sum((line[j] for line in lines), F(0)) for j in range(2))


def scale(line, amount):
    return tuple(F(amount)*v for v in line)


def partition(left, right, lines, levels):
    knots = {left, right}
    for slope, offset in lines:
        if slope:
            for threshold in levels:
                point = (F(threshold)-offset)/slope
                if left < point < right:
                    knots.add(point)
    ordered = sorted(knots)
    return list(zip(ordered, ordered[1:]))


def tent_line(line, midpoint):
    point = value(line, midpoint)
    if point <= 0 or point >= 1:
        return ZERO
    if point <= F(1, 2):
        return scale(line, 2)
    return add(scale(line, -2), (F(0), F(2)))


def scalar_limits(coeff, offset, error_low, error_high, intervals):
    low, high = reference.affine_bounds(coeff, F(0), intervals)
    return add(offset, (F(0), low), error_low), add(offset, (F(0), high), error_high)


def compile_box(inverse, intervals, *, max_segments=4096):
    begin = time.perf_counter()
    inv = tuple(tuple(map(F, row)) for row in inverse)
    box = tuple(tuple(map(F, pair)) for pair in intervals)
    d = len(inv)
    assert d > 0 and len(box) == d and all(len(row) == d for row in inv)
    assert all(len(pair) == 2 and pair[0] <= pair[1] for pair in box)
    assert type(max_segments) is int and max_segments > 0
    zero = (F(0),)*d
    # Segment: left, right, parameter coefficient, q-offset, error bounds, crossings.
    states = [(F(0), F(1), zero, (F(1), F(0)), ZERO, ZERO, 0)]
    trace = []; peak_segments = 1
    def fallback(layer, count):
        return dict(compiled=False, inverse=inv, intervals=box, segments=[], knots=[],
            layer_segment_counts=trace, peak_segments=max(peak_segments, count),
            cap_reached_layer=layer, max_segments=max_segments, compile_seconds=time.perf_counter()-begin,
            fallback='whole_original_enclosure', exact_rational=True, query_targets_accessed=False)
    for layer in range(d):
        next_states = []
        for left, right, coeff, offset, el, eh, crossings in states:
            z = tuple(a+b for a, b in zip(coeff, inv[layer]))
            low, high = scalar_limits(z, offset, el, eh, box)
            for a, b in partition(left, right, [low, high], [F(0), F(1, 2), F(1)]):
                midpoint = (a+b)/2; lv, hv = value(low, midpoint), value(high, midpoint)
                assert lv <= hv
                if hv <= 0 or lv >= 1:
                    next_states.append((a, b, zero, ZERO, ZERO, ZERO, crossings))
                elif 0 <= lv <= hv <= F(1, 2):
                    next_states.append((a, b, tuple(2*t for t in z), scale(offset, 2), scale(el, 2), scale(eh, 2), crossings))
                elif F(1, 2) <= lv <= hv <= 1:
                    next_states.append((a, b, tuple(-2*t for t in z), add(scale(offset, -2), (F(0), F(2))),
                                        scale(eh, -2), scale(el, -2), crossings))
                else:
                    gl, gh = tent_line(low, midpoint), tent_line(high, midpoint)
                    for aa, bb in partition(a, b, [add(gl, scale(gh, -1))], [0]):
                        middle = (aa+bb)/2
                        if value(gl, middle) <= value(gh, middle):
                            minimum, maximum = gl, gh
                        else:
                            minimum, maximum = gh, gl
                        if value(low, middle) <= F(1, 2) <= value(high, middle):
                            maximum = ONE
                        next_states.append((aa, bb, zero, ZERO, minimum, maximum, crossings+1))
                if len(next_states) > max_segments:
                    return fallback(layer, len(next_states))
        states = next_states; trace.append(len(states)); peak_segments = max(peak_segments, len(states))
        assert states[0][0] == 0 and states[-1][1] == 1
        assert all(a[1] == b[0] for a, b in zip(states, states[1:]))
    segments = []
    for left, right, coeff, offset, el, eh, crossings in states:
        low, high = scalar_limits(coeff, offset, el, eh, box)
        for a, b in partition(left, right, [low, high], [0, 1]):
            midpoint = (a+b)/2
            lo = ZERO if value(low, midpoint) < 0 else low
            hi = ONE if value(high, midpoint) > 1 else high
            assert 0 <= value(lo, midpoint) <= value(hi, midpoint) <= 1
            segments.append(dict(left=a, right=b, low=lo, high=hi, crossings=crossings))
            if len(segments) > max_segments:
                return fallback(d, len(segments))
    assert segments and segments[0]['left'] == 0 and segments[-1]['right'] == 1
    assert all(a['right'] == b['left'] for a, b in zip(segments, segments[1:]))
    knots = [s['left'] for s in segments]+[F(1)]
    assert len(knots) == len(set(knots))
    return dict(compiled=True, inverse=inv, intervals=box, segments=segments, knots=knots,
        layer_segment_counts=trace, peak_segments=max(peak_segments, len(segments)),
        max_segments=max_segments, compile_seconds=time.perf_counter()-begin,
        exact_rational=True, boundary_uses_original=True, query_targets_accessed=False,
        exact_network_extrema_claim=False)


def evaluate(compiled, q):
    point = F(q)
    route = None
    if not compiled['compiled']:
        route = 'cap_reference'
    elif not 0 <= point <= 1:
        route = 'outside_reference'
    else:
        index = bisect_right(compiled['knots'], point)-1
        if compiled['knots'][index] == point:
            route = 'boundary_reference'
    if route is not None:
        bound, crossings = reference.enclose(point, compiled['inverse'], compiled['intervals'])
        return bound, crossings, route
    segment = compiled['segments'][index]
    assert segment['left'] < point < segment['right']
    bound = value(segment['low'], point), value(segment['high'], point)
    assert 0 <= bound[0] <= bound[1] <= 1
    return bound, segment['crossings'], 'compiled'


def evaluate_many(compiled, queries):
    begin = time.perf_counter(); bounds = []; crossings = 0; routes = Counter()
    for q in queries:
        bound, count, route = evaluate(compiled, q)
        bounds.append(bound); crossings += count; routes[route] += 1
    return dict(intervals=bounds, crossing_operations=crossings, routes=dict(routes),
                reading_seconds=time.perf_counter()-begin)
