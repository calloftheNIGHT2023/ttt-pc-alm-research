"""333 descriptive post-seal accounting; no fitting or new significance tests."""
import hashlib
import json
import math
from pathlib import Path
import time
import traceback
import numpy as np

BASE = 'results/online_credit_fresh_pilot'
DESIGN = 'outputs/ttt-pc-alm-research/333_fresh_cost_readout_diagnostic_v1.md'
COMPONENTS = ['preparation_seconds', 'atomic_seconds', 'continuation_collection_seconds',
              'search_seconds', 'original_geometry_seconds', 'extra_geometry_seconds',
              'geometry_seconds', 'sampling_seconds', 'read_seconds']
EXTRA = ['credit_control_probe33', 'probe_all_alm64', 'probe_then_adam1920_33',
         'probe_then_adam3840_33']


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)


def moments(points, query, block=31, reverse=False):
    p = points[::-1] if reverse else points
    means, variances = [], []
    for start in range(0, len(query), block):
        h = np.broadcast_to(query[start:start+block], (len(p), len(query[start:start+block])))
        for layer in range(4):
            z = h + p[:, layer, None]
            # Independent tent expression, including both flat tails.
            h = np.maximum(0., 1. - np.abs(2. * z - 1.))
        means.extend(h.mean(axis=0))
        variances.extend(h.var(axis=0, ddof=1) if len(p) > 1 else np.zeros(h.shape[1]))
    return np.array(means), np.array(variances)


def selftest():
    q = np.array([0., .25, .5, .75, 1.])
    p = np.zeros((1, 4))
    m, v = moments(p, q)
    assert np.array_equal(m, np.zeros(5)) and np.array_equal(v, np.zeros(5))
    # Enumerate every two-draw sample from Bernoulli(.5): unbiased correction.
    risks, corrected = [], []
    target = .3
    for x in [0., 1.]:
        for y in [0., 1.]:
            a = np.array([x, y]); risk = float((a.mean()-target)**2)
            risks.append(risk); corrected.append(risk-float(a.var(ddof=1))/2)
    assert abs(np.mean(corrected)-(.5-target)**2) < 1e-15
    assert abs(np.mean(risks)-np.mean(corrected)-.25/2) < 1e-15
    return dict(tent_checks=10, unbiased_correction_samples=4)


def run(root, out):
    begin = time.perf_counter(); pred = root/BASE/'pilot_predictions_v2'; ev = root/BASE/'pilot_evaluation_v2'
    ps, es = read(pred/'summary.json'), read(ev/'summary.json')
    assert ps['passed'] and es['passed'] and ps['tasks'] == es['tasks'] == 512
    assert sha(pred/'rows.json') == ps['outputs_sha256']['rows.json']
    assert sha(ev/'task_metrics.npz') == es['outputs_sha256']['task_metrics.npz']
    rows = read(pred/'rows.json')
    with np.load(ev/'task_metrics.npz', allow_pickle=False) as z:
        seeds, names, risk = z['seeds'].tolist(), z['methods'].tolist(), z['risk'][:, :, 0]
    methods = [n for n in names if n.startswith('online_')] + EXTRA
    assert len(methods) == len(set(methods)) == 16
    sources = {str(Path(__file__).relative_to(root)): sha(Path(__file__)), DESIGN: sha(root/DESIGN)}
    write(out/'protocol.json', dict(source_sha256=sources, prediction_summary_sha256=sha(pred/'summary.json'),
        evaluation_summary_sha256=sha(ev/'summary.json'), methods=methods, seeds=seeds,
        posthoc_development_diagnostic=True, new_significance_tests=False, selftest=selftest(),
        query_targets_generated=False, reads_sealed_evaluation_risk=True, posterior_reference_accessed=False))
    lookup = {(r['seed'], r['method']): r for r in rows}; records = []; gaps = dict(prediction=0., order_mean=0., order_variance=0.)
    for i, seed in enumerate(seeds):
        for name in methods:
            row = lookup[seed, name]; mp, ap = pred/row['metadata_file'], pred/row['file']
            assert sha(mp) == row['metadata_sha256'] and sha(ap) == row['sha256']
            meta = read(mp)['metadata']; assert meta['charged_complete_seconds'] == row['seconds']
            assert not meta['query_targets_accessed']
            components = {k: float(meta.get(k, 0.)) for k in COMPONENTS}
            # Alternative geometry field names cannot both be active.
            assert not (components['geometry_seconds'] and (components['original_geometry_seconds'] or components['extra_geometry_seconds']))
            remainder = row['seconds'] - math.fsum(components.values())
            assert remainder >= -1e-8, (seed, name, remainder)
            with np.load(ap, allow_pickle=False) as z:
                points, query, prediction = z['points'], z['q_observed'], z['prediction']
                assert points.ndim == 2 and points.shape[1] == 4 and len(points) in [1, 2048]
                mean, variance = moments(points, query)
                other_mean, other_variance = moments(points, query, block=19, reverse=True)
                gaps['prediction'] = max(gaps['prediction'], float(np.max(abs(mean-prediction))))
                gaps['order_mean'] = max(gaps['order_mean'], float(np.max(abs(mean-other_mean))))
                gaps['order_variance'] = max(gaps['order_variance'], float(np.max(abs(variance-other_variance))))
                assert max(gaps.values()) < 2e-12
                assert np.all(variance >= 0)
                variance_of_mean = math.fsum(map(float, variance))/len(query)/len(points)
                # Sample variance bound differs from population bound by M/(M-1).
                assert len(points) == 1 or variance_of_mean <= 1/(4*(len(points)-1))+1e-15
            actual = float(risk[i, names.index(name)])
            record = dict(seed=seed, method=name, charged_seconds=row['seconds'], components=components,
                unallocated_seconds=remainder, overlapping_trigger_seconds=meta.get('trigger_selection_seconds'),
                overlapping_credit_seconds=meta.get('credit_seconds'),
                dp_table_seconds=(meta.get('proposal') or {}).get('meta', {}).get('table_seconds'),
                dp_selection_seconds=(meta.get('proposal') or {}).get('meta', {}).get('selection_seconds'),
                particles=len(points), execution_failed=row['execution_failed'], actual_mse=actual,
                estimated_readout_variance=variance_of_mean, corrected_fixed_teacher_risk_estimate=actual-variance_of_mean)
            records.append(record)
        if (i+1) % 32 == 0:
            print(dict(tasks=i+1, methods=16, seconds=time.perf_counter()-begin), flush=True)
    aggregate = []
    fields = ['charged_seconds', 'unallocated_seconds', 'overlapping_trigger_seconds', 'overlapping_credit_seconds',
              'dp_table_seconds', 'dp_selection_seconds', 'actual_mse', 'estimated_readout_variance', 'corrected_fixed_teacher_risk_estimate']
    for name in methods:
        rr = [r for r in records if r['method'] == name]; assert len(rr) == 512
        result = dict(method=name, tasks=512, single_point_fallbacks=sum(r['particles'] == 1 for r in rr),
            execution_failures=sum(r['execution_failed'] for r in rr),
            components={k: math.fsum(r['components'][k] for r in rr)/512 for k in COMPONENTS})
        for field in fields:
            values = [r[field] for r in rr if r[field] is not None]
            assert len(values) in [0, 512]
            result[field] = math.fsum(values)/512 if values else None
        aggregate.append(result)
    for p, digest in sources.items(): assert sha(root/p) == digest
    write(out/'tasks.json', records); write(out/'methods.json', aggregate)
    summary = dict(passed=True, tasks=512, methods=16, predictors=len(records), max_gaps=gaps,
        seconds=time.perf_counter()-begin, no_new_significance_tests=True, core_research_goal_complete=False,
        outputs_sha256={n: sha(out/n) for n in ['protocol.json', 'tasks.json', 'methods.json']})
    write(out/'summary.json', summary); print(summary, flush=True)


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[2]; out = root/BASE/'cost_readout_diagnostic_v1'
    out.mkdir(parents=True, exist_ok=False)
    try: run(root, out)
    except Exception:
        write(out/'failure.json', dict(traceback=traceback.format_exc(), automatic_retry=False)); raise
