"""384 canonical full proof replay, byte-identical repeats and paired timing."""
from collections import Counter, defaultdict
from pathlib import Path
import math
import time
import traceback
import numpy as np
import audit_retired_region_join_v1 as search_audit
import complete_credit_mode_geometry_v1 as classifier
import run_contiguous_regional_repeat_v1 as run


def audit_canonical(a, m):
    sa, ra = run.split(a); rm = m['readout']; positive = rm['positive_modes']; counts = Counter()
    assert ra['x'].tobytes() == sa['original_x'].tobytes() and ra['v'].tobytes() == sa['original_v'].tobytes()
    assert ra['search_regions'].tobytes() == sa['regions'].tobytes()
    assert np.array_equal(ra['q'], np.linspace(0., 1., 257))
    keys = sorted(r.tobytes().hex() for r in ra['search_regions'])
    assert keys == rm['classified_modes'] == sorted(rm['classifications_detail'])
    expected_positive = []; unresolved = []; kinds = Counter()
    for key in keys:
        note = rm['classifications_detail'][key]; _, aa, rhs, _, _ = classifier.matrices(ra['x'], ra['v'], key)
        counts['exact_geometry_certificates'] += classifier.verify_certificates(aa, rhs, note)
        strict = any(c.get('strict_interior', False) for c in note['certificates'])
        infeasible = any(c['type'] == 'negative_constant_row' or c.get('conclusion') == 'infeasible' for c in note['certificates'])
        zero = any(c.get('conclusion') == 'zero_volume_or_empty' for c in note['certificates'])
        assert not (strict and (infeasible or zero))
        if note['classification'] == 'positive_volume': assert strict and note['positive_volume_certified']
        elif note['classification'] == 'infeasible': assert infeasible and not strict
        elif note['classification'] == 'zero_volume_or_empty': assert zero and not (strict or infeasible)
        else: assert note['classification'] == 'unresolved' and not (strict or infeasible or zero)
        if note['numerical_volume_available']:
            assert note['classification'] == 'positive_volume'; expected_positive.append(key)
        if note['classification'] == 'unresolved' or (note['classification'] == 'positive_volume' and not note['numerical_volume_available']): unresolved.append(key)
        kinds[note['classification']] += 1
    assert positive == expected_positive and unresolved == rm['unresolved_modes'] and dict(kinds) == rm['classification_counts']
    assert m['completed'] == rm['search_completed'] == m['search']['completed']
    assert m['full_candidate_set_resolved'] == rm['full_candidate_set_resolved'] == (m['completed'] and not unresolved)
    assert m['readout_available'] == rm['readout_available'] == bool(positive)
    if positive:
        assert np.array_equal(ra['positive_volumes'], [rm['classifications_detail'][k]['volume'] for k in positive])
        assert int(ra['allocation'].sum()) == len(ra['points']) == rm['particles']
        for point in ra['points']:
            assert np.max(abs(run.model.reader.shared.geometry.base.forward(ra['x'], point)-ra['v'])) <= .001+1e-7
            counts['posterior_support_checks'] += 1
        pred = run.model.reader.project(run.model.reader.shared.geometry.make_predict(ra['points'])(ra['q']))
        assert pred.tobytes() == ra['prediction'].tobytes(); counts['prediction_replays'] += 1
    else: assert ra['prediction'].size == ra['points'].size == ra['allocation'].size == 0
    counts.update(search_audit.audit_case(sa, m['search'], positive, exact=True))
    assert m['search_seconds'] == m['search']['total_seconds'] and m['readout_seconds'] == rm['charged_readout_seconds']
    assert m['contiguous_seconds'] >= m['search_seconds']+m['readout_seconds']
    assert m['returned_array_bytes'] == sum(value.nbytes for value in a.values())
    assert rm['returned_array_bytes'] == sum(value.nbytes for value in ra.values())
    assert not any(m[k] for k in ['query_targets_accessed', 'archived_search_used', 'global_bp_used', 'cross_method_cache_used'])
    counts['continuous_pipeline_checks'] += 1
    return counts


def audit(root, out):
    begin = time.perf_counter(); source = root/run.BASE/'development_v1'; summary = run.complete(source)
    protocol = run.read(source/'protocol.json'); assert summary['all_outputs_sealed_before_audit'] and protocol['source_sha256'] == run.hashes(root)
    rows = run.read(source/'rows.json'); assert len(rows) == 960
    counts = Counter(); references = {}; group_rows = defaultdict(list); records = []
    for index, row in enumerate(rows):
        directory = root/row['directory']
        for f, value in row['files'].items(): assert run.sha(directory/f) == value
        m = run.read(directory/'metadata.json'); digest = run.read(directory/'array_hashes.json')
        assert run.sha(root/row['array_file']) == row['array_sha256']
        key = row['seed'], row['method'], row['schedule'], row['ordering']
        if row['canonical']:
            assert key not in references
            a = run.previous.previous.load(root/row['array_file']); assert run.digests(a) == digest
            one = audit_canonical(a, m); counts.update(one)
            references[key] = dict(digest=digest, metadata=run.untimed(m), file=row['array_file'], sha=row['array_sha256'])
            records.append(dict(directory=row['directory'], checks=dict(one)))
        else:
            ref = references[key]
            assert row['array_file'] == ref['file'] and row['array_sha256'] == ref['sha']
            assert digest == ref['digest'] and run.untimed(m) == ref['metadata']
            counts['deduplicated_repeat_array_digests'] += len(digest)
            counts['identical_non_time_repeat_metadata'] += 1
        assert row['seconds'] == m['contiguous_seconds'] and row['search_seconds'] == m['search_seconds'] and row['readout_seconds'] == m['readout_seconds']
        assert row['completed'] == m['completed'] and row['resolved'] == m['full_candidate_set_resolved']
        group_rows[key].append(row)
        if (index+1) % 60 == 0: print(dict(stage='repeat_audit', calls=index+1, canonical=len(references), seconds=time.perf_counter()-begin), flush=True)
    task_groups = []
    for key, rr in group_rows.items():
        seed, method, schedule, order = key; assert sorted(r['repetition'] for r in rr) == [0, 1, 2]
        task_groups.append(dict(seed=seed, method=method, schedule=schedule, ordering=order,
            all_resolved=all(r['resolved'] for r in rr), times=[r['seconds'] for r in sorted(rr, key=lambda r:r['repetition'])],
            median_seconds=float(np.median([r['seconds'] for r in rr])), mean_seconds=math.fsum(r['seconds'] for r in rr)/3))
    lookup = {(g['seed'], g['method'], g['schedule'], g['ordering']): g for g in task_groups}
    comparisons = []
    for si, (schedule, order) in enumerate(run.SETTINGS):
        for ci, control in enumerate(n for n in run.NAMES if n != run.PRIMARY):
            seeds = sorted({r['seed'] for r in rows})
            primary = [lookup[s, run.PRIMARY, schedule, order] for s in seeds]
            other = [lookup[s, control, schedule, order] for s in seeds]
            delta = np.array([p['median_seconds']-c['median_seconds'] for p, c in zip(primary, other)])
            rng = np.random.default_rng(np.random.SeedSequence([384991, si, ci]))
            boot = delta[rng.integers(0, len(delta), (20000, len(delta)))].mean(1)
            low, high = np.quantile(boot, [.025, .975])
            comparisons.append(dict(schedule=schedule, ordering=order, primary=run.PRIMARY, control=control,
                predeclared_primary_comparator=control == protocol['primary_comparators'][schedule+'_'+order],
                all_primary_resolved=all(p['all_resolved'] for p in primary), all_control_resolved=all(c['all_resolved'] for c in other),
                mean_primary_median=math.fsum(p['median_seconds'] for p in primary)/16,
                mean_control_median=math.fsum(c['median_seconds'] for c in other)/16,
                mean_paired_difference=float(delta.mean()), bootstrap95=[float(low), float(high)],
                primary_faster_tasks=int((delta < 0).sum()), independent_tasks=16, technical_repeats=3))
    assert len(references) == 320 and protocol['source_sha256'] == run.hashes(root)
    run.save(out/'checks.json', records); run.save(out/'task_groups.json', task_groups); run.save(out/'comparisons.json', comparisons)
    result = dict(passed=True, calls=len(rows), checks=dict(counts), seconds=time.perf_counter()-begin,
        query_targets_accessed=False, core_research_goal_complete=False, development_summary_sha256=run.sha(source/'summary.json'),
        outputs_sha256={f: run.sha(out/f) for f in ['checks.json', 'task_groups.json', 'comparisons.json']})
    run.save(out/'summary.json', result); print(dict(stage='repeat_audit_complete', **result), flush=True)


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[2]; out = root/run.BASE/'audit_v1'; out.mkdir(parents=True, exist_ok=False)
    try: audit(root, out)
    except Exception:
        run.save(out/'failure.json', dict(traceback=traceback.format_exc(), automatic_retry=False)); raise
