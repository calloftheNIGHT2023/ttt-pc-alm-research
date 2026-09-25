"""382 serialized exact geometry, support validity and common readout replay."""
from collections import Counter, defaultdict
from pathlib import Path
import math
import time
import traceback
import numpy as np
import complete_credit_mode_geometry_v1 as classifier
import run_candidate_set_readout_v1 as run


def audit(root, out):
    begin = time.perf_counter(); source = root/run.BASE/'development_v1'; summary = run.complete(source)
    protocol = run.read(source/'protocol.json'); assert summary['all_readouts_sealed_before_audit'] and protocol['source_sha256'] == run.hashes(root)
    rows = run.read(source/'rows.json'); assert len(rows) == 544
    search = {(r['seed'], r['method'], r['schedule'], r['ordering']): r for r in protocol['source_rows']}
    counts = Counter(); references = {}; groups = defaultdict(list); records = []
    for index, row in enumerate(rows):
        directory = root/row['directory']
        for f, value in row['files'].items(): assert run.sha(directory/f) == value
        a = run.previous.load(directory/'arrays.npz'); m = run.read(directory/'metadata.json')
        source_row = search[row['seed'], row['method'], row['schedule'], row['ordering']]
        for f, value in source_row['files'].items(): assert run.sha(root/source_row['directory']/f) == value
        original = run.previous.load(root/source_row['directory']/'arrays.npz')
        assert a['x'].tobytes() == original['original_x'].tobytes() and a['v'].tobytes() == original['original_v'].tobytes()
        assert a['search_regions'].tobytes() == original['regions'].tobytes()
        assert np.array_equal(a['q'], np.linspace(0., 1., 257))
        keys = sorted(r.tobytes().hex() for r in a['search_regions'])
        assert keys == m['classified_modes'] == sorted(m['classifications_detail'])
        positive = []; unresolved = []; one = Counter(); kinds = Counter()
        for key in keys:
            note = m['classifications_detail'][key]; _, matrix, rhs, _, _ = classifier.matrices(a['x'], a['v'], key)
            one['exact_geometry_certificates'] += classifier.verify_certificates(matrix, rhs, note)
            assert note['mode'] == key and note['observations'] == len(a['x']) and note['depth'] == 4
            strict = any(c.get('strict_interior', False) for c in note['certificates'])
            infeasible = any(c['type'] == 'negative_constant_row' or c.get('conclusion') == 'infeasible' for c in note['certificates'])
            zero = any(c.get('conclusion') == 'zero_volume_or_empty' for c in note['certificates'])
            assert not (strict and (infeasible or zero))
            if note['classification'] == 'positive_volume': assert strict and note['positive_volume_certified']
            elif note['classification'] == 'infeasible': assert infeasible and not strict
            elif note['classification'] == 'zero_volume_or_empty': assert zero and not (strict or infeasible)
            else: assert note['classification'] == 'unresolved' and not (strict or infeasible or zero)
            kinds[note['classification']] += 1
            if note['numerical_volume_available']:
                assert note['classification'] == 'positive_volume' and note['positive_volume_certified']
                positive.append(key)
            if note['classification'] == 'unresolved' or (note['classification'] == 'positive_volume' and not note['numerical_volume_available']): unresolved.append(key)
        assert positive == m['positive_modes'] and unresolved == m['unresolved_modes'] and dict(kinds) == m['classification_counts'] == row['classification_counts']
        assert m['search_completed'] == row['completed'] == source_row['completed']
        assert m['full_candidate_set_resolved'] == row['full_candidate_set_resolved'] == (row['completed'] and not unresolved)
        assert m['readout_available'] == row['readout_available'] == bool(positive)
        assert not any(m[k] for k in ['global_bp_used', 'query_targets_accessed', 'mother_trajectory_called', 'cross_method_cache_used', 'hidden_recovery_used'])
        assert m['returned_array_bytes'] == row['returned_array_bytes'] == sum(t.nbytes for t in a.values())
        assert row['search_seconds'] == source_row['seconds'] and row['readout_seconds'] == m['charged_readout_seconds']
        assert row['staged_total_seconds'] == row['search_seconds']+row['readout_seconds']
        if positive:
            assert np.array_equal(a['positive_volumes'], [m['classifications_detail'][k]['volume'] for k in positive])
            assert len(a['allocation']) == len(positive) and int(a['allocation'].sum()) == len(a['points']) == m['particles']
            prediction = run.model.project(run.model.shared.geometry.make_predict(a['points'])(a['q']))
            assert prediction.tobytes() == a['prediction'].tobytes()
            for point in a['points']:
                assert np.max(abs(run.model.shared.geometry.base.forward(a['x'], point)-a['v'])) <= .001+1e-7
                one['posterior_support_checks'] += 1
            one['prediction_replays'] += 1
        else:
            assert a['prediction'].size == a['points'].size == a['allocation'].size == a['positive_volumes'].size == 0
            one['explicit_unavailable_readouts'] += 1
        if m['full_candidate_set_resolved']:
            assert positive, 'A completed noisily feasible task has no positive region'
            fields = ['positive_volumes', 'allocation', 'points', 'prediction']
            if row['seed'] not in references: references[row['seed']] = (positive, {k: a[k] for k in fields})
            else:
                expected_keys, expected_arrays = references[row['seed']]
                assert positive == expected_keys
                one['common_complete_readout_arrays'] += run.previous.same(expected_arrays, {k: a[k] for k in fields})
            one['full_candidate_readout_checks'] += 1
        counts.update(one); groups[row['method'], row['schedule'], row['ordering']].append(row)
        records.append(dict(directory=row['directory'], checks=dict(one)))
        if (index+1) % 34 == 0: print(dict(stage='readout_audit', calls=index+1, checks=dict(counts), seconds=time.perf_counter()-begin), flush=True)
    result_groups = []
    for (method, schedule, order), rr in groups.items():
        g = dict(method=method, schedule=schedule, ordering=order, calls=len(rr),
            completed=sum(r['completed'] for r in rr), resolved=sum(r['full_candidate_set_resolved'] for r in rr),
            available=sum(r['readout_available'] for r in rr), candidates=sum(r['remaining'] for r in rr),
            geometry_numeric_bytes=sum(r['geometry_numeric_bytes'] for r in rr), returned_array_bytes=sum(r['returned_array_bytes'] for r in rr))
        for key in ['search_seconds', 'readout_seconds', 'staged_total_seconds', 'geometry_seconds', 'sampling_seconds', 'reading_seconds']:
            g['mean_'+key] = math.fsum(r[key] for r in rr)/len(rr)
        result_groups.append(g)
    run.save(out/'groups.json', result_groups); run.save(out/'checks.json', records)
    assert protocol['source_sha256'] == run.hashes(root)
    result = dict(passed=True, calls=len(rows), checks=dict(counts), seconds=time.perf_counter()-begin,
        development_summary_sha256=run.sha(source/'summary.json'), query_targets_accessed=False, core_research_goal_complete=False,
        outputs_sha256={f: run.sha(out/f) for f in ['groups.json', 'checks.json']})
    run.save(out/'summary.json', result); print(dict(stage='readout_audit_complete', **result), flush=True)


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[2]; out = root/run.BASE/'audit_v1'; out.mkdir(parents=True, exist_ok=False)
    try: audit(root, out)
    except Exception:
        run.save(out/'failure.json', dict(traceback=traceback.format_exc(), automatic_retry=False)); raise
