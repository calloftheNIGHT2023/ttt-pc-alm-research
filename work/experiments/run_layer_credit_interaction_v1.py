"""309 support-only layer interventions, seal before geometry evaluation."""
import argparse
from collections import Counter
from fractions import Fraction as F
import gzip
import json
from pathlib import Path
import time
import traceback

import numpy as np
import layer_credit_interaction_v1 as method
from evaluate_complete_credit_mode_geometry_v1 import load_support, read, save, sha, FAMILIES


def run(root, out):
    started = time.perf_counter()
    test = root/'results/layer_credit_interaction/tests_attempt02/summary.json'
    testing = read(test)
    assert testing['passed'] and not testing['real_task_data_accessed']
    for name, digest in testing['source_sha256'].items():
        assert sha(root/'work/experiments'/name) == digest
    folder, old_summary, old_protocol, grouped = load_support(root)
    source_names = set(old_protocol['source_sha256']) | set(testing['source_sha256']) | {
        Path(__file__).name, 'evaluate_complete_credit_mode_geometry_v1.py'}
    save(out/'protocol.json', dict(
        source_sha256={n: sha(root/'work/experiments'/n) for n in sorted(source_names)},
        tests_sha256=sha(test), support_summary_sha256=sha(folder/'summary.json'),
        design_sha256=sha(root/'outputs/ttt-pc-alm-research/309_layer_credit_interaction_protocol.md'),
        task_manifest=old_protocol['task_manifest'], families=FAMILIES,
        masks=method.masks(4), fixed_float_threshold=1e-10,
        query_targets_accessed=False, geometry_accessed=False, resources_matched=False,
        new_confirmation=False, global_bp_in_candidate=False))
    counts = Counter(); rows = []; files = {}
    for number, (key, group) in enumerate(sorted(grouped.items()), 1):
        zero_reference = None
        for family in FAMILIES:
            old = group[family]
            with gzip.open(folder/old['file'], 'rt', encoding='utf-8') as f:
                payload = json.load(f)
            state = method.decode(payload['state'])
            endpoints = {F(p['alpha']): p for p in payload['seven_points'] if F(p['alpha']) in (0, 1)}
            del payload
            result, costs = method.combine(state)
            for alpha, idx in [(0, 0), (1, 15)]:
                rr, previous = result[idx], endpoints[F(alpha)]
                for name in ['b', 'h']:
                    def eval_old(v):
                        if isinstance(v[0], list): return [eval_old(q) for q in v]
                        return sum(F(c)*F(alpha)**j for j,c in enumerate(v))
                    assert [eval_old(v) for v in previous['exact_'+name]] == rr['exact'][name]
                    assert np.array_equal(rr['float_'+name], previous['float_'+name]), (key, family, alpha, name)
                    counts['old_endpoint_array_checks'] += 2
                assert previous['exact_mode'] == rr['exact_mode'] and previous['float_mode'] == rr['float_mode']
            zero = result[0]['exact']
            if zero_reference is None: zero_reference = zero
            else: assert zero_reference == zero
            filename = '_'.join(map(str, key))+'_'+family+'.json'
            save(out/filename, dict(seed=key[0], location=key[1:], family=family,
                 source_file=old['file'], source_sha256=old['sha256'], state=state,
                 proposals=result, costs=costs))
            files[filename] = sha(out/filename)
            gaps = [max(r['gaps'].values()) for r in result]
            rows.append(dict(seed=key[0], location=key[1:], family=family, file=filename,
                 sha256=files[filename], proposals=len(result), max_float_gap=max(gaps),
                 float_value_discrepancies=sum(r['float_value_discrepancy'] for r in result),
                 float_mode_discrepancies=sum(r['float_mode_discrepancy'] for r in result), costs=costs))
            counts['proposals'] += len(result)
            counts['exact_full_step_crosschecks'] += len(result)
            counts['float_value_discrepancies'] += rows[-1]['float_value_discrepancies']
            counts['float_mode_discrepancies'] += rows[-1]['float_mode_discrepancies']
        if number % 8 == 0:
            print(dict(states=number, directions=len(rows), counts=dict(counts), seconds=time.perf_counter()-started), flush=True)
    assert len(rows) == 393 and counts['proposals'] == 6288
    save(out/'rows.json', rows)
    files.update({n:sha(out/n) for n in ['protocol.json','rows.json']})
    save(out/'before_geometry_manifest.json', dict(files_sha256=files, proposals=6288,
         all_states_processed=True, query_targets_accessed=False, geometry_accessed=False))
    summary = dict(passed=True, tasks=64, selected_states=131, directions=393,
         counts=dict(counts), max_float_gap=max(r['max_float_gap'] for r in rows),
         seconds=time.perf_counter()-started, outputs_sha256=files,
         manifest_sha256=sha(out/'before_geometry_manifest.json'),
         query_targets_accessed=False, geometry_accessed=False, resources_matched=False)
    save(out/'summary.json', summary)
    print({k:v for k,v in summary.items() if k!='outputs_sha256'}, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--out', type=Path, required=True)
    args=p.parse_args(); args.out.mkdir(parents=True, exist_ok=False)
    try: run(Path(__file__).resolve().parents[2], args.out)
    except Exception:
        save(args.out/'failure.json', dict(traceback=traceback.format_exc()))
        raise
