"""288: analytic checks and all54 OLD preflight readouts; no query targets."""
import argparse
from datetime import datetime, timezone
from fractions import Fraction as F
from pathlib import Path
import time

import numpy as np
from diagnose_gradient_flat_split_states_v1 import read, save, sha


def project(prediction):
    a = np.asarray(prediction, dtype=np.float64)
    if not np.isfinite(a).all():
        raise ValueError('Non-finite original prediction; preserve and diagnose it')
    return np.clip(a, 0., 1.)


def scalar_reference(a):
    return np.array([0. if x < 0 else 1. if x > 1 else float(x)
                     for x in np.asarray(a).flat]).reshape(np.shape(a))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    root = parser.parse_args().project.resolve()
    out = root/'results/known_range_projection/preflight_v1'
    out.mkdir(parents=True, exist_ok=False)
    begin = time.perf_counter()
    base = root/'results/probe_credit_confirmation'
    audit = read(base/'evaluation_audit_v6/summary.json')
    assert audit['passed'] and audit['tasks'] == 8192 and audit['bootstrap_replicates'] == 100000
    assert audit['evaluation_summary_sha256'] == sha(base/'evaluation_v6/summary.json')
    for name, digest in audit['outputs_sha256'].items():
        assert sha(base/'evaluation_audit_v6'/name) == digest
    inp = base/'runner_preflight_v2'
    parent = read(inp/'summary.json')
    before = read(inp/'before_query_manifest.json')
    assert parent['passed'] and parent['tasks'] == 2 and parent['predictors'] == 54
    assert before['protocol_sha256'] == sha(inp/'protocol.json')
    assert [r['seed'] for r in before['task_commits']] == [5910000, 5910063]
    save(out/'protocol.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
         scope='OLD two-task functional/analytic projection preflight, not new task quality',
         seeds=[5910000, 5910063], methods=read(inp/'protocol.json')['methods'],
         original_statistics_complete_sha256=sha(base/'evaluation_audit_v6/summary.json'),
         source_sha256=sha(Path(__file__)),
         plan_sha256=sha(root/'outputs/ttt-pc-alm-research/288_bounded_output_baseline_audit_plan.md'),
         before_query_manifest_sha256=sha(inp/'before_query_manifest.json'),
         query_targets_accessed=False, original_predictions_modified=False, adaptation_rerun=False,
         resource_scope='New O(Q) projection and output buffers; no claim original timings include them'))
    examples = np.array([-1e200, -3., -1e-12, -0., 0., .1, .5, 1., 1.+1e-12, 3., 1e200])
    got = project(examples)
    np.testing.assert_array_equal(got, scalar_reference(examples))
    np.testing.assert_array_equal(project(got), got)
    assert not np.shares_memory(examples, got)
    exact_checks = 0
    for prediction, clipped in zip(examples, got):
        for truth in [0., .1, .5, .9, 1.]:
            p, c, y = map(lambda z: F(float(z)), [prediction, clipped, truth])
            assert (p-y)**2-(c-y)**2 >= (p-c)**2 >= 0
            exact_checks += 1
    for value in [np.nan, np.inf, -np.inf]:
        try:
            project(np.array([value]))
        except ValueError:
            pass
        else:
            raise AssertionError('Nonfinite input not rejected')
    records = []
    for task in before['task_commits']:
        path = inp/task['file']
        assert sha(path) == task['sha256']
        commit = read(path)
        assert commit['protocol_sha256'] == before['protocol_sha256']
        assert sha(inp/commit['rows_file']) == commit['files'][commit['rows_file']]
        for row in read(inp/commit['rows_file']):
            path = inp/row['file']
            assert sha(path) == row['sha256'] == commit['files'][row['file']]
            transformed, changes = {}, {}
            with np.load(path, allow_pickle=False) as z:
                for field in ['prediction', 'point_prediction']:
                    original = z[field]
                    answer = project(original)
                    np.testing.assert_array_equal(answer, scalar_reference(original))
                    assert np.all((answer >= 0) & (answer <= 1))
                    assert project(answer).tobytes() == answer.tobytes()
                    assert original.shape == answer.shape == (257,)
                    transformed[field] = answer
                    changes[field] = dict(changed_values=int(np.count_nonzero(answer != original)),
                         bitwise_unchanged=answer.tobytes() == original.tobytes(),
                         original_min=float(original.min()), original_max=float(original.max()),
                         maximum_absolute_change=float(np.max(abs(answer-original))),
                         output_numeric_buffer_bytes=answer.nbytes)
            name = f'{task["seed"]}_{row["method"]}.npz'
            with (out/name).open('xb') as stream:
                np.savez_compressed(stream, **transformed)
            records.append(dict(seed=task['seed'], method=row['method'], original_file=row['file'],
                                original_sha256=row['sha256'], file=name, sha256=sha(out/name), changes=changes))
    assert len(records) == 54
    save(out/'rows.json', records)
    primary = [r for r in records if r['method'] == 'credit_control_probe33']
    assert len(primary) == 2 and all(c['bitwise_unchanged'] for r in primary for c in r['changes'].values())
    summary = dict(passed=True, tasks=2, readouts=54, prediction_arrays=108, exact_inequality_checks=exact_checks,
                   changed_readouts=sum(any(c['changed_values'] for c in r['changes'].values()) for r in records),
                   primary_old_predictions_bitwise_unchanged=True, all_scalar_references_equal=True,
                   query_targets_accessed=False, adaptation_rerun=False,
                   original_confirmation_statistics_complete=True, confirmation_projection_not_run=True,
                   scope='Functional old-task preflight only; no new 8192 projected comparison yet',
                   seconds=time.perf_counter()-begin, protocol_sha256=sha(out/'protocol.json'),
                   rows_sha256=sha(out/'rows.json'),
                   outputs_sha256={r['file']: r['sha256'] for r in records})
    save(out/'summary.json', summary)
    print({k:v for k,v in summary.items() if k != 'outputs_sha256'}, flush=True)


if __name__ == '__main__':
    main()
