"""300 evaluation-only recovery: use sealed v1 proposals; actual moment key is q."""
import argparse
from pathlib import Path
import math
import time
import numpy as np
from diagnose_counterfactual_conditional_risk_v1 import read, save, sha
from diagnose_stagnant_trigger_exploration_v1 import aggregate


def run(root, out):
    tick = time.perf_counter()
    support = root/'results/stagnant_trigger_exploration/development_v1'
    seal = read(support/'before_reference_manifest.json')
    protocol = read(support/'protocol.json')
    for name,digest in seal['files_sha256'].items():
        assert sha(support/name) == digest,name
    for name,digest in protocol['source_sha256'].items():
        assert sha(root/'work/experiments'/name) == digest,name
    selftests = read(support/'selftests.json')
    assert selftests['passed'] and seal['tasks'] == 64 and seal['new_configs'] == 69
    failure = read(support/'failure.json')
    assert failure['error_type'] == 'KeyError' and 'q257' in failure['message']
    save(out/'protocol.json',dict(source_sha256=sha(Path(__file__)),
        sealed_support_manifest_sha256=sha(support/'before_reference_manifest.json'),
        preserved_failure_sha256=sha(support/'failure.json'),
        support_directory=str(support.relative_to(root)).replace('\\','/'),
        scoring_formula_unchanged=True,fix='moment archive key q, not q257',
        new_support_adaptation=False,new_configuration_selection=False,
        grids=[257,129],batch_pairs=[[0,1],[2,3]],query_targets_accessed=False))
    cached = root/'results/counterfactual_conditional_risk/development_v2'
    assert sha(cached/'summary.json') == '95852bf236596935f7ac0dd3c6fe389118b6976b5b2f368e09cc8ac37325a3ec'
    refsummary = read(cached/'summary.json')
    assert refsummary['passed']
    for name,digest in refsummary['outputs_sha256'].items():
        assert sha(cached/name) == digest,name
    for name,digest in read(cached/'input_hashes.json').items():
        assert sha(root/name) == digest,name
    previous = root/'results/dual_amplitude_paths/development_v1'
    assert sha(previous/'summary.json') == '1a90bc14baebe225c2501f8886e1e5e59077031fa894a0080a5f3a27d95a972c'
    prevsummary = read(previous/'summary.json')
    for name,digest in prevsummary['outputs_sha256'].items():
        assert sha(previous/name) == digest,name
    background = [dict(r,method='background__'+r['method'],scope='298_background_not_trigger_matched') for r in read(previous/'scores.json')]
    assert len(background) == 5888
    moments = root/'results/confirmation_conditional_risk/moments'
    task_moments = {r['seed']:r for r in read(moments/'tasks.json')}
    scores, max_scalar_gap = [], 0.
    for task_index,row in enumerate(read(support/'proposals.json')):
        task = task_moments[row['seed']]
        assert sha(moments/task['file']) == task['sha256']
        with np.load(moments/task['file'],allow_pickle=False) as z:
            vol,means = z['volumes'],z['means']
            assert np.array_equal(z['q'],np.linspace(0,1,257))
        assert np.all(vol > 0) and means.shape == (len(vol),4,257)
        keys = task['keys']; keyset = set(keys)
        weights = vol/vol.sum()
        full = np.einsum('k,kbq->bq',weights,means)
        old = set(row['original_positive_modes'])
        assert old and old <= keyset
        def mixture(pool):
            mask = np.array([k in pool for k in keys])
            mass = float(weights[mask].sum())
            assert mass > 0
            return mass,np.einsum('k,kbq->bq',weights*mask/mass,means)
        oldmass,oldmu = mixture(old)
        for name,method in row['methods'].items():
            extra = (set(method['modes']) & keyset)-old
            mass,mu = mixture(old|extra)
            for grid in [257,129]:
                idx = np.arange(257) if grid == 257 else np.arange(0,257,2)
                estimates = []
                for a,b in [(0,1),(2,3)]:
                    bias = float(np.mean((mu[a,idx]-full[a,idx])*(mu[b,idx]-full[b,idx])))
                    base = float(np.mean((oldmu[a,idx]-full[a,idx])*(oldmu[b,idx]-full[b,idx])))
                    scalar = math.fsum(float(c)*float(d) for c,d in zip(mu[a,idx]-full[a,idx],mu[b,idx]-full[b,idx]))/grid
                    max_scalar_gap = max(max_scalar_gap,abs(bias-scalar))
                    estimates.append(dict(pair=[a,b],policy_excess=bias,delta=bias-base))
                scores.append(dict(seed=row['seed'],method=name,scope='300_same_stagnation_locations',grid=grid,
                    mass=mass,mass_added=mass-oldmass,new_positive_modes=sorted(extra),shadow_steps=method['shadow_steps'],
                    pairs=estimates,policy_excess=float(np.mean([e['policy_excess'] for e in estimates])),
                    delta=float(np.mean([e['delta'] for e in estimates]))))
                if row['selection_counts'][method['group']] == 0:
                    assert not extra and scores[-1]['delta'] == 0 and method['shadow_steps'] == 0
        if (task_index+1)%16 == 0:
            print(dict(phase='evaluation_only_recovery',tasks=task_index+1),flush=True)
    assert max_scalar_gap < 1e-12 and len(scores) == 8832
    scores += background
    assert len(scores) == 14720
    aggregates = aggregate(scores)
    assert len(aggregates) == 230
    lookup = {(r['method'],r['grid']):r for r in aggregates}
    max_replay = 0.
    for r in read(previous/'aggregates.json'):
        rr = lookup['background__'+r['method'],r['grid']]
        for metric in ['policy_excess','delta','mass','shadow_steps']:
            max_replay = max(max_replay,abs(rr[metric]-r[metric]))
    assert max_replay == 0
    save(out/'scores.json',scores)
    save(out/'aggregates.json',aggregates)
    for name,digest in seal['files_sha256'].items():
        assert sha(support/name) == digest,name
    for name,digest in protocol['source_sha256'].items():
        assert sha(root/'work/experiments'/name) == digest,name
    summary = dict(passed=True,tasks=64,new_configs=69,background_configs=46,grid_rows=len(scores),
        counts=selftests['counts'],empty_tasks=selftests['empty_tasks'],selftest_cases=selftests['cases'],
        max_norm_gap=selftests['max_norm_gap'],max_scalar_gap=max_scalar_gap,max_background_replay_gap=max_replay,
        evaluation_only_seconds=time.perf_counter()-tick,support_recomputed=False,
        query_targets_accessed=False,reference_only_after_all_proposals_sealed=True,numerical_reference_only=True,
        resources_matched=False,new_confirmation=False,core_research_goal_complete=False,
        support_manifest_sha256=sha(support/'before_reference_manifest.json'),
        cached_reference_summary_sha256=sha(cached/'summary.json'),background_summary_sha256=sha(previous/'summary.json'),
        outputs_sha256={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()})
    save(out/'summary.json',summary)
    print({k:v for k,v in summary.items() if k != 'outputs_sha256'},flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,required=True)
    root = parser.parse_args().project.resolve()
    out = root/'results/stagnant_trigger_exploration/evaluation_v2'
    out.mkdir(parents=True,exist_ok=False)
    try:
        run(root,out)
    except BaseException as exc:
        save(out/'failure.json',dict(error_type=type(exc).__name__,message=str(exc),no_automatic_retry=True))
        raise
