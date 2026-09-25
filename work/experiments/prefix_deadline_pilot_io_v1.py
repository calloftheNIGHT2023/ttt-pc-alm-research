"""397 frozen registry and bounded stage input helpers."""
from pathlib import Path
import numpy as np
import deadline_risk_io_v1 as old
import deadline_risk_registry_v1 as previous
import prefix_matched_controls_v1 as controls

BASE='results/prefix_deadline_pilot'
DESIGN='outputs/ttt-pc-alm-research/397_prefix_deadline_pilot_v1.md'
PRIMARY='regional_active1024_dfs_farthest_x'
PRIMARY_BUDGET=.5
SEEDS=list(range(5920000,5920004))
STAGES=[4,8,16,24]
BUDGETS=[.25,.5]
read,save,sha,load_arrays,observations=old.read,old.save,old.sha,old.load_arrays,old.observations


def configs(root):
    cfg=[c for c in previous.configs() if c['kind']!='meta']+controls.cohort_configs(root)
    cfg += [dict(name=name,kind='portfolio') for name in controls.PORTFOLIOS]
    assert len(cfg)==len({c['name'] for c in cfg})==42
    return cfg


def dependencies(root):
    folder=root/'results/prefix_matched_controls/preflight_v1'
    checked=read(folder/'summary.json');assert checked['passed'] and not (folder/'failure.json').exists()
    old.verify_hashes(root,checked['source_sha256'])
    for f,digest in checked['outputs_sha256'].items():assert sha(folder/f)==digest
    files=[*sorted((root/'work/experiments').glob('*.py')),root/DESIGN]
    train=controls.require_training(root)
    weights={p.relative_to(root).as_posix():sha(p) for p in [train/'protocol.json',train/'training_summary.json',*sorted(train.glob('*.pt'))]}
    return dict(source_sha256={p.relative_to(root).as_posix():sha(p) for p in files},pretrained_sha256=weights,
        interface_preflight_sha256=sha(folder/'summary.json'))


def jobs(configurations,seeds,stages,budgets):
    for ci in np.random.default_rng(397101).permutation(len(configurations)):
        ci=int(ci);cfg=configurations[ci]
        all_jobs=[(seed,n,budget) for seed in seeds for n in stages for budget in budgets]
        order=np.random.default_rng(np.random.SeedSequence([397103,ci])).permutation(len(all_jobs))
        yield ci,cfg,[all_jobs[int(j)] for j in order]
