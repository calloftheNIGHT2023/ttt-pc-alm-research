"""387 immutable inputs, all-source allowlist and prediction/event archives."""
import hashlib
import json
from pathlib import Path
import numpy as np

BASE = 'results/new_task_deadline_risk'
DESIGN = 'outputs/ttt-pc-alm-research/387_new_task_deadline_risk_protocol_v1.md'


def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p, value):
    Path(p).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def verify_hashes(root, hashes):
    for p, h in hashes.items(): assert sha(root/p) == h, p


def prerequisites(root):
    for p in ['contiguous_regional_repeat/audit_v1/summary.json', 'n24_task_baselines/preflight_v2/summary.json',
              'n24_optimizer_controls/preflight_v1/summary.json', 'deadline_prediction/preflight_v1/summary.json']:
        assert read(root/'results'/p)['passed'], p
    old = root/'results/contiguous_regional_repeat/development_v1/protocol.json'
    verify_hashes(root, read(old)['source_sha256'])
    # A conservative source allowlist covers project-local lazy imports as well.
    # Future unrelated new files are allowed; none of these frozen files may change.
    files = sorted((root/'work/experiments').glob('*.py'))
    files.append(root/DESIGN)
    sources = {p.relative_to(root).as_posix():sha(p) for p in files}
    folder = root/'results/scalar_matched_batched/meta_prior'
    weights = {p.relative_to(root).as_posix():sha(p) for p in [folder/'protocol.json', folder/'training_summary.json', *sorted(folder.glob('*.pt'))]}
    return dict(source_sha256=sources, pretrained_sha256=weights,
        parent384_audit_sha256=sha(root/'results/contiguous_regional_repeat/audit_v1/summary.json'))


def save_call(root, directory, prediction, metadata, events, archive):
    directory.mkdir(parents=True, exist_ok=False)
    arrays = dict(prediction=prediction); notes = []
    for i, event in enumerate(events):
        notes.append({k:value for k, value in event.items() if k != 'prediction'})
        if 'prediction' in event: arrays[f'event_{i}_prediction'] = event['prediction']
    np.savez_compressed(directory/'outputs.npz', **arrays)
    save(directory/'metadata.json', metadata); save(directory/'events.json', notes)
    if archive is not None:
        np.savez_compressed(directory/'state.npz', **archive['arrays'])
        save(directory/'state.json', archive['metadata'])
    return dict(directory=directory.relative_to(root).as_posix(),
        files={p.name:sha(p) for p in directory.iterdir() if p.is_file()},
        selected=metadata['selected'], received_final=metadata['received_final'],
        online_seconds=metadata['decision_seconds'], setup_seconds=metadata['setup_seconds'],
        cleanup_seconds=metadata['cleanup_seconds'], task_cleanup_seconds=metadata['task_cleanup_seconds'],
        archive_seconds=metadata['archive_seconds'], controller_overrun_seconds=metadata['controller_overrun_seconds'],
        worker_max_sampled_rss=metadata['worker_max_sampled_rss'], controller_max_sampled_rss=metadata['controller_max_sampled_rss'],
        worker_pid=metadata['worker_pid'], generation=metadata['generation'], error=metadata['error'])


def load_arrays(p):
    with np.load(p, allow_pickle=False) as archive: return {k:archive[k] for k in archive.files}


def observations(seed):
    # Generator belongs to the runner/evaluator, never to predictor arguments.
    from run_independent_hybrid_memory import observations as original
    return original(seed)


def explicit_seed_history(root, seeds):
    import re
    pattern = re.compile(r'\b(?:'+'|'.join(map(str, seeds))+r')\b')
    own = {'work/experiments/deadline_risk_registry_v1.py', DESIGN}
    files = list((root/'work/experiments').glob('*.py'))+list((root/'results').rglob('*protocol*.json'))+list((root/'outputs').rglob('*.md'))
    hits = []; checked = 0
    for p in files:
        name = p.relative_to(root).as_posix()
        if name in own or name.startswith(BASE+'/'): continue
        found = sorted(set(pattern.findall(p.read_text(encoding='utf-8')))); checked += 1
        if found: hits.append(dict(file=name, matches=found))
    assert not hits, hits
    return dict(passed=True, files_checked=checked, explicit_old_seed_hits=hits,
        scope='Exact explicit seed identifiers in old code/protocols/notes; not a formal exclusion of all implicit historical intervals')
