"""Frozen event-channel diagnostic; reference is evaluator-only after proposals."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import activity_path_event_proposals as model
from diagnose_missing_mode_graph import distances
from diagnose_split_activity_modes import risk


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--project', type=Path, required=True); a = parser.parse_args()
    root = a.project / 'results/activity_path_events/diagnostic'; refroot = a.project / 'results/posterior_state_reuse'
    inherited = json.loads((a.project / 'results/batched_credit_bank/development/protocol.json').read_text()); hashes = inherited['source_sha256'].copy()
    for name, h in hashes.items(): assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() == h, name
    for name in [Path(__file__).name, Path(model.__file__).name, 'diagnose_missing_mode_graph.py', 'diagnose_split_activity_modes.py']:
        hashes[name] = hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
    configs = [dict(name='alm16', generator='alm', sweeps=16), dict(name='nodual16', generator='nodual', sweeps=16), dict(name='pc80', generator='pc', sweeps=80),
        dict(name='adam8', generator='adam', steps=8), dict(name='adam16', generator='adam', steps=16), dict(name='adam60', generator='adam', steps=60)]
    protocol = dict(seeds=list(range(5900000, 5900016)), configs=configs, source_sha256=hashes, verification=model.verify(),
        primary_diagnostic='extra valid activity-path modes beyond original endpoint plus full parameter-segment modes; all six frozen paths receive both channels',
        gate='original optimizer best and endpoint mode universe bitwise preserved', scope='old-task support diagnostic; reference never enters proposals; no online timing or new-query quality claim')
    root.mkdir(parents=True, exist_ok=True); assert not (root / 'protocol.json').exists()
    (root / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8'); rows = []; audits = []
    references = {r['seed']: r for r in json.loads((refroot / 'first_write_reference/coverage.json').read_text())}
    for seed in protocol['seeds']:
        rng = np.random.default_rng(seed); truth = rng.uniform(-.12, .12, 4); allx = rng.uniform(0, 1, 24)
        allv = model.base.forward(allx, truth) + np.random.default_rng(seed + 19000000).uniform(-model.base.EPS, model.base.EPS, 24); x, v = allx[:4], allv[:4]
        completed = []
        for cfg in configs:
            obs, audit = model.discover(x, v, cfg)
            groups = dict(endpoint=set(obs.forward) | set(obs.split), parameter=set(obs.parameter_path), activity=set(obs.activity_path))
            assert all(len(key) == 16 for keys in groups.values() for key in keys)
            path = root / f'events_{seed}_{cfg["name"]}.json'
            data = dict(seed=seed, method=cfg['name'], **{key: sorted(k.hex() for k in value) for key, value in groups.items()},
                parameter_witnesses={k.hex(): value for k, value in obs.parameter_path.items()}, activity_witnesses={k.hex(): value for k, value in obs.activity_path.items()})
            path.write_text(json.dumps(data, indent=2), encoding='utf-8')
            audits.append(dict(seed=seed, method=cfg['name'], **audit, event_file=path.name, event_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            completed.append((cfg, obs, groups))
        # Full reference first enters here, after all candidate proposals for this support.
        path = refroot / 'first_write_reference' / references[seed]['reference_file']; assert hashlib.sha256(path.read_bytes()).hexdigest() == references[seed]['reference_sha256']
        reference = json.loads(path.read_text()); assert reference['numerical_volume_reference_complete'] and np.array_equal(reference['x'], x) and np.array_equal(reference['v'], v)
        keys = sorted(c['pattern'] for c in reference['positive_regions']); patterns = np.array([np.frombuffer(bytes.fromhex(k), np.uint8) for k in keys])
        adjacency = np.abs(patterns[:, None].astype(int) - patterns[None, :].astype(int)).sum(-1) == 1
        curve_audit = json.loads((refroot / f'conditional_risk/audit_{seed}.json').read_text()); path = refroot / 'conditional_risk' / curve_audit['curve_file']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == curve_audit['curve_sha256']
        with np.load(path) as curve:
            assert np.array_equal(curve['patterns'], np.array(keys)); weights, means, grid = curve['weights'], curve['region_means'], curve['q']
            for cfg, obs, groups in completed:
                variants = dict(endpoint=groups['endpoint'], parameter=groups['endpoint'] | groups['parameter'], activity=groups['endpoint'] | groups['activity'], both=set().union(*groups.values()))
                scores = {}; masks = {}
                for label, members in variants.items():
                    mask = np.array([bytes.fromhex(k) in members for k in keys]); masks[label] = mask
                    regs = np.array([np.frombuffer(k, np.uint8).reshape(4, 4) for k in sorted(members)])
                    screened = model.old.old.screen.contract(x, v, regs, 5)
                    assert not any(bytes.fromhex(k) in {r.tobytes() for r in regs[screened]} for k in np.array(keys)[mask])
                    scores[label] = dict(**risk(mask, weights, means, grid), proposals=len(members), after_c5=int((~screened).sum()))
                steps = distances(adjacency, masks['endpoint']); extra = masks['both'] & ~masks['parameter']
                rows.append(dict(seed=seed, method=cfg['name'], scores=scores, activity_extra_beyond_parameter=int(extra.sum()), activity_extra_mass=float(weights[extra].sum()),
                    activity_extra_beyond_endpoint_h1=int((extra & ((steps < 0) | (steps > 1))).sum()), activity_extra_beyond_endpoint_h2=int((extra & ((steps < 0) | (steps > 2))).sum()),
                    activity_extra_new_component_mass=float(weights[extra & (steps < 0)].sum()), extra_activity_keys=np.array(keys)[extra].tolist(),
                    parameter_segment_seconds=obs.parameter_seconds, activity_segment_seconds=obs.activity_seconds, parameter_segments=obs.parameter_segments, activity_segments=obs.activity_segments,
                    endpoint_key_bytes=sum(map(len, groups['endpoint'])), additional_path_key_bytes=sum(map(len, variants['both'] - groups['endpoint'])), snapshot_array_bytes=obs.max_snapshot_bytes))
        (root / 'audits.json').write_text(json.dumps(audits, indent=2), encoding='utf-8'); (root / 'diagnostics.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
        print(json.dumps(dict(completed=seed - protocol['seeds'][0] + 1, total=len(protocol['seeds']), trajectories=len(audits))), flush=True)
    summary = []
    for cfg in configs:
        rr = [r for r in rows if r['method'] == cfg['name']]
        scores = {label: {field: float(np.mean([r['scores'][label][field] for r in rr])) for field in ['mass', 'truncation_risk', 'positive_modes', 'proposals', 'after_c5']} for label in ['endpoint', 'parameter', 'activity', 'both']}
        summary.append(dict(method=cfg['name'], scores=scores, tasks_with_activity_extra=sum(r['activity_extra_beyond_parameter'] > 0 for r in rr),
            **{field: sum(r[field] for r in rr) for field in ['activity_extra_beyond_parameter', 'activity_extra_beyond_endpoint_h1', 'activity_extra_beyond_endpoint_h2']},
            mean_activity_extra_mass=float(np.mean([r['activity_extra_mass'] for r in rr])), mean_parameter_segment_seconds=float(np.mean([r['parameter_segment_seconds'] for r in rr])),
            mean_activity_segment_seconds=float(np.mean([r['activity_segment_seconds'] for r in rr])), max_additional_path_key_bytes=max(r['additional_path_key_bytes'] for r in rr), max_snapshot_array_bytes=max(r['snapshot_array_bytes'] for r in rr)))
    result = dict(summary=summary, original_paths_and_endpoint_sets_exact=len(audits), source_hashes=len(hashes), scope=protocol['scope'])
    (root / 'summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8'); print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
