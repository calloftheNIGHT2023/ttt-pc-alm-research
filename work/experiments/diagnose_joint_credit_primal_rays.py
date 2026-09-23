"""Fixed saved-parent joint-direction diagnostic; not an online implementation."""
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
import joint_credit_primal_ray as model
from diagnose_credit_activity_rays import Shadow
from diagnose_split_activity_modes import risk
old = model.previous.old


def main():
    p = argparse.ArgumentParser(); p.add_argument('--project', type=Path, required=True); args = p.parse_args()
    source = args.project / 'results/credit_activity_rays/diagnostic_v2'; out = args.project / 'results/joint_credit_primal_rays/diagnostic'
    refroot = args.project / 'results/posterior_state_reuse'; inherited = json.loads((source / 'protocol.json').read_text()); hashes = inherited['source_sha256'].copy()
    for name, expected in hashes.items(): assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() == expected
    for name in [Path(__file__).name, Path(model.__file__).name]: hashes[name] = hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
    protocol = dict(seeds=inherited['seeds'], source_sha256=hashes, verification=model.verify(),
        primary='local accepted-credit joint-ray positive modes beyond all fixed same-parent controls; shared-bias reachability escape',
        source_parent_protocol_sha256=hashlib.sha256((source / 'protocol.json').read_bytes()).hexdigest(),
        scope='old saved-parent support diagnostic; offline snapshot loading is not free online state; reference is evaluator-only; no query or speed claim')
    out.mkdir(parents=True, exist_ok=True); assert not (out / 'protocol.json').exists(); (out / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    parent_audits = json.loads((source / 'audits.json').read_text()); audits = []; rows = []
    reach = {(r['seed'], r['method']): r for r in json.loads((source / 'fixed_bias_reachability.json').read_text())['rows']}
    refs = {r['seed']: r for r in json.loads((refroot / 'first_write_reference/coverage.json').read_text())}
    for seed in protocol['seeds']:
        rng = np.random.default_rng(seed); truth = rng.uniform(-.12, .12, 4); allx = rng.uniform(0, 1, 24)
        allv = model.base.forward(allx, truth) + np.random.default_rng(seed + 19000000).uniform(-model.base.EPS, model.base.EPS, 24); x, v = allx[:4], allv[:4]; completed = []
        for parent_audit in [r for r in parent_audits if r['seed'] == seed]:
            path = source / parent_audit['file']; assert hashlib.sha256(path.read_bytes()).hexdigest() == parent_audit['sha256']; data = json.loads(path.read_text()); name = data['method']
            shadow = None
            if name == 'alm16':
                shadow = Shadow(x, v); anchor = np.zeros(4); starts, _ = old.old.interface.select_pool(x, v, anchor, old.old.interface.make_pool(4, 'prior256'), 64)
                with old.old.core.pipeline.discovery_box(.12): old.trace.refine(starts, x, v, anchor, 'alm', 16, shadow)
            groups = {k: set() for k in ['selected', 'residual', 'random', 'bp_current', 'bp_history', 'bp_combined']}; random = np.random.default_rng(329221 + seed)
            start = time.perf_counter()
            for snap in data['snapshots']:
                b = np.array(snap['b']); h = np.array(snap['h']); directions = dict(selected=np.array(snap['a']), residual=np.array(snap['residual']), random=random.standard_normal(h.shape))
                if shadow is not None:
                    bb, current, history = shadow.events[snap['step'], snap['phase']]; i = snap['restart']; assert np.array_equal(bb[i], b)
                    directions.update(bp_current=current[i], bp_history=history[i], bp_combined=current[i] + history[i])
                for label, direction in directions.items(): groups[label].update(model.ray_patterns(x, b, h, direction))
            seconds = time.perf_counter() - start; endpoint = {bytes.fromhex(k) for k in data['groups']['endpoint']}
            file = out / f'joint_rays_{seed}_{name}.json'; file.write_text(json.dumps(dict(seed=seed, method=name, endpoint=sorted(k.hex() for k in endpoint), groups={key: sorted(k.hex() for k in value) for key, value in groups.items()}), indent=2), encoding='utf-8')
            audits.append(dict(seed=seed, method=name, saved_parent_file=parent_audit['file'], saved_parent_sha256=parent_audit['sha256'], file=file.name, sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
                shadow_gradient_checks=0 if shadow is None else shadow.obs.gradient_checks))
            completed.append((name, endpoint, groups, seconds))
        path = refroot / 'first_write_reference' / refs[seed]['reference_file']; assert hashlib.sha256(path.read_bytes()).hexdigest() == refs[seed]['reference_sha256']; full = json.loads(path.read_text())
        keys = sorted(c['pattern'] for c in full['positive_regions']); positives = {bytes.fromhex(k) for k in keys}
        ca = json.loads((refroot / f'conditional_risk/audit_{seed}.json').read_text()); path = refroot / 'conditional_risk' / ca['curve_file']; assert hashlib.sha256(path.read_bytes()).hexdigest() == ca['curve_sha256']
        with np.load(path) as curve:
            assert np.array_equal(curve['patterns'], np.array(keys)); weights, means, grid = curve['weights'], curve['region_means'], curve['q']
            for name, endpoint, groups, seconds in completed:
                bp = set().union(*(groups[k] for k in ['bp_current', 'bp_history', 'bp_combined'])); variants = dict(endpoint=endpoint, selected=endpoint | groups['selected'], residual=endpoint | groups['residual'], random=endpoint | groups['random'])
                if name == 'alm16': variants['bp_union'] = endpoint | bp
                scores = {}
                for label, members in variants.items():
                    mask = np.array([bytes.fromhex(k) in members for k in keys]); regs = np.array([np.frombuffer(k, np.uint8).reshape(4, 4) for k in sorted(members)]); rejected = old.screen.contract(x, v, regs, 5)
                    assert not {r.tobytes() for r in regs[rejected]} & positives
                    scores[label] = dict(**risk(mask, weights, means, grid), proposals=len(members), after_c5=int((~rejected).sum()))
                added = (groups['selected'] - endpoint) & positives; extra = added - groups['residual'] - groups['random'] - bp
                fixed_blocked = {bytes.fromhex(r['pattern']) for r in reach[seed, name]['details'] if r['necessary_box_compatible_parents'] == 0}
                rows.append(dict(seed=seed, method=name, scores=scores, selected_new_positive=len(added), independent_extra_positive=len(extra),
                    escaped_fixed_bias_blocked_positive=len(added & fixed_blocked), added_keys=sorted(k.hex() for k in added), independent_extra_keys=sorted(k.hex() for k in extra),
                    diagnostic_all_channel_generation_seconds=seconds))
        (out / 'audits.json').write_text(json.dumps(audits, indent=2), encoding='utf-8'); (out / 'diagnostics.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
        print(json.dumps(dict(completed=seed - protocol['seeds'][0] + 1, total=16, paths=len(audits))), flush=True)
    summary = []
    for name in ['alm16', 'adam8', 'adam16']:
        rr = [r for r in rows if r['method'] == name]
        scores = {label: {field: float(np.mean([r['scores'][label][field] for r in rr])) for field in ['mass', 'truncation_risk', 'positive_modes', 'proposals', 'after_c5']} for label in rr[0]['scores']}
        summary.append(dict(method=name, scores=scores, **{key: sum(r[key] for r in rr) for key in ['selected_new_positive', 'independent_extra_positive', 'escaped_fixed_bias_blocked_positive']},
            mean_all_channel_generation_seconds=float(np.mean([r['diagnostic_all_channel_generation_seconds'] for r in rr]))))
    result = dict(summary=summary, saved_parent_audits=len(audits), source_hashes=len(hashes), scope=protocol['scope'])
    (out / 'summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8'); print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
