"""Necessary reachability bounds for any activity-only proposal at fixed b."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser(); p.add_argument('--project', type=Path, required=True); args = p.parse_args()
    root = args.project / 'results/credit_activity_rays/diagnostic_v2'; protocol = json.loads((root / 'protocol.json').read_text()); audits = json.loads((root / 'audits.json').read_text())
    for name, h in protocol['source_sha256'].items(): assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() == h
    refs = {r['seed']: r for r in json.loads((args.project / 'results/posterior_state_reuse/first_write_reference/coverage.json').read_text())}; rows = []
    low = np.array([-np.inf, 0, .5, 1]); high = np.array([0, .5, 1, np.inf])
    for audit in audits:
        path = root / audit['file']; assert hashlib.sha256(path.read_bytes()).hexdigest() == audit['sha256']; data = json.loads(path.read_text())
        ref = refs[audit['seed']]; path = args.project / 'results/posterior_state_reuse/first_write_reference' / ref['reference_file']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == ref['reference_sha256']; complete = json.loads(path.read_text())
        endpoint = set(data['groups']['endpoint']); missing = [r for r in complete['positive_regions'] if r['pattern'] not in endpoint]
        total = sum(r['volume'] for r in complete['positive_regions']); detail = []
        for cell in missing:
            target = np.frombuffer(bytes.fromhex(cell['pattern']), np.uint8).reshape(4, 4); first_parents = 0; box_parents = 0
            for snap in data['snapshots']:
                source = np.frombuffer(bytes.fromhex(snap['pattern']), np.uint8).reshape(4, 4)
                if not np.array_equal(source[0], target[0]): continue
                first_parents += 1; b = np.array(snap['b'])
                # Necessary only, deliberately allows exact branch endpoints.
                possible = np.maximum(low[target[1:]], b[1:, None]) <= np.minimum(high[target[1:]], 1 + b[1:, None])
                if np.all(possible): box_parents += 1
            detail.append(dict(pattern=cell['pattern'], mass=cell['volume'] / total, first_layer_matching_parents=first_parents, necessary_box_compatible_parents=box_parents))
        rows.append(dict(seed=audit['seed'], method=audit['method'], missing_modes=len(missing), parents=len(data['snapshots']),
            blocked_by_fixed_first_layer=sum(d['first_layer_matching_parents'] == 0 for d in detail),
            blocked_by_fixed_bias_boxes=sum(d['necessary_box_compatible_parents'] == 0 for d in detail), details=detail))
    summary = []
    for name in ['alm16', 'adam8', 'adam16']:
        rr = [r for r in rows if r['method'] == name]
        summary.append(dict(method=name, **{f: sum(r[f] for r in rr) for f in ['missing_modes', 'blocked_by_fixed_first_layer', 'blocked_by_fixed_bias_boxes']},
            mean_missing_mass=float(np.mean([sum(d['mass'] for d in r['details']) for r in rr])),
            mean_box_blocked_mass=float(np.mean([sum(d['mass'] for d in r['details'] if d['necessary_box_compatible_parents'] == 0) for r in rr]))))
    result = dict(summary=summary, rows=rows, source_sha256=protocol['source_sha256'], audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='necessary structural bounds at frozen chosen parents; does not establish reachable sufficiency or new-method success')
    (root / 'fixed_bias_reachability.json').write_text(json.dumps(result, indent=2), encoding='utf-8'); print(json.dumps(summary, indent=2))


if __name__ == '__main__': main()
