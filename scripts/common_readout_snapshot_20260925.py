"""382--383 bounded immutable candidate validation/readout evidence release."""
import argparse
import hashlib
import re
from pathlib import Path
import region_credit_snapshot_20260925 as pack

PARENT = '60553bf650acbf415d5da63a5ba5ae227cdd0d34'
BASE = 'results/candidate_set_readout'
MANIFEST = 'release/common-readout-20260925.json'


def paths(root):
    tree = {}
    for item in pack.git(root, 'ls-tree', '-r', '-z', PARENT).decode().strip('\0').split('\0'):
        header, name = item.split('\t', 1); tree[name] = header.split()[2]
    files = set(); protocol = pack.read(root/BASE/'development_v1/protocol.json')
    for name, h in protocol['source_sha256'].items():
        payload = (root/name).read_bytes(); assert pack.sha(root/name) == h
        if name in tree: assert tree[name] == hashlib.sha1(b'blob '+str(len(payload)).encode()+b'\0'+payload).hexdigest()
        else: files.add(name)
    for name in ['prototype_preflight_v1', 'development_v1', 'audit_v1']:
        folder = root/BASE/name; summary = pack.read(folder/'summary.json')
        assert summary['passed'] and not (folder/'failure.json').exists()
        for f, h in summary.get('outputs_sha256', {}).items(): assert pack.sha(folder/f) == h
        files.update(str(f.relative_to(root)).replace('\\', '/') for f in folder.rglob('*') if f.is_file())
    for row in pack.read(root/BASE/'development_v1/rows.json'):
        for f, h in row['files'].items(): assert pack.sha(root/row['directory']/f) == h
    report = root/BASE/'report_v1'; manifest = pack.read(report/'manifest.json')
    qa = pack.read(report/'qa_numeric.json'); visual = pack.read(report/'visual_qa.json')
    assert qa['passed'] and qa['manifest_sha256'] == pack.sha(report/'manifest.json')
    assert visual['passed'] and visual['actually_viewed'] and visual['numeric_qa_sha256'] == pack.sha(report/'qa_numeric.json')
    for im in visual['images']: assert im['sha256'] == pack.sha(report/im['file'])
    for f, h in manifest['outputs_sha256'].items(): assert pack.sha(report/f) == h
    assert manifest['entry_sha256'] == pack.sha(root/manifest['entry_file'])
    assert manifest['source_sha256'] == pack.sha(root/'work/experiments/report_candidate_set_readout_v1.py')
    for doc in [report/'report.md', root/manifest['entry_file'], root/'release/COMMON-READOUT-20260925.md']:
        for link in re.findall(r'\]\(([^)]+)\)', doc.read_text(encoding='utf-8')):
            if not link.startswith(('https:', 'http:', '#')): assert (doc.parent/link.split('#')[0]).resolve().is_file(), (doc, link)
    files.update(str(f.relative_to(root)).replace('\\', '/') for f in report.iterdir() if f.is_file())
    files.update([manifest['entry_file'], 'release/COMMON-READOUT-20260925.md',
        'scripts/common_readout_snapshot_20260925.py', 'scripts/region_credit_snapshot_20260925.py',
        'work/experiments/report_candidate_set_readout_v1.py'])
    return sorted(files)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['build', 'stage', 'verify']); args = p.parse_args()
    pack.MANIFEST = MANIFEST; pack.paths = paths
    print(dict(action=args.action, **getattr(pack, args.action)(Path(__file__).resolve().parents[1])), flush=True)
