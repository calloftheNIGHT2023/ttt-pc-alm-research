"""Bounded 417--419 interface snapshot; no unrun deadline result claim."""
import argparse
import hashlib
import re
from pathlib import Path
import region_credit_snapshot_20260925 as pack

PARENT = '6507718a4ceafc8ec401c0f0706bb7b613fa1a26'
MANIFEST = 'release/continuous-frontier-20260925.json'
ENTRY = 'release/CONTINUOUS-FRONTIER-20260925.md'


def paths(root):
    tree = {}; submodules = {}; files = set()
    for record in pack.git(root, 'ls-tree', '-r', '-z', PARENT).decode().strip('\0').split('\0'):
        header, name = record.split('\t', 1); mode, kind, oid = header.split(); tree[name] = oid
        if mode == '160000':
            submodules[name] = oid
    def immutable(name, digest):
        assert pack.sha(root/name) == digest, name
        external = next((s for s in submodules if name.startswith(s+'/')), None)
        if external:
            assert pack.git(root/external, 'rev-parse', 'HEAD').decode().strip() == submodules[external]
        elif name in tree:
            payload = (root/name).read_bytes()
            assert tree[name] == hashlib.sha1(b'blob '+str(len(payload)).encode()+b'\0'+payload).hexdigest(), name
        else:
            files.add(name)
    folders = ['results/continuous_frontier/preflight_v1', 'results/continuous_frontier/receipt_preflight_v1',
               'results/frontier_deadline/binding_preflight_v2']
    for relative in folders:
        folder = root/relative; summary = pack.read(folder/'summary.json')
        assert summary['passed'] and not (folder/'failure.json').exists()
        for name, digest in summary['outputs_sha256'].items():
            assert pack.sha(folder/name) == digest, name
        for name, digest in summary['source_sha256'].items():
            immutable(name, digest)
        files.update(p.relative_to(root).as_posix() for p in folder.rglob('*') if p.is_file())
    failure = root/'results/frontier_deadline/binding_preflight_v1/failure.json'
    assert 'analyze_dual_jump_query.py' in pack.read(failure)['traceback']
    files.add(failure.relative_to(root).as_posix())
    files.update(['scripts/continuous_frontier_snapshot_20260925.py', ENTRY,
                  'outputs/ttt-pc-alm-research/420_continuous_frontier_preflight_results_v1.md'])
    for relative in [ENTRY, 'outputs/ttt-pc-alm-research/420_continuous_frontier_preflight_results_v1.md']:
        doc = root/relative
        for link in re.findall(r'\]\(([^)]+)\)', doc.read_text(encoding='utf-8')):
            if not link.startswith(('http:', 'https:', '#')):
                assert (doc.parent/link.split('#')[0]).resolve().is_file(), (relative, link)
    return sorted(files)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['build', 'stage', 'verify'])
    args = parser.parse_args(); pack.MANIFEST = MANIFEST; pack.paths = paths
    print(dict(action=args.action, **getattr(pack, args.action)(Path(__file__).resolve().parents[1])), flush=True)
