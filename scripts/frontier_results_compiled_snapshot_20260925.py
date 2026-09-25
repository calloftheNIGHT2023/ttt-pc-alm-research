"""Bounded complete 419 results and 422/425/427 preflight snapshot."""
import argparse
import hashlib
import re
from pathlib import Path
import region_credit_snapshot_20260925 as pack

PARENT = '18e465c743dc46668538283569ee4f8b5e8983d2'
MANIFEST = 'release/frontier-results-compiled-20260925.json'
ENTRY = 'release/FRONTIER-RESULTS-COMPILED-20260925.md'


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
    folders = ['results/frontier_deadline/development_v1', 'results/frontier_deadline/audit_v1',
        'results/frontier_deadline/evaluation_v1', 'results/piecewise_frontier/preflight_v1',
        'results/piecewise_frontier/component_v1', 'results/compiled_continuous_frontier/preflight_v1',
        'results/compiled_frontier_deadline/binding_preflight_v1']
    for relative in folders:
        folder = root/relative; summary = pack.read(folder/'summary.json')
        assert summary['passed'] and not (folder/'failure.json').exists()
        for name, digest in summary['outputs_sha256'].items():
            assert pack.sha(folder/name) == digest, (relative, name)
        for document in [summary, *([pack.read(folder/'protocol.json')] if (folder/'protocol.json').exists() else [])]:
            for key in ['source_sha256', 'pretrained_sha256']:
                for name, digest in document.get(key, {}).items():
                    immutable(name, digest)
        if relative == 'results/frontier_deadline/development_v1':
            for row in pack.read(folder/'rows.json'):
                for name, digest in row['files'].items():
                    assert pack.sha(root/row['directory']/name) == digest
        files.update(p.relative_to(root).as_posix() for p in folder.rglob('*') if p.is_file())
    report = root/'results/frontier_deadline/report_v1'
    qa = pack.read(report/'qa_numeric.json'); visual = pack.read(report/'visual_qa.json')
    manifest = pack.read(report/'manifest.json')
    assert qa['passed'] and qa['manifest_sha256'] == pack.sha(report/'manifest.json')
    assert visual['passed'] and visual['actually_viewed'] and visual['numeric_qa_sha256'] == pack.sha(report/'qa_numeric.json')
    for row in visual['images']:
        assert pack.sha(report/row['file']) == row['sha256']
    for name, digest in manifest['outputs_sha256'].items():
        assert pack.sha(report/name) == digest
    assert manifest['evaluation_summary_sha256'] == pack.sha(root/'results/frontier_deadline/evaluation_v1/summary.json')
    immutable(manifest['entry_file'], manifest['entry_sha256'])
    files.update(p.relative_to(root).as_posix() for p in report.iterdir() if p.is_file())
    documents = [ENTRY, 'outputs/ttt-pc-alm-research/423_frontier_deadline_results_v1.md',
        'outputs/ttt-pc-alm-research/424_piecewise_frontier_component_results_v1.md',
        'outputs/ttt-pc-alm-research/426_compiled_continuous_preflight_results_v1.md']
    files.update(documents+['scripts/frontier_results_compiled_snapshot_20260925.py'])
    for relative in documents:
        doc = root/relative
        for link in re.findall(r'\]\(([^)]+)\)', doc.read_text(encoding='utf-8')):
            if not link.startswith(('http:', 'https:', '#')):
                assert (doc.parent/link.split('#')[0]).resolve().is_file(), (relative, link)
    return sorted(files)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['build', 'stage', 'verify'])
    args = parser.parse_args(); pack.MANIFEST = MANIFEST; pack.paths = paths
    print(dict(action=args.action, **getattr(pack, args.action)(Path(__file__).resolve().parents[1])), flush=True)
