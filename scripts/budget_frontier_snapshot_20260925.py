"""360--361 all saved-input component outputs, fixed-parent input verification."""
import argparse
import hashlib
from pathlib import Path
import region_credit_snapshot_20260925 as pack

BASE = 'results/budget_frontier'
PARENT = 'ee0907ad75ce87299d480bbcbb050ca56a0e4556'
MANIFEST = 'release/budget-frontier-20260925.json'


def paths(root):
    source = root/BASE/'development_v1'; p = pack.read(source/'protocol.json')
    files = set(p['source_sha256'])
    files.update(['scripts/budget_frontier_snapshot_20260925.py', 'scripts/region_credit_snapshot_20260925.py',
                  'release/BUDGET-FRONTIER-20260925.md',
                  'outputs/ttt-pc-alm-research/361_budget_frontier_results_v1.md',
                  'work/experiments/report_budget_frontier_v1.py', 'work/experiments/audit_budget_frontier_report_v1.py'])
    for phase in ['preflight_v1', 'development_v1', 'audit_v1']:
        folder = root/BASE/phase
        assert not (folder/'failure.json').exists() and pack.read(folder/'summary.json')['passed']
        files.add(BASE+'/'+phase+'/summary.json')
    for name in ['protocol.json','rows.json','frontiers.json','positive_frontiers.json','input_seals.json']:
        files.add(BASE+'/development_v1/'+name)
    for row in pack.read(source/'rows.json'):
        prefix = BASE+'/development_v1/'+str(row['seed'])+'/'
        files.add(prefix+'commit.json')
        for ext, digest in row['files'].items():
            name = prefix+row['method']+ext
            assert pack.sha(root/name) == digest; files.add(name)
    tree = {}
    for record in pack.git(root,'ls-tree','-r','-z',PARENT).decode().strip('\0').split('\0'):
        header, name = record.split('\t',1); tree[name] = header.split()[2]
    def parent_check(name, digest):
        payload = (root/name).read_bytes()
        assert pack.sha(root/name) == digest
        assert tree[name] == hashlib.sha1(b'blob '+str(len(payload)).encode()+b'\0'+payload).hexdigest()
    parent_check('results/cross_region_credit/development_v1/protocol.json',p['source_protocol_sha256'])
    parent_check('results/cross_region_credit/development_v1/tasks.json',p['source_tasks_sha256'])
    parent_check('results/cross_region_credit/audit_v1/summary.json',p['geometry_summary_sha256'])
    for row in pack.read(source/'input_seals.json'):
        prefix = 'results/cross_region_credit/development_v1/'+str(row['seed'])+'/'
        for name, digest in row['source_files'].items():
            parent_check(prefix+name,digest)
        parent_check('results/cross_region_credit/audit_v1/'+str(row['seed'])+'_geometry.json',row['geometry_sha256'])
    report = root/BASE/'report_v1'; m = pack.read(report/'manifest.json'); qa = pack.read(report/'qa_numeric.json'); visual = pack.read(report/'visual_qa.json')
    assert qa['passed'] and qa['manifest_sha256'] == pack.sha(report/'manifest.json')
    assert visual['passed'] and visual['actually_viewed'] and visual['numeric_qa_sha256'] == pack.sha(report/'qa_numeric.json')
    for image in visual['images']:
        assert image['sha256'] == pack.sha(report/image['file'])
    for name, digest in m['outputs_sha256'].items():
        assert pack.sha(report/name) == digest
    assert pack.sha(root/m['entry_file']) == m['entry_sha256']
    files.update(BASE+'/report_v1/'+n for n in [*m['outputs_sha256'],'manifest.json','qa_numeric.json','visual_qa.json'])
    for name, digest in p['source_sha256'].items():
        assert pack.sha(root/name) == digest
    return sorted(files)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('action',choices=['build','verify','stage']); args = ap.parse_args()
    pack.MANIFEST = MANIFEST; pack.paths = paths
    print(dict(action=args.action, **getattr(pack,args.action)(Path(__file__).resolve().parents[1])), flush=True)
