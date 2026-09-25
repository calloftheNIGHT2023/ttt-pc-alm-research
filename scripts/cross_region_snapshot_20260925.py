"""357--359 component evidence plus compact sealed online snapshot."""
import argparse
import hashlib
from pathlib import Path
import region_credit_snapshot_20260925 as pack

MANIFEST='release/cross-region-credit-20260925.json'
PARENT='3976c2c088ca339fca3b5ec3d93ad7efba2e5cd2'


def paths(root):
    online='results/cross_region_online';component='results/cross_region_credit'
    protocol=pack.read(root/online/'development_predictions_v1/protocol.json')
    assert protocol['primary']=='cross_dual_reuse_g8'
    # Prior dependencies already belong to the immutable parent Git tree.
    # Do not repackage its unrelated historical analysis scripts as new code.
    tree={}
    for item in pack.git(root,'ls-tree','-r','-z',PARENT).decode().strip('\0').split('\0'):
        header,name=item.split('\t',1);tree[name]=header.split()[2]
    files=set()
    for path in protocol['source_sha256']:
        if path in tree:
            payload=(root/path).read_bytes()
            assert tree[path]==hashlib.sha1(b'blob '+str(len(payload)).encode()+b'\0'+payload).hexdigest()
        else:files.add(path)
    files.update(['scripts/cross_region_snapshot_20260925.py','scripts/region_credit_snapshot_20260925.py',
        'release/CROSS-REGION-CREDIT-20260925.md','outputs/ttt-pc-alm-research/359_cross_region_online_results_v1.md',
        'work/experiments/report_cross_region_online_v1.py','work/experiments/audit_cross_region_report_v1.py',
        'work/experiments/audit_cross_region_attribution_v1.py',online+'/attribution_v1/summary.json'])
    for name in ['preflight_v1','development_v1','audit_v1']:
        assert pack.read(root/component/name/'summary.json')['passed']
        files.add(component+'/'+name+'/summary.json')
    files.update(component+'/'+p for p in ['preflight_attempt1/failure.json','preflight_attempt1/test_source.py',
                                         'development_v1/protocol.json','development_v1/tasks.json','audit_v1/rows.json'])
    for task in pack.read(root/component/'development_v1/tasks.json'):
        prefix=component+'/development_v1/'+str(task['seed'])+'/'
        files.add(prefix+'commit.json');files.update(prefix+n for n in task['files'])
        files.add(component+'/audit_v1/'+str(task['seed'])+'_geometry.json')
    for phase,names in {
        'preflight_predictions_v1':['summary.json','protocol.json','rows.json','before_query_manifest.json'],
        'development_predictions_v1':['summary.json','protocol.json','rows.json','before_query_manifest.json'],
        'development_evaluation_v1':['summary.json','protocol.json','methods.json','comparisons.json','groups.json','task_metrics.npz'],
        'development_audit_v1':['summary.json','mechanisms.json']}.items():
        folder=root/online/phase;assert not (folder/'failure.json').exists() and pack.read(folder/'summary.json')['passed']
        files.update(online+'/'+phase+'/'+n for n in names)
    report=root/online/'report_v1';manifest=pack.read(report/'manifest.json');qa=pack.read(report/'qa_numeric.json');visual=pack.read(report/'visual_qa.json')
    assert qa['passed'] and qa['manifest_sha256']==pack.sha(report/'manifest.json')
    assert visual['passed'] and visual['actually_viewed'] and visual['numeric_qa_sha256']==pack.sha(report/'qa_numeric.json')
    for image in visual['images']:assert pack.sha(report/image['file'])==image['sha256']
    for n,h in manifest['outputs_sha256'].items():assert pack.sha(report/n)==h
    assert pack.sha(root/manifest['entry_file'])==manifest['entry_sha256']
    files.update(online+'/report_v1/'+n for n in [*manifest['outputs_sha256'],'manifest.json','qa_numeric.json','visual_qa.json'])
    for p,h in protocol['source_sha256'].items():assert pack.sha(root/p)==h
    return sorted(files)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['build','verify','stage']);args=ap.parse_args()
    pack.MANIFEST=MANIFEST;pack.paths=paths
    print(dict(action=args.action,**getattr(pack,args.action)(Path(__file__).resolve().parents[1])),flush=True)
