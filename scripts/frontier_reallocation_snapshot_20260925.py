"""362--364 full component evidence and compact sealed actual-online results."""
import argparse
import hashlib
from pathlib import Path
import region_credit_snapshot_20260925 as pack

COMPONENT='results/frontier_reallocation';ONLINE='results/frontier_online'
PARENT='289800a3f948900b480aff4e647a1af6ce7e3326'
MANIFEST='release/frontier-reallocation-20260925.json'


def paths(root):
    p=pack.read(root/ONLINE/'development_predictions_v1/protocol.json')
    assert p['primary']=='frontier_dual_frontier_g8' and len(p['configs'])==17
    tree={}
    for record in pack.git(root,'ls-tree','-r','-z',PARENT).decode().strip('\0').split('\0'):
        header,name=record.split('\t',1);tree[name]=header.split()[2]
    def parent_check(name,digest):
        payload=(root/name).read_bytes();assert pack.sha(root/name)==digest
        assert tree[name]==hashlib.sha1(b'blob '+str(len(payload)).encode()+b'\0'+payload).hexdigest()
    files=set()
    for name,digest in p['source_sha256'].items():
        assert pack.sha(root/name)==digest
        if name in tree:parent_check(name,digest)
        else:files.add(name)
    files.update(['scripts/frontier_reallocation_snapshot_20260925.py','scripts/region_credit_snapshot_20260925.py',
        'release/FRONTIER-REALLOCATION-20260925.md','outputs/ttt-pc-alm-research/364_frontier_online_results_v1.md',
        'outputs/ttt-pc-alm-research/363_posterior_coverage_bound_appendix.md',
        'work/experiments/audit_frontier_attribution_v1.py','work/experiments/report_frontier_online_v1.py',
        'work/experiments/audit_frontier_report_v1.py'])
    for phase in ['preflight_v1','development_v1','audit_v1']:
        folder=root/COMPONENT/phase;assert pack.read(folder/'summary.json')['passed'] and not (folder/'failure.json').exists()
        files.add(COMPONENT+'/'+phase+'/summary.json')
    files.update(COMPONENT+'/'+n for n in ['development_v1/protocol.json','development_v1/rows.json','development_v1/input_seals.json','audit_v1/rows.json'])
    for row in pack.read(root/COMPONENT/'development_v1/rows.json'):
        prefix=COMPONENT+'/development_v1/'+str(row['seed'])+'/'
        files.add(prefix+'commit.json')
        for ext,digest in row['files'].items():
            name=prefix+row['method']+ext;assert pack.sha(root/name)==digest;files.add(name)
    for seal in pack.read(root/COMPONENT/'development_v1/input_seals.json'):
        prefix='results/cross_region_credit/development_v1/'+str(seal['seed'])+'/'
        for name,digest in seal['source_files'].items():parent_check(prefix+name,digest)
    for phase,names in {
        'preflight_predictions_v1':['summary.json','protocol.json','rows.json','before_query_manifest.json'],
        'development_predictions_v1':['summary.json','protocol.json','rows.json','before_query_manifest.json'],
        'development_evaluation_v1':['summary.json','protocol.json','methods.json','comparisons.json','groups.json','task_metrics.npz'],
        'development_audit_v1':['summary.json','mechanisms.json'],
        'attribution_v1':['summary.json','pairs.json']}.items():
        folder=root/ONLINE/phase;assert pack.read(folder/'summary.json')['passed'] and not (folder/'failure.json').exists()
        files.update(ONLINE+'/'+phase+'/'+name for name in names)
    report=root/ONLINE/'report_v1';m=pack.read(report/'manifest.json');qa=pack.read(report/'qa_numeric.json');visual=pack.read(report/'visual_qa.json')
    assert qa['passed'] and qa['manifest_sha256']==pack.sha(report/'manifest.json')
    assert visual['passed'] and visual['actually_viewed'] and visual['numeric_qa_sha256']==pack.sha(report/'qa_numeric.json')
    for image in visual['images']:assert image['sha256']==pack.sha(report/image['file'])
    for name,digest in m['outputs_sha256'].items():assert pack.sha(report/name)==digest
    assert pack.sha(root/m['entry_file'])==m['entry_sha256']
    files.update(ONLINE+'/report_v1/'+name for name in [*m['outputs_sha256'],'manifest.json','qa_numeric.json','visual_qa.json'])
    return sorted(files)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['build','verify','stage']);args=ap.parse_args()
    pack.MANIFEST=MANIFEST;pack.paths=paths
    print(dict(action=args.action,**getattr(pack,args.action)(Path(__file__).resolve().parents[1])),flush=True)
