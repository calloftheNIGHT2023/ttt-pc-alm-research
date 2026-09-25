"""365--367 full new predictions, auditable noon report and explicit Git scope."""
import argparse
import hashlib
from pathlib import Path
import region_credit_snapshot_20260925 as pack

PARENT='ea9eea9de87b6258f3f75fb05cd321243bc8ad42'
BASE='results/direct_language'
MANIFEST='release/direct-language-20260925.json'


def paths(root):
    p=pack.read(root/BASE/'development_predictions_v1/protocol.json')
    tree={}
    for record in pack.git(root,'ls-tree','-r','-z',PARENT).decode().strip('\0').split('\0'):
        header,name=record.split('\t',1);tree[name]=header.split()[2]
    files=set()
    for name,digest in p['source_sha256'].items():
        payload=(root/name).read_bytes();assert pack.sha(root/name)==digest
        if name in tree:assert tree[name]==hashlib.sha1(b'blob '+str(len(payload)).encode()+b'\0'+payload).hexdigest()
        else:files.add(name)
    for phase in ['preflight_predictions_v1','development_predictions_v1','development_evaluation_v1','replay_v1']:
        directory=root/BASE/phase;summary=pack.read(directory/'summary.json')
        assert summary['passed'] and not (directory/'failure.json').exists()
        for name,digest in summary.get('outputs_sha256',{}).items():assert pack.sha(directory/name)==digest
        files.update(str(f.relative_to(root)).replace('\\','/') for f in directory.rglob('*') if f.is_file())
    report=root/BASE/'report_v1';m=pack.read(report/'manifest.json');qa=pack.read(report/'qa_numeric.json');visual=pack.read(report/'visual_qa.json')
    assert qa['passed'] and qa['manifest_sha256']==pack.sha(report/'manifest.json')
    assert visual['passed'] and visual['actually_viewed'] and visual['numeric_qa_sha256']==pack.sha(report/'qa_numeric.json')
    for image in visual['images']:assert image['sha256']==pack.sha(report/image['file'])
    for name,digest in m['outputs_sha256'].items():assert pack.sha(report/name)==digest
    assert pack.sha(root/m['entry_file'])==m['entry_sha256']
    noon=pack.read(report/'qa_noon_overview.json');assert noon['passed']
    assert noon['overview_sha256']==pack.sha(root/'outputs/ttt-pc-alm-research/367_noon_research_delivery_20260925.md')
    for link in noon['links']:
        assert pack.sha(root/link['file'])==link['sha256']
        if link['file'] not in tree:files.add(link['file'])
    files.update(str(f.relative_to(root)).replace('\\','/') for f in report.iterdir() if f.is_file())
    files.update(['scripts/direct_language_snapshot_20260925.py','scripts/region_credit_snapshot_20260925.py',
        'release/NOON-20260925.md','outputs/ttt-pc-alm-research/366_direct_language_results_v1.md',
        'outputs/ttt-pc-alm-research/367_noon_research_delivery_20260925.md',
        'work/experiments/report_direct_language_v1.py','work/experiments/replay_direct_language_v1.py',
        'work/experiments/audit_noon_delivery_v1.py'])
    return sorted(files)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['build','verify','stage']);args=ap.parse_args()
    pack.MANIFEST=MANIFEST;pack.paths=paths
    print(dict(action=args.action,**getattr(pack,args.action)(Path(__file__).resolve().parents[1])),flush=True)
