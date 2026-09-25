"""368--371 complete component evidence, preserved repair, theorem and figures."""
import argparse
import hashlib
import re
from pathlib import Path
import region_credit_snapshot_20260925 as pack

PARENT='53aba6cdacc9682cb7537dbc4552db3a1cf3c596'
BASE='results/multiplier_constraint_order'
MANIFEST='release/constraint-order-20260925.json'


def paths(root):
    p=pack.read(root/BASE/'development_v1/protocol.json');tree={}
    for record in pack.git(root,'ls-tree','-r','-z',PARENT).decode().strip('\0').split('\0'):
        header,name=record.split('\t',1);tree[name]=header.split()[2]
    files=set()
    def frozen(source):
        for name,digest in source.items():
            payload=(root/name).read_bytes();assert pack.sha(root/name)==digest
            if name in tree:assert tree[name]==hashlib.sha1(b'blob '+str(len(payload)).encode()+b'\0'+payload).hexdigest()
            else:files.add(name)
    frozen(p['source_sha256'])
    frozen(pack.read(root/'results/prefix_language_join_v2/development_v1/protocol.json')['source_sha256'])
    for name in ['results/prefix_language_join/preflight_v1','results/prefix_language_join_v2/preflight_v1',
        'results/prefix_language_join_v2/development_v1','results/parameter_arrangement_bound/audit_v1',
        BASE+'/preflight_v1',BASE+'/development_v1',BASE+'/audit_v1']:
        folder=root/name;s=pack.read(folder/'summary.json');assert s['passed'] and not (folder/'failure.json').exists()
        for file,h in s.get('outputs_sha256',{}).items():assert pack.sha(folder/file)==h
        files.update(str(f.relative_to(root)).replace('\\','/') for f in folder.rglob('*') if f.is_file())
    failed=root/'results/prefix_language_join/development_v1';assert (failed/'failure.json').is_file()
    files.update(str(f.relative_to(root)).replace('\\','/') for f in failed.rglob('*') if f.is_file())
    assert pack.read(root/'results/prefix_language_join_v2/development_v1/repair_replay.json')['passed']
    report=root/BASE/'report_v3';m=pack.read(report/'manifest.json');qa=pack.read(report/'qa_numeric.json');visual=pack.read(report/'visual_qa.json')
    assert qa['passed'] and qa['manifest_sha256']==pack.sha(report/'manifest.json')
    assert visual['passed'] and visual['actually_viewed'] and visual['numeric_qa_sha256']==pack.sha(report/'qa_numeric.json')
    for image in visual['images']:assert image['sha256']==pack.sha(report/image['file'])
    for file,h in m['outputs_sha256'].items():assert pack.sha(report/file)==h
    assert pack.sha(root/m['entry_file'])==m['entry_sha256']
    assert pack.sha(root/'work/experiments/report_multiplier_constraint_order_v3.py')==m['source_sha256']
    for document in [report/'report.md',root/m['entry_file'],root/'release/CONSTRAINT-ORDER-20260925.md']:
        for link in re.findall(r'\]\(([^)]+)\)',document.read_text(encoding='utf-8')):
            if not link.startswith(('https:','http:','#')):
                assert (document.parent/link.split('#')[0]).resolve().is_file(), (document,link)
    files.update(str(f.relative_to(root)).replace('\\','/') for f in report.iterdir() if f.is_file())
    theorem=pack.read(root/'results/parameter_arrangement_bound/audit_v1/summary.json')
    assert theorem['appendix_sha256']==pack.sha(root/'outputs/ttt-pc-alm-research/369_parameter_arrangement_bound_appendix_v1.md')
    files.update(['scripts/constraint_order_snapshot_20260925.py','scripts/region_credit_snapshot_20260925.py',
        'release/CONSTRAINT-ORDER-20260925.md','outputs/ttt-pc-alm-research/369_parameter_arrangement_bound_appendix_v1.md',
        'outputs/ttt-pc-alm-research/371_constraint_order_results_v3.md',
        'outputs/ttt-pc-alm-research/370_visual_revision_v2.md',
        'work/experiments/audit_parameter_arrangement_bound_v1.py','work/experiments/audit_multiplier_constraint_order_v1.py',
        'work/experiments/report_multiplier_constraint_order_v1.py','work/experiments/report_multiplier_constraint_order_v2.py',
        'work/experiments/report_multiplier_constraint_order_v3.py'])
    return sorted(files)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['build','verify','stage']);args=ap.parse_args()
    pack.MANIFEST=MANIFEST;pack.paths=paths
    print(dict(action=args.action,**getattr(pack,args.action)(Path(__file__).resolve().parents[1])),flush=True)
