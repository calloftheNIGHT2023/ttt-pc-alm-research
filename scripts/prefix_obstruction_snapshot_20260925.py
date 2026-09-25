"""372--373 complete diagnostic, preserved harness failures and exact audit."""
import argparse,hashlib,re
from pathlib import Path
import region_credit_snapshot_20260925 as pack

PARENT='4a50252828eac8780f7665f9c5f710d359560b53'
BASE='results/prefix_obstruction'
MANIFEST='release/prefix-obstruction-20260925.json'


def paths(root):
    tree={}
    for item in pack.git(root,'ls-tree','-r','-z',PARENT).decode().strip('\0').split('\0'):
        header,name=item.split('\t',1);tree[name]=header.split()[2]
    files=set();protocol=pack.read(root/BASE/'development_v3/protocol.json')
    for name,digest in protocol['source_sha256'].items():
        payload=(root/name).read_bytes();assert pack.sha(root/name)==digest
        if name in tree:assert tree[name]==hashlib.sha1(b'blob '+str(len(payload)).encode()+b'\0'+payload).hexdigest()
        else:files.add(name)
    for name in ['selection_v1','selection_v2','selection_v3','preflight_v2','preflight_v3','development_v3','audit_v3']:
        folder=root/BASE/name;s=pack.read(folder/'summary.json');assert s['passed'] and not (folder/'failure.json').exists()
        for f,h in s.get('outputs_sha256',{}).items():assert pack.sha(folder/f)==h
        files.update(str(p.relative_to(root)).replace('\\','/') for p in folder.rglob('*') if p.is_file())
    for name in ['preflight_v1','development_v2']:
        folder=root/BASE/name;assert (folder/'failure.json').exists()
        files.update(str(p.relative_to(root)).replace('\\','/') for p in folder.rglob('*') if p.is_file())
    assert pack.read(root/BASE/'selection_v3/repair_replay.json')['passed']
    assert pack.read(root/BASE/'development_v3/5920000_alm_dual/failed_v2_mask_replay.json')['passed']
    report=root/BASE/'report_v1';m=pack.read(report/'manifest.json');qa=pack.read(report/'qa_numeric.json');visual=pack.read(report/'visual_qa.json')
    assert qa['passed'] and qa['manifest_sha256']==pack.sha(report/'manifest.json')
    assert visual['passed'] and visual['actually_viewed'] and visual['numeric_qa_sha256']==pack.sha(report/'qa_numeric.json')
    for img in visual['images']:assert img['sha256']==pack.sha(report/img['file'])
    for f,h in m['outputs_sha256'].items():assert h==pack.sha(report/f)
    assert m['entry_sha256']==pack.sha(root/m['entry_file'])
    assert m['source_sha256']==pack.sha(root/'work/experiments/report_prefix_obstruction_v1.py')
    for document in [report/'report.md',root/m['entry_file'],root/'release/PREFIX-OBSTRUCTION-20260925.md']:
        for link in re.findall(r'\]\(([^)]+)\)',document.read_text(encoding='utf-8')):
            if not link.startswith(('https:','http:','#')):assert (document.parent/link.split('#')[0]).resolve().is_file(),(document,link)
    files.update(str(p.relative_to(root)).replace('\\','/') for p in report.iterdir() if p.is_file())
    files.update([m['entry_file'],'release/PREFIX-OBSTRUCTION-20260925.md','scripts/prefix_obstruction_snapshot_20260925.py',
        'scripts/region_credit_snapshot_20260925.py','work/experiments/report_prefix_obstruction_v1.py'])
    return sorted(files)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['build','stage','verify']);args=parser.parse_args()
    pack.MANIFEST=MANIFEST;pack.paths=paths
    print(dict(action=args.action,**getattr(pack,args.action)(Path(__file__).resolve().parents[1])),flush=True)
