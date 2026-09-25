"""376--377 complete active/passive regional states, proof arrays and resources."""
import argparse,hashlib,re
from pathlib import Path
import region_credit_snapshot_20260925 as pack

PARENT='5a9b6eaa6052c3ede1b298fafb280fb238a2f7e0'
BASE='results/regional_alm'
MANIFEST='release/regional-alm-20260925.json'


def paths(root):
    tree={}
    for item in pack.git(root,'ls-tree','-r','-z',PARENT).decode().strip('\0').split('\0'):
        header,name=item.split('\t',1);tree[name]=header.split()[2]
    files=set();p=pack.read(root/BASE/'development_v1/protocol.json')
    for name,h in p['source_sha256'].items():
        payload=(root/name).read_bytes();assert pack.sha(root/name)==h
        if name in tree:assert tree[name]==hashlib.sha1(b'blob '+str(len(payload)).encode()+b'\0'+payload).hexdigest()
        else:files.add(name)
    for name in ['preflight_v1','development_v1','audit_v1']:
        folder=root/BASE/name;s=pack.read(folder/'summary.json');assert s['passed'] and not (folder/'failure.json').exists()
        for file,h in s.get('outputs_sha256',{}).items():assert pack.sha(folder/file)==h
        files.update(str(f.relative_to(root)).replace('\\','/') for f in folder.rglob('*') if f.is_file())
    for row in pack.read(root/BASE/'development_v1/rows.json'):
        for f,h in row['files'].items():assert pack.sha(root/row['directory']/f)==h
    report=root/BASE/'report_v1';m=pack.read(report/'manifest.json');qa=pack.read(report/'qa_numeric.json');visual=pack.read(report/'visual_qa.json')
    assert qa['passed'] and qa['manifest_sha256']==pack.sha(report/'manifest.json')
    assert visual['passed'] and visual['actually_viewed'] and visual['numeric_qa_sha256']==pack.sha(report/'qa_numeric.json')
    for image in visual['images']:assert image['sha256']==pack.sha(report/image['file'])
    for file,h in m['outputs_sha256'].items():assert pack.sha(report/file)==h
    assert m['entry_sha256']==pack.sha(root/m['entry_file'])
    assert m['source_sha256']==pack.sha(root/'work/experiments/report_regional_alm_v1.py')
    for document in [report/'report.md',root/m['entry_file'],root/'release/REGIONAL-ALM-20260925.md']:
        for link in re.findall(r'\]\(([^)]+)\)',document.read_text(encoding='utf-8')):
            if not link.startswith(('https:','http:','#')):assert (document.parent/link.split('#')[0]).resolve().is_file(),(document,link)
    files.update(str(f.relative_to(root)).replace('\\','/') for f in report.iterdir() if f.is_file())
    files.update([m['entry_file'],'release/REGIONAL-ALM-20260925.md','scripts/regional_alm_snapshot_20260925.py',
        'scripts/region_credit_snapshot_20260925.py','work/experiments/report_regional_alm_v1.py'])
    return sorted(files)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['build','stage','verify']);args=parser.parse_args()
    pack.MANIFEST=MANIFEST;pack.paths=paths
    print(dict(action=args.action,**getattr(pack,args.action)(Path(__file__).resolve().parents[1])),flush=True)
