"""413--415 exact support certificates and archived-frontier component evidence."""
import argparse
import hashlib
import re
from pathlib import Path
import region_credit_snapshot_20260925 as pack

PARENT='f6c9b10f6245548ce861c47debbcc308bc2e4856'
MANIFEST='release/frontier-certificates-20260925.json'
ENTRY='release/FRONTIER-CERTIFICATES-20260925.md'


def paths(root):
    tree={};submodules={};files=set()
    for record in pack.git(root,'ls-tree','-r','-z',PARENT).decode().strip('\0').split('\0'):
        header,name=record.split('\t',1);mode,kind,oid=header.split();tree[name]=oid
        if mode=='160000':submodules[name]=oid
    def immutable(name,digest):
        assert pack.sha(root/name)==digest,name
        external=next((s for s in submodules if name.startswith(s+'/')),None)
        if external:
            assert pack.git(root/external,'rev-parse','HEAD').decode().strip()==submodules[external]
            return
        if name in tree:
            payload=(root/name).read_bytes()
            assert tree[name]==hashlib.sha1(b'blob '+str(len(payload)).encode()+b'\0'+payload).hexdigest(),name
        else:files.add(name)
    for base,stages in [('results/partial_support_certificate',['preflight_v1','development_v1','audit_v1']),
                        ('results/frontier_range_certificate',['preflight_v1','development_v1'])]:
        for stage in stages:
            folder=root/base/stage;summary=pack.read(folder/'summary.json')
            assert summary['passed'] and not (folder/'failure.json').exists()
            for path,digest in summary.get('outputs_sha256',{}).items():assert pack.sha(folder/path)==digest
            for path,digest in summary.get('source_sha256',{}).items():immutable(path,digest)
            if stage=='development_v1':
                protocol=pack.read(folder/'protocol.json')
                for path,digest in protocol['source_sha256'].items():immutable(path,digest)
                rows=pack.read(folder/'rows.json');assert len(rows)==16
                for row in rows:
                    directory=root/row['directory']
                    for path,digest in row['files'].items():assert pack.sha(directory/path)==digest
                    if (directory/'baseline_references.json').is_file():
                        for path,digest in pack.read(directory/'baseline_references.json').items():immutable(path,digest)
            files.update(p.relative_to(root).as_posix() for p in folder.rglob('*') if p.is_file())
    partial=root/'results/partial_support_certificate';frontier=root/'results/frontier_range_certificate'
    a=pack.read(partial/'audit_v1/summary.json')
    assert a['development_summary_sha256']==pack.sha(partial/'development_v1/summary.json')
    p=pack.read(frontier/'development_v1/protocol.json')
    assert p['source_summary_sha256']==pack.sha(partial/'development_v1/summary.json')
    assert p['source_audit_sha256']==pack.sha(partial/'audit_v1/summary.json')
    assert p['preflight_sha256']==pack.sha(frontier/'preflight_v1/summary.json')
    report=pack.read(frontier/'development_v1/report_manifest.json')
    assert report['entry_sha256']==pack.sha(root/report['entry_file'])
    assert report['summary_sha256']==pack.sha(frontier/'development_v1/summary.json')
    files.update(['scripts/frontier_certificate_snapshot_20260925.py',ENTRY,report['entry_file']])
    for doc in [root/ENTRY,root/report['entry_file']]:
        for link in re.findall(r'\]\(([^)]+)\)',doc.read_text(encoding='utf-8')):
            if not link.startswith(('https:','http:','#')):assert (doc.parent/link.split('#')[0]).resolve().is_file(),(doc,link)
    return sorted(files)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['build','stage','verify']);args=parser.parse_args()
    pack.MANIFEST=MANIFEST;pack.paths=paths
    print(dict(action=args.action,**getattr(pack,args.action)(Path(__file__).resolve().parents[1])),flush=True)
