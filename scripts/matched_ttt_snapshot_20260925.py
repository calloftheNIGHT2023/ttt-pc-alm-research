"""390 / 393 / 395 / 396 completed, exact bounded evidence publication."""
import argparse
import hashlib
from pathlib import Path
import re
import region_credit_snapshot_20260925 as pack

PARENT='a5e540f8ea2f6866274e0517d92a30cd84f6a579'
MANIFEST='release/matched-ttt-20260925.json'


def paths(root):
    tree={};submodules={};files=set()
    for record in pack.git(root,'ls-tree','-r','-z',PARENT).decode().strip('\0').split('\0'):
        header,name=record.split('\t',1);mode,kind,oid=header.split();tree[name]=oid
        if mode=='160000':submodules[name]=oid
    def immutable(name,digest):
        assert pack.sha(root/name)==digest,name
        external=next((s for s in submodules if name.startswith(s+'/')),None)
        if external is not None:
            assert pack.git(root/external,'rev-parse','HEAD').decode().strip()==submodules[external]
            return  # Source is retained by the existing pinned official submodule.
        if name in tree:
            payload=(root/name).read_bytes()
            assert tree[name]==hashlib.sha1(b'blob '+str(len(payload)).encode()+b'\0'+payload).hexdigest(),name
        else:
            files.add(name)
    def folder(name):
        location=root/name;summary=pack.read(location/'summary.json')
        assert summary['passed'] and not (location/'failure.json').exists()
        for f,digest in summary.get('outputs_sha256',{}).items():
            assert pack.sha(location/f)==digest
        hashes=summary.get('source_sha256')
        if isinstance(hashes,dict):
            for f,digest in hashes.items():immutable(f,digest)
        files.update(p.relative_to(root).as_posix() for p in location.rglob('*') if p.is_file())
    for name in ['results/matched_official_ttt/preflight_v1',
                 'results/matched_official_ttt/training_v1',
                 'results/matched_official_ttt/training_audit_v1',
                 'results/low_support_coverage/math_preflight_v1',
                 'results/low_support_coverage/development_v1',
                 'results/low_support_coverage/audit_v1',
                 'results/prefix_matched_controls/preflight_v1']:
        folder(name)
    for name in ['results/matched_official_ttt/training_v1/protocol.json',
                 'results/low_support_coverage/development_v1/protocol.json']:
        for f,digest in pack.read(root/name)['source_sha256'].items():immutable(f,digest)
    for row in pack.read(root/'results/matched_official_ttt/training_v1/training_summary.json'):
        prefix=root/'results/matched_official_ttt/training_v1'
        assert pack.sha(prefix/(row['method']+'.pt'))==row['checkpoint_sha256']
        assert pack.sha(prefix/(row['method']+'_consumed_batches.json'))==row['consumed_batches_sha256']
    for row in pack.read(root/'results/low_support_coverage/development_v1/sealed_banks.json'):
        for f,digest in row['files'].items():assert pack.sha(root/row['directory']/f)==digest
    for row in pack.read(root/'results/low_support_coverage/development_v1/references.json'):
        for f,digest in row['files'].items():assert pack.sha(root/row['directory']/f)==digest
        if row['fully_resolved']:assert pack.sha(root/row['directory']/'integration.npz')==row['integration_sha256']
    report=root/'results/matched_official_ttt/report_v1';manifest=pack.read(report/'manifest.json')
    numeric=pack.read(report/'qa_numeric.json');visual=pack.read(report/'visual_qa.json')
    assert numeric['passed'] and numeric['manifest_sha256']==pack.sha(report/'manifest.json')
    assert visual['passed'] and visual['actually_viewed'] and visual['numeric_qa_sha256']==pack.sha(report/'qa_numeric.json')
    for row in visual['images']:assert row['sha256']==pack.sha(report/row['file'])
    for name,digest in manifest['outputs_sha256'].items():assert pack.sha(report/name)==digest
    assert manifest['source_sha256']==pack.sha(root/'work/experiments/report_matched_training_v1.py')
    assert manifest['entry_sha256']==pack.sha(root/manifest['entry_file'])
    files.update(p.relative_to(root).as_posix() for p in report.iterdir() if p.is_file())
    for name in ['work/experiments/audit_matched_official_ttt_training_v1.py',
                 'work/experiments/audit_low_support_coverage_v1.py',
                 'work/experiments/report_matched_training_v1.py',
                 'scripts/matched_ttt_snapshot_20260925.py',
                 'scripts/region_credit_snapshot_20260925.py',
                 'release/MATCHED-TTT-20260925.md',manifest['entry_file']]:files.add(name)
    for doc in [report/'report.md',root/manifest['entry_file'],root/'release/MATCHED-TTT-20260925.md']:
        for link in re.findall(r'\]\(([^)]+)\)',doc.read_text(encoding='utf-8')):
            if not link.startswith(('https:','http:','#')):
                assert (doc.parent/link.split('#')[0]).resolve().is_file(),(doc,link)
    return sorted(files)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['build','stage','verify']);args=parser.parse_args()
    pack.MANIFEST=MANIFEST;pack.paths=paths
    print(dict(action=args.action,**getattr(pack,args.action)(Path(__file__).resolve().parents[1])),flush=True)
