"""397--403 prefix pilot, readout preflight and setup correction publication."""
import argparse
import hashlib
from pathlib import Path
import re
import region_credit_snapshot_20260925 as pack

PARENT='71c38f7b2f08c1932464224e7d1c0261b50e03dd'
MANIFEST='release/prefix-deadline-20260925.json'
BASE='results/prefix_deadline_pilot'


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
            return
        if name in tree:
            payload=(root/name).read_bytes()
            assert tree[name]==hashlib.sha1(b'blob '+str(len(payload)).encode()+b'\0'+payload).hexdigest(),name
        else:files.add(name)
    for stage in ['preflight_v1','development_v1','audit_preflight_v1','audit_v1','evaluation_v1']:
        folder=root/BASE/stage;summary=pack.read(folder/'summary.json')
        assert summary['passed'] and not (folder/'failure.json').exists()
        for name,digest in summary['outputs_sha256'].items():assert pack.sha(folder/name)==digest
        if stage in ['preflight_v1','development_v1']:
            protocol=pack.read(folder/'protocol.json')
            for field in ['source_sha256','pretrained_sha256']:
                for name,digest in protocol[field].items():immutable(name,digest)
            for name,digest in pack.read(folder/'inputs_manifest.json').items():assert pack.sha(folder/'inputs'/name)==digest
            rows=pack.read(folder/'rows.json');assert len(rows)==protocol['expected_calls']
            for row in rows:
                for name,digest in row['files'].items():assert pack.sha(root/row['directory']/name)==digest
        elif stage in ['audit_preflight_v1','audit_v1']:
            predicted='preflight_v1' if stage=='audit_preflight_v1' else 'development_v1'
            assert summary['prediction_summary_sha256']==pack.sha(root/BASE/predicted/'summary.json')
        files.update(p.relative_to(root).as_posix() for p in folder.rglob('*') if p.is_file())
    before=pack.read(root/BASE/'evaluation_v1/before_query.json')
    assert before['prediction_summary_sha256']==pack.sha(root/BASE/'development_v1/summary.json')
    assert before['audit_summary_sha256']==pack.sha(root/BASE/'audit_v1/summary.json')
    report=root/BASE/'report_v1';manifest=pack.read(report/'manifest.json')
    numeric=pack.read(report/'qa_numeric.json');visual=pack.read(report/'visual_qa.json')
    assert numeric['passed'] and numeric['manifest_sha256']==pack.sha(report/'manifest.json')
    assert visual['passed'] and visual['actually_viewed'] and visual['numeric_qa_sha256']==pack.sha(report/'qa_numeric.json')
    assert manifest['evaluation_summary_sha256']==pack.sha(root/BASE/'evaluation_v1/summary.json')
    for row in visual['images']:assert row['sha256']==pack.sha(report/row['file'])
    for name,digest in manifest['outputs_sha256'].items():assert pack.sha(report/name)==digest
    assert manifest['source_sha256']==pack.sha(root/'work/experiments/report_prefix_deadline_pilot_v1.py')
    assert manifest['entry_sha256']==pack.sha(root/manifest['entry_file'])
    files.update(p.relative_to(root).as_posix() for p in report.iterdir() if p.is_file())
    math_folder=root/'results/certified_partial_readout/math_preflight_v1'
    math_summary=pack.read(math_folder/'summary.json')
    assert math_summary['passed'] and not (math_folder/'failure.json').exists()
    for name,digest in math_summary['outputs_sha256'].items():assert pack.sha(math_folder/name)==digest
    for name,digest in math_summary['source_sha256'].items():immutable(name,digest)
    files.update(p.relative_to(root).as_posix() for p in math_folder.iterdir() if p.is_file())
    math_entry='outputs/ttt-pc-alm-research/401_certified_partial_readout_preflight_v1.md'
    setup_folder=root/'results/portfolio_runtime_setup/preflight_v1'
    setup_summary=pack.read(setup_folder/'summary.json')
    assert setup_summary['passed'] and not (setup_folder/'failure.json').exists()
    for name,digest in setup_summary['outputs_sha256'].items():assert pack.sha(setup_folder/name)==digest
    for name,digest in setup_summary['source_sha256'].items():immutable(name,digest)
    for row in pack.read(setup_folder/'rows.json'):
        for name,digest in row['files'].items():assert pack.sha(root/row['directory']/name)==digest
    files.update(p.relative_to(root).as_posix() for p in setup_folder.rglob('*') if p.is_file())
    setup_entry='outputs/ttt-pc-alm-research/403_portfolio_runtime_setup_results_v1.md'
    files.update(['scripts/prefix_deadline_snapshot_20260925.py','scripts/region_credit_snapshot_20260925.py',
                  'release/PREFIX-DEADLINE-20260925.md',manifest['entry_file'],math_entry,setup_entry])
    for doc in [report/'report.md',root/manifest['entry_file'],root/'release/PREFIX-DEADLINE-20260925.md',root/math_entry,root/setup_entry]:
        for link in re.findall(r'\]\(([^)]+)\)',doc.read_text(encoding='utf-8')):
            if not link.startswith(('https:','http:','#')):
                assert (doc.parent/link.split('#')[0]).resolve().is_file(),(doc,link)
    return sorted(files)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['build','stage','verify']);args=parser.parse_args()
    pack.MANIFEST=MANIFEST;pack.paths=paths
    print(dict(action=args.action,**getattr(pack,args.action)(Path(__file__).resolve().parents[1])),flush=True)
