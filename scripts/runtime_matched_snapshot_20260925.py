"""405--410 corrected-runtime evidence and common-objective component checks."""
import argparse
import hashlib
import re
from pathlib import Path
import region_credit_snapshot_20260925 as pack

PARENT='00aabd408a85b086e97889272a5da625126aa4e0'
MANIFEST='release/runtime-matched-20260925.json'
BASE='results/runtime_matched_prefix'
ENTRY='release/RUNTIME-MATCHED-20260925.md'


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
    def checked_outputs(folder,summary):
        assert summary['passed'] and not (folder/'failure.json').exists()
        for name,digest in summary.get('outputs_sha256',{}).items():assert pack.sha(folder/name)==digest
        for name,digest in summary.get('source_sha256',{}).items():immutable(name,digest)
    binding=root/BASE/'binding_preflight_v1';bound=pack.read(binding/'summary.json')
    checked_outputs(binding,bound)
    assert bound['registry_entries']==42 and bound['deterministic_unique_jobs']==1344
    assert bound['new_worker_calls']==0 and bound['reused_worker_preflight_calls']==32
    for field in ['pretrained_sha256','reused_preflight_summary_sha256']:
        for name,digest in bound[field].items():immutable(name,digest)
    files.update(p.relative_to(root).as_posix() for p in binding.rglob('*') if p.is_file())
    for stage in ['development_v1','audit_v1','evaluation_v1']:
        folder=root/BASE/stage;summary=pack.read(folder/'summary.json');checked_outputs(folder,summary)
        if stage=='development_v1':
            protocol=pack.read(folder/'protocol.json')
            assert protocol['portfolio_task_free_module_preload'] and protocol['expected_calls']==1344
            for field in ['source_sha256','pretrained_sha256','reused_preflight_summary_sha256']:
                for name,digest in protocol[field].items():immutable(name,digest)
            for name,digest in pack.read(folder/'inputs_manifest.json').items():assert pack.sha(folder/'inputs'/name)==digest
            rows=pack.read(folder/'rows.json');assert len(rows)==1344
            for row in rows:
                for name,digest in row['files'].items():assert pack.sha(root/row['directory']/name)==digest
        elif stage=='audit_v1':
            assert summary['prediction_summary_sha256']==pack.sha(root/BASE/'development_v1/summary.json')
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
    assert manifest['source_sha256']==pack.sha(root/'work/experiments/report_runtime_matched_prefix_v1.py')
    assert manifest['entry_sha256']==pack.sha(root/manifest['entry_file'])
    files.update(p.relative_to(root).as_posix() for p in report.iterdir() if p.is_file())
    for relative in ['results/common_band_objective/math_preflight_v1',
                     'results/common_band_optimizers/preflight_v1']:
        folder=root/relative;summary=pack.read(folder/'summary.json');checked_outputs(folder,summary)
        files.update(p.relative_to(root).as_posix() for p in folder.rglob('*') if p.is_file())
    opt_protocol=pack.read(root/'results/common_band_optimizers/preflight_v1/protocol.json')
    assert opt_protocol['exact_math_summary_sha256']==pack.sha(root/'results/common_band_objective/math_preflight_v1/summary.json')
    component_entry='outputs/ttt-pc-alm-research/410_common_band_preflight_results_v1.md'
    files.update(['scripts/runtime_matched_snapshot_20260925.py',ENTRY,manifest['entry_file'],component_entry])
    for doc in [report/'report.md',root/manifest['entry_file'],root/ENTRY,root/component_entry]:
        for link in re.findall(r'\]\(([^)]+)\)',doc.read_text(encoding='utf-8')):
            if not link.startswith(('https:','http:','#')):
                assert (doc.parent/link.split('#')[0]).resolve().is_file(),(doc,link)
    return sorted(files)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['build','stage','verify']);args=parser.parse_args()
    pack.MANIFEST=MANIFEST;pack.paths=paths
    print(dict(action=args.action,**getattr(pack,args.action)(Path(__file__).resolve().parents[1])),flush=True)
