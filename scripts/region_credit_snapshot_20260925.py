"""355--356 full saved-input component evidence; exact bounded Git staging."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess

BASE='results/region_conditioned_credit'
MANIFEST='release/region-credit-20260925.json'


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def git(root,*args):return subprocess.check_output(['git',*args],cwd=root)


def paths(root):
    protocol=read(root/BASE/'development_v1/protocol.json')
    files=set(protocol['source_sha256'])
    files.update(['scripts/region_credit_snapshot_20260925.py','release/REGION-CREDIT-20260925.md',
        'outputs/ttt-pc-alm-research/356_region_conditioned_credit_results_v1.md',
        'work/experiments/run_region_conditioned_credit_serialization_v2.py',
        'work/experiments/report_region_conditioned_credit_v1.py',
        'work/experiments/audit_region_conditioned_report_v1.py',
        'work/experiments/replay_region_conditioned_credit_v1.py',
        'work/experiments/streaming_branch_projection.py','work/experiments/local_branch_memory.py'])
    for relative in ['preflight_v1/summary.json','development_v1/summary.json','development_v1/protocol.json',
        'development_v1/tasks.json','development_attempt1/failure.json','development_attempt1/protocol.json',
        'audit_v1/summary.json','audit_v1/rows.json','replay_v1/summary.json']:
        files.add(BASE+'/'+relative)
    for row in read(root/BASE/'development_v1/tasks.json'):
        prefix=BASE+'/development_v1/'+str(row['seed'])+'/'
        files.add(prefix+'commit.json');files.update(prefix+n for n in row['files'])
    report=root/BASE/'report_v1';manifest=read(report/'manifest.json');qa=read(report/'qa_numeric.json');visual=read(report/'visual_qa.json')
    assert qa['passed'] and qa['manifest_sha256']==sha(report/'manifest.json')
    assert visual['passed'] and visual['actually_viewed'] and visual['numeric_qa_sha256']==sha(report/'qa_numeric.json')
    for row in visual['images']:assert sha(report/row['file'])==row['sha256']
    for n,h in manifest['outputs_sha256'].items():assert sha(report/n)==h
    files.update(BASE+'/report_v1/'+n for n in [*manifest['outputs_sha256'],'manifest.json','qa_numeric.json','visual_qa.json'])
    for name in ['preflight_v1','development_v1','audit_v1','replay_v1']:
        assert read(root/BASE/name/'summary.json')['passed']
    for p,h in protocol['source_sha256'].items():assert sha(root/p)==h
    repair=protocol['serialization_repair']
    assert sha(root/repair['adapter_file'])==repair['adapter_sha256']
    assert sha(root/repair['failure_file'])==repair['failure_sha256']
    return sorted(files)


def build(root):
    entries=[]
    for name in paths(root):
        p=root/name;assert p.is_file() and p.resolve().is_relative_to(root.resolve())
        assert p.stat().st_size<90*2**20
        if p.suffix=='.py':ast.parse(p.read_text(encoding='utf-8'),filename=name)
        entries.append(dict(path=name,sha256=sha(p),bytes=p.stat().st_size))
    value=dict(parent_commit=git(root,'rev-parse','HEAD').decode().strip(),file_count=len(entries),
        bytes=sum(e['bytes'] for e in entries),files=entries,full_saved_input_component_outputs=True,
        parent_trajectory_archives_included=False,old_raw_geometry_included=False,
        end_to_end_scientific_reproduction=False)
    with (root/MANIFEST).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False)
    return verify(root)


def verify(root):
    value=read(root/MANIFEST);assert [e['path'] for e in value['files']]==paths(root)
    assert len(value['files'])==value['file_count']
    assert sum(e['bytes'] for e in value['files'])==value['bytes']
    for e in value['files']:assert sha(root/e['path'])==e['sha256'] and (root/e['path']).stat().st_size==e['bytes']
    return dict(passed=True,files=value['file_count'],bytes=value['bytes'])


def stage(root):
    result=verify(root);value=read(root/MANIFEST)
    assert git(root,'rev-parse','HEAD').decode().strip()==value['parent_commit']
    assert not git(root,'diff','--cached','--name-only','-z')
    names=[e['path'] for e in value['files']]+[MANIFEST]
    tracked=set(git(root,'ls-files','-z').decode().strip('\0').split('\0'))
    dirty=set(git(root,'diff','HEAD','--name-only','-z').decode().strip('\0').split('\0'))
    expected={n for n in names if n not in tracked or n in dirty}
    subprocess.run(['git','add','-f','--pathspec-from-file=-','--pathspec-file-nul'],cwd=root,
        input=('\0'.join(names)+'\0').encode(),check=True)
    staged=set(git(root,'diff','--cached','--name-only','-z').decode().strip('\0').split('\0'))
    assert staged==expected
    index={}
    for record in git(root,'ls-files','--stage','-z').decode().strip('\0').split('\0'):
        header,name=record.split('\t',1);mode,oid,stage=header.split();assert stage=='0';index[name]=oid
    for name in names:
        payload=(root/name).read_bytes()
        assert index[name]==hashlib.sha1(b'blob '+str(len(payload)).encode()+b'\0'+payload).hexdigest()
    return dict(**result,staged_exact_files=len(staged),already_tracked_unchanged=len(names)-len(staged))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['build','verify','stage']);args=ap.parse_args()
    print(dict(action=args.action,**globals()[args.action](Path(__file__).resolve().parents[1])),flush=True)
