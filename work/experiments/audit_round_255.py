"""Immutable handoff audit for251-255;256 remains an unexecuted protocol."""
import argparse
import json
from pathlib import Path
import re
from run_multiplier_fixed_point_screen import sha,dump

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();src=Path(__file__).parent
    out=root/'results/round_255_audit.json';assert not out.exists();base=root/'results/local_dual_jump';parent=json.loads((root/'results/round_250_audit.json').read_text())
    hashes=dict(parent['source_sha256']);summaries={}
    for directory in ['screen_v2','audit','transfer','transfer_audit','transfer_analysis','credit_proof','continuation','continuation_audit','continuation_analysis','figures','mode_audit']:
        path=base/directory/'summary.json';data=json.loads(path.read_text());assert data['passed'];summaries[directory]=sha(path)
        if 'query_targets_accessed' in data:assert not data['query_targets_accessed']
        protocol=base/directory/'protocol.json'
        if protocol.exists():
            p=json.loads(protocol.read_text());hashes.update(p.get('source_sha256',{}))
            if 'protocol_sha256' in data:assert sha(protocol)==data['protocol_sha256']
    extras=['analyze_local_dual_jump_transfer.py','prove_local_dual_parameter_credit.py','analyze_local_dual_jump_continuation.py','plot_local_dual_jump_results.py','audit_local_dual_jump_modes.py',Path(__file__).name]
    for name in extras:hashes[name]=sha(src/name)
    for name,h in hashes.items():assert sha(src/name)==h,name
    docs=root/'outputs/ttt-pc-alm-research';reports=dict(parent['report_sha256'])
    for name,h in reports.items():assert sha(docs/name)==h,name
    extra_docs=['252_local_dual_jump_transfer_protocol.md','253_local_dual_jump_transfer_results.md','254_local_dual_jump_continuation_protocol.md','255_local_dual_jump_continuation_results.md','256_local_dual_jump_query_protocol.md']
    links=0
    for name in extra_docs:
        path=docs/name;reports[name]=sha(path)
        for target in re.findall(r'\]\(([^)]+)\)',path.read_text(encoding='utf-8')):
            if '://' in target:continue
            assert (path.parent/target).resolve().exists(),(name,target);links+=1
    assert reports['252_local_dual_jump_transfer_protocol.md']==json.loads((base/'transfer/protocol.json').read_text())['design_sha256']
    assert reports['254_local_dual_jump_continuation_protocol.md']==json.loads((base/'continuation/protocol.json').read_text())['design_sha256']
    figure_manifest=json.loads((base/'figures/summary.json').read_text())
    for name,h in figure_manifest['figures'].items():assert sha(base/'figures'/name)==h,name
    rows=0
    for directory in ['transfer','continuation']:
        for row in json.loads((base/directory/'support_rows.json').read_text()):assert sha(base/directory/row['file'])==row['sha256'];rows+=1
    result=dict(passed=True,source_sha256=hashes,report_sha256=reports,source_count=len(hashes),parent_audit_sha256=sha(root/'results/round_250_audit.json'),
        summary_sha256=summaries,linked_local_targets=links,artifact_method_states=rows,figures=figure_manifest['figures'],
        scope='all support-only251-255 results verified; no query targets;256 frozen but not executed; core goal active',
        failed_attempt_preserved='local_dual_jump/screen: original NumPy integer JSON serialization failure; screen_v2 calculation unchanged')
    dump(out,result);print(json.dumps({k:v for k,v in result.items() if k not in ['source_sha256','report_sha256','summary_sha256']}),flush=True)

if __name__=='__main__':main()
