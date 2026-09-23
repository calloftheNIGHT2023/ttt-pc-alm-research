"""Freeze complementary-information diagnosis and the next implementation preflight."""
import argparse,json,re
from collections import Counter
from pathlib import Path
from run_multiplier_fixed_point_screen import sha,dump


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();src=Path(__file__).parent
    out=root/'results/round_270_audit.json';assert not out.exists();parent=root/'results/round_268_audit.json';old=json.loads(parent.read_text());assert old['passed']
    hashes=dict(old['source_sha256']);reports=dict(old['report_sha256']);checks=Counter();summaries={}
    dirs=['confirmation_conditional_risk/complementarity','confirmation_conditional_risk/complementarity_audit','confirmation_conditional_risk/complementarity_figures','fixed_restart_credit/primitive_v2']
    for name in dirs:
        directory=root/'results'/name;s=json.loads((directory/'summary.json').read_text());assert s['passed'];p=json.loads((directory/'protocol.json').read_text());summaries[name]=sha(directory/'summary.json')
        for n,h in p['source_sha256'].items():
            if n in hashes:assert hashes[n]==h,n
            hashes[n]=h
        for n,h in s.get('outputs_sha256',{}).items():assert sha(directory/n)==h;checks['output_manifests']+=1
        for key in ['protocol','rows']:
            if key+'_sha256' in s:assert sha(directory/(key+'.json'))==s[key+'_sha256']
    hashes['verify_mixed_restart_credit.py']=sha(src/'verify_mixed_restart_credit.py');hashes[Path(__file__).name]=sha(Path(__file__))
    for n,h in hashes.items():assert sha(src/n)==h,n
    comp=root/'results/confirmation_conditional_risk/complementarity'
    for n,h in json.loads((comp/'files.json').read_text()).items():assert sha(comp/n)==h;checks['union_curve_files']+=1
    pre=root/'results/fixed_restart_credit/primitive_v2'
    for r in json.loads((pre/'rows.json').read_text()):assert sha(pre/r['file'])==r['sha256'];checks['preflight_predictors']+=1
    assert checks['union_curve_files']==64 and checks['preflight_predictors']==32
    pp=json.loads((pre/'protocol.json').read_text());failed=root/'results/fixed_restart_credit/primitive/protocol.json';assert sha(failed)==pp['failed_attempt_protocol_sha256']
    for n,h in json.loads(failed.read_text())['source_sha256'].items():assert sha(src/n)==h
    docs=root/'outputs/ttt-pc-alm-research'
    for n,h in reports.items():assert sha(docs/n)==h,n
    for n in ['270_complementary_mode_results.md','271_fixed_restart_credit_diversity_protocol.md']:
        path=docs/n;reports[n]=sha(path)
        for target in re.findall(r'\]\(([^)]+)\)',path.read_text(encoding='utf-8')):
            if '://' not in target:assert (path.parent/target).resolve().exists(),target;checks['local_report_links']+=1
    assert reports['271_fixed_restart_credit_diversity_protocol.md']==pp['design_sha256']
    counts=json.loads((root/'results/confirmation_conditional_risk/complementarity_audit/summary.json').read_text())['counts']
    checks['union_curves']=counts['union_curves'];checks['pair_metric_replays']=counts['pair_metric_replays'];checks['paired_intervals']=counts['paired_intervals']
    prechecks=json.loads((pre/'summary.json').read_text())['counts'];assert prechecks['pure_frozen_arrays']==108 and prechecks['selected_trajectory_arrays']==1632
    ans=dict(passed=True,parent_audit_sha256=sha(parent),source_count=len(hashes),source_sha256=hashes,report_sha256=reports,summary_sha256=summaries,checks=checks,
        failed_preflight_protocol_sha256=sha(failed),failed_preflight_note_sha256=sha(failed.with_name('ATTEMPT.md')),
        scope='Complementarity is diagnostic, not free online union; only four-context implementation preflight is complete for271, not its full development evaluation',
        next='Continue fixed_restart_credit/development process, then independent all64 audit before development risks; goal active')
    dump(out,ans);print(json.dumps({k:v for k,v in ans.items() if k not in ['source_sha256','report_sha256','summary_sha256']}),flush=True)


if __name__=='__main__':main()
