"""Close the shared numerical issue BEFORE opening either set of query risks."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
from run_multiplier_fixed_point_screen import sha,dump

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();src=Path(__file__).parent;base=root/'results/matched_budget_confirmation';out=base/'geometry_disposition.json'
    assert not out.exists() and not (base/'evaluation/protocol.json').exists() and not (base/'joint_evaluation/protocol.json').exists()
    required={};hashes={}
    for directory in ['confirmation','prediction_audit','geometry_diagnosis','geometry_determinant_audit','geometry_preflight','conditioned_confirmation','conditioned_prediction_audit','mode_identity']:
        summary=json.loads((base/directory/'summary.json').read_text());assert summary['passed'] and not summary['query_targets_accessed'];required[f'{directory}/summary.json']=sha(base/directory/'summary.json');p=json.loads((base/directory/'protocol.json').read_text());required[f'{directory}/protocol.json']=sha(base/directory/'protocol.json')
        for n,h in p['source_sha256'].items():
            if n in hashes:assert hashes[n]==h,n
            hashes[n]=h
    for directory in ['confirmation','conditioned_confirmation']:required[f'{directory}/before_query_manifest.json']=sha(base/directory/'before_query_manifest.json')
    for n in [Path(__file__).name,'evaluate_joint_budget_confirmation.py','audit_joint_budget_evaluation.py','plot_joint_budget_confirmation.py']:hashes[n]=sha(src/n)
    for n,h in hashes.items():assert sha(src/n)==h,n
    original=json.loads((base/'confirmation/summary.json').read_text());new=json.loads((base/'conditioned_confirmation/summary.json').read_text());assert original['predictions']==new['predictions']==2944 and original['failures']==32
    assert new['unchanged_successes']==2912 and new['repaired_original_failures']==32 and new['failures']==0
    ans=dict(may_open_query_answers=True,numerical_failures_reviewed=True,query_targets_accessed=False,created_utc=datetime.now(timezone.utc).isoformat(),
        required_artifacts=required,source_sha256=hashes,original_failures=32,corrected_failures=0,unchanged_successes=2912,
        disposition='original frozen failure-policy results AND observation-only universal geometry correction are both retained; no selective implementation choice; same64 tasks, not128; no method, prior, information or parameter-range change',
        attribution='common numerical readout correction is not a PC-ALM contribution; independent task gains require same-readout optimizer comparisons and real full costs',
        next='one joint query evaluation, independent risk audit and complete report')
    dump(out,ans);print(json.dumps({k:v for k,v in ans.items() if k not in ['required_artifacts','source_sha256']}),flush=True)

if __name__=='__main__':main()
