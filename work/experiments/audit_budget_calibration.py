"""Independent budget inclusion reconstruction and artifact/input audit."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
from run_independent_hybrid_memory import observations
from run_multiplier_fixed_point_screen import sha,dump

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();src=Path(__file__).parent;inp=root/'results/matched_budget_confirmation/calibration';out=root/'results/matched_budget_confirmation/calibration_audit';out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    p=json.loads((inp/'protocol.json').read_text());summary=json.loads((inp/'summary.json').read_text());assert summary['passed'];hashes=dict(p['source_sha256']);hashes[Path(__file__).name]=sha(Path(__file__))
    for n,h in hashes.items():assert sha(src/n)==h,n
    for name in ['protocol','timings','memory','selected_configs']:assert sha(inp/f'{name}.json')==summary[f'{name}_sha256']
    dump(out/'protocol.json',dict(source_sha256=hashes,calibration_summary_sha256=sha(inp/'summary.json'),query_targets_accessed=False));counts=Counter();rows=json.loads((inp/'timings.json').read_text());memory=json.loads((inp/'memory.json').read_text());selection=json.loads((inp/'selected_configs.json').read_text());cfgs=p['configs']
    assert len(rows)==400 and len(memory)==50;assert len({(r['seed'],r['method']) for r in rows})==400
    for row in rows:
        assert sha(inp/row['file'])==row['sha256'];x,v=observations(row['seed'])
        with np.load(inp/row['file']) as a:
            assert a['x_observed'].tobytes()==x[:4].tobytes() and a['v_observed'].tobytes()==v[:4].tobytes();assert a['q_observed'].tobytes()==np.linspace(0,1,257).tobytes()
            for k in a.files:
                if np.issubdtype(a[k].dtype,np.number):assert np.isfinite(a[k]).all()
        assert row['seconds']==row['metadata']['charged_complete_seconds'] and row['seconds']>0;counts['input_and_array_files']+=1
    means={c['name']:sum(r['seconds'] for r in rows if r['method']==c['name'])/16 for c in cfgs};cap=means[p['primary']]*1.10;assert abs(cap-selection['budget_seconds'])<1e-14
    for n,m in means.items():assert abs(m-selection['calibration_means'][n])<1e-14;assert abs(m-summary['methods'][n]['mean_seconds'])<1e-14;counts['method_means']+=1
    # New configurations carry an explicit group; frozen old ones do not.
    extras=[c for c in cfgs if c['family'] in ['warm_plus','cold_plus','prior_plus']];eligible=[c for c in extras if means[c['name']]<=cap]
    assert [c['name'] for c in eligible]==selection['within_budget_extras'];over=[]
    for group in sorted({c['group'] for c in extras}):
        groupconfigs=[c for c in extras if c['group']==group]
        if not any(c in eligible for c in groupconfigs):over.append(min(groupconfigs,key=lambda c:(means[c['name']],c['name'])))
    assert [c['name'] for c in over]==selection['over_budget_background_extras'];assert selection['configs'][36:]==eligible+over
    old=json.loads((root/'results/minimum_dual_query/resources/protocol.json').read_text())['configs'];assert selection['configs'][:36]==old
    for row in memory:
        assert row['traced_peak_bytes']>=row['traced_current_bytes']>=0;counts['memory_rows']+=1
    ans=dict(passed=True,counts=counts,selected_configs=len(selection['configs']),eligible_new=len(eligible),above_budget_background_new=len(over),
        protocol_sha256=sha(out/'protocol.json'),query_targets_accessed=False)
    dump(out/'summary.json',ans);print(json.dumps(ans),flush=True)

if __name__=='__main__':main()
