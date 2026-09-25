"""324 independent reverse bounds and exact best-first ordering on every call."""
import argparse
from collections import Counter
from fractions import Fraction as F
from pathlib import Path
import time
import traceback
import numpy as np
from independent_branch_image_search_v1 import IndependentSearch
from evaluate_complete_credit_mode_geometry_v1 import read,save,sha

CHANNELS=['dual','dual_plus_residual','residual','bp','random_sign','zero']


def run(root,out):
    started=time.perf_counter();folder=root/'results/support_consistency_trigger/development_v1';states=folder.parent/'states_v1'
    summary=read(folder/'summary.json');assert summary['passed'];manifest=read(folder/'before_geometry_manifest.json')
    assert sha(folder/'before_geometry_manifest.json')==summary['manifest_sha256']
    for n,digest in manifest['files_sha256'].items():assert sha(folder/n)==digest
    st=read(folder.parent/'states_audit_v1/summary.json');assert st['passed'] and st['input_summary_sha256']==sha(states/'summary.json')
    tests=read(root/'results/branch_image_chain/independent_tests_v1/summary.json');assert tests['passed']
    for n,digest in tests['source_sha256'].items():assert sha(root/'work/experiments'/n)==digest
    save(out/'protocol.json',dict(input_summary_sha256=sha(folder/'summary.json'),trigger_audit_sha256=sha(folder.parent/'states_audit_v1/summary.json'),
         source_sha256={n:sha(root/'work/experiments'/n) for n in [Path(__file__).name,'independent_branch_image_search_v1.py']},query_targets_accessed=False))
    counts=Counter();rows=[];aggregate={}
    for index,row in enumerate(read(folder/'rows.json')):
        rec=read(folder/row['file']);old=read(states/row['file']);assert rec['selected_state_sha256']==sha(states/row['file'])
        assert all(rec[k]==v for k,v in old.items());counts['source_states']+=1
        x=np.array(rec['x_observed']);v=np.array(rec['v_observed']);pattern=np.array(list(bytes.fromhex(rec['original_mode']))).reshape(4,4)
        details={}
        for name in CHANNELS:
            checker=IndependentSearch(x,v,np.array(rec['credits'][name]),pattern);result=rec['results'][name]
            best={m:checker.best(0,-1,m) for m in range(17)}
            assert {str(m):str(p[0]) for m,p in best.items() if p is not None}==result['shell_minima'];counts['shell_minima']+=len(result['shell_minima'])
            current=checker.fixed(pattern);assert result['current_lower']==(None if current is None else str(current))
            assert result['current_structurally_infeasible']==(current is None)
            excluded=current is None or current>0;assert result['current_certified_infeasible']==excluded
            if rec['support_fit']:assert current is not None and current<=0;counts['feasible_current_credit_bounds']+=1
            allowed=[m for m,p in best.items() if m>=1 and p is not None and p[0]<=0];minimum=min(allowed) if allowed else None
            assert result['minimum_nonexcluded_hamming']==minimum
            assert result['necessary_hamming_lower_bound']==(minimum if excluded else None)
            expected=[]
            if minimum is not None:
                for m in range(minimum,min(minimum+2,16)+1):
                    pool=checker.kbest(m,k=8);counts['independent_kbest_shells']+=1
                    for rank,(value,path) in enumerate(pool):
                        if value<=0:expected.append(dict(mode=bytes(path).hex(),hamming=m,rank=rank,lower=str(value)))
            assert expected==result['proposals'];counts['independent_kbest_proposals']+=len(expected)
            for p in expected:
                raw=bytes.fromhex(p['mode']);pp=[raw[j*4:(j+1)*4] for j in range(4)]
                assert checker.fixed(pp)==F(p['lower']);counts['fixed_proposal_bounds']+=1
            group=aggregate.setdefault(rec['policy']+'/'+name,Counter());group['cases']+=1;group['proposals']+=len(expected);group['current_certified_infeasible']+=int(excluded)
            details[name]=dict(proposals=len(expected),best_first_expansions=checker.expansions)
            counts['best_first_expansions']+=checker.expansions;checker.clear();checker.best=None
        rows.append(dict(seed=rec['seed'],policy=rec['policy'],checks=details))
        if (index+1)%8==0:print(dict(audited_states=index+1,seconds=time.perf_counter()-started),flush=True)
    assert len(rows)==128
    for name,group in aggregate.items():
        for field,value in group.items():assert summary['aggregate'][name][field]==value;counts['aggregate_fields']+=1
    save(out/'rows.json',rows);result=dict(passed=True,counts=dict(counts),seconds=time.perf_counter()-started,
        input_summary_sha256=sha(folder/'summary.json'),protocol_sha256=sha(out/'protocol.json'),rows_sha256=sha(out/'rows.json'),
        query_targets_accessed=False,resources_matched=False,independent_task_gain_established=False)
    save(out/'summary.json',result);print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    try:run(Path(__file__).resolve().parents[2],a.out)
    except Exception:save(a.out/'failure.json',dict(traceback=traceback.format_exc()));raise
