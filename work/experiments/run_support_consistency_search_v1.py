"""324 all 128 sealed states x six chain selector credits before geometry."""
import argparse
from collections import Counter
from pathlib import Path
import time
import traceback
import numpy as np
import scipy.optimize as opt
import batched_bp_discovery as bp
import branch_image_chain_v1 as chain
from diagnose_branch_image_chain_v1 import conflicts
from evaluate_complete_credit_mode_geometry_v1 import read,save,sha

CHANNELS=['dual','dual_plus_residual','residual','bp','random_sign','zero']


def run(root,out):
    begin=time.perf_counter();source=root/'results/support_consistency_trigger/states_v1'
    summary=read(source/'summary.json');manifest=read(source/'before_search_manifest.json');assert summary['passed']
    assert sha(source/'before_search_manifest.json')==summary['manifest_sha256']
    for name,digest in manifest['files_sha256'].items():assert sha(source/name)==digest
    tested=read(root/'results/branch_image_chain/tests_v1/summary.json');assert tested['passed']
    for name,digest in tested['source_sha256'].items():assert sha(root/'work/experiments'/name)==digest
    protocol=dict(source_sha256={n:sha(root/'work/experiments'/n) for n in [Path(__file__).name,'branch_image_chain_v1.py','factorized_dual_branch_search_v1.py']},
                  design_sha256=sha(root/'outputs/ttt-pc-alm-research/324_support_consistency_trigger_protocol_v1.md'),
                  input_summary_sha256=sha(source/'summary.json'),input_manifest_sha256=summary['manifest_sha256'],
                  tasks=64,states=128,channels=CHANNELS,policies=['first_fit','uniform_state'],k=8,
                  max_modes_per_state_channel=24,cached_prefix_not_free=True,query_targets_accessed=False,
                  geometry_accessed=False,resources_matched=False,bp_credit_scope='Explicit bp control only.')
    save(out/'protocol.json',protocol);rows=[];files={};counts=Counter();aggregate={}
    previous=[]
    def forbid(*args,**kwargs):raise AssertionError('Global BP/LP/optimizer entered selector')
    try:
        for obj,name in [(bp,'evaluate'),(bp,'refine'),(opt,'linprog'),(opt,'minimize')]:
            previous.append((obj,name,getattr(obj,name)));setattr(obj,name,forbid)
        for row in read(source/'rows.json'):
            old=read(source/row['file']);x=np.array(old['x_observed']);v=np.array(old['v_observed'])
            original=np.frombuffer(bytes.fromhex(old['original_mode']),dtype=np.uint8).reshape(4,4);results={}
            for name in CHANNELS:
                result=chain.propose(x,v,np.array(old['credits'][name]),original,k=8)
                for p in result['proposals']:
                    pattern=np.frombuffer(bytes.fromhex(p['mode']),dtype=np.uint8).reshape(4,4)
                    assert not conflicts(x,v,pattern);counts['structural_proposal_checks']+=1
                if old['support_fit']:assert not result['current_certified_infeasible']
                results[name]=result;group=aggregate.setdefault(old['policy']+'/'+name,Counter());group['cases']+=1
                group['current_certified_infeasible']+=int(result['current_certified_infeasible']);group['proposals']+=len(result['proposals'])
                for field in ['layer_rows_evaluated','pair_outputs','table_seconds','selection_seconds','total_seconds']:
                    group[field]+=result['meta'][field]
            record=dict(old,results=results,selected_state_sha256=row['sha256'])
            save(out/row['file'],record);files[row['file']]=sha(out/row['file'])
            rows.append(dict(seed=old['seed'],policy=old['policy'],file=row['file'],sha256=files[row['file']]))
            counts['states']+=1;counts['channels']+=len(CHANNELS)
            if counts['states']%4==0:print(dict(states=counts['states'],channels=counts['channels'],seconds=time.perf_counter()-begin),flush=True)
    finally:
        for obj,name,func in previous:setattr(obj,name,func)
    assert counts['states']==128 and counts['channels']==768
    for n,digest in protocol['source_sha256'].items():assert sha(root/'work/experiments'/n)==digest
    for name,obj in [('rows.json',rows),('aggregate.json',aggregate)]:save(out/name,obj);files[name]=sha(out/name)
    files['protocol.json']=sha(out/'protocol.json')
    save(out/'before_geometry_manifest.json',dict(files_sha256=files,all_proposals_sealed=True,query_targets_accessed=False,geometry_accessed=False))
    result=dict(passed=True,counts=dict(counts),aggregate=aggregate,seconds=time.perf_counter()-begin,
                manifest_sha256=sha(out/'before_geometry_manifest.json'),runtime_no_global_bp_guard_passed=True,
                query_targets_accessed=False,resources_matched=False,independent_task_gain_established=False)
    save(out/'summary.json',result);print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    try:run(Path(__file__).resolve().parents[2],a.out)
    except Exception:save(a.out/'failure.json',dict(traceback=traceback.format_exc()));raise
