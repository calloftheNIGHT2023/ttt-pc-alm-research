"""321 all-channel chain proposals from sealed 320 inputs, not its outcomes."""
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
    begin=time.perf_counter();source=root/'results/factorized_dual_branch_search/development_v1'
    summary=read(source/'summary.json');manifest=read(source/'before_geometry_manifest.json');assert summary['passed']
    assert sha(source/'before_geometry_manifest.json')==summary['manifest_sha256']
    for name,digest in manifest['files_sha256'].items():assert sha(source/name)==digest
    tested=read(root/'results/branch_image_chain/tests_v1/summary.json');assert tested['passed']
    for name,digest in tested['source_sha256'].items():assert sha(root/'work/experiments'/name)==digest
    names=[Path(__file__).name,'branch_image_chain_v1.py','factorized_dual_branch_search_v1.py','diagnose_branch_image_chain_v1.py',
           'evaluate_complete_credit_mode_geometry_v1.py']
    docs=['321_branch_image_chain_design_v1.md','321_execution_details_v1.md']
    protocol=dict(source_sha256={n:sha(root/'work/experiments'/n) for n in names},
                  design_sha256={n:sha(root/'outputs/ttt-pc-alm-research'/n) for n in docs},
                  input_summary_sha256=sha(source/'summary.json'),input_manifest_sha256=summary['manifest_sha256'],
                  tests_sha256=sha(root/'results/branch_image_chain/tests_v1/summary.json'),
                  tasks=64,states=131,channels=CHANNELS,k=8,max_modes_per_state_channel=24,
                  empty_tasks=read(source/'protocol.json')['empty_tasks'],cached_prefix_not_free=True,
                  query_targets_accessed=False,geometry_accessed=False,resources_matched=False,
                  bp_credit_scope='Only the bp control consumes its independently saved BP array.')
    save(out/'protocol.json',protocol);rows=[];files={};counts=Counter();aggregate={n:Counter() for n in CHANNELS}
    previous=[]
    def forbid(*args,**kwargs):raise AssertionError('Global BP or optimizer entered chain selector')
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
                results[name]=result;aggregate[name]['cases']+=1
                aggregate[name]['current_certified_infeasible']+=int(result['current_certified_infeasible'])
                aggregate[name]['current_structurally_infeasible']+=int(result['current_structurally_infeasible'])
                aggregate[name]['proposals']+=len(result['proposals'])
                for field in ['layer_rows_evaluated','pair_outputs','table_seconds','selection_seconds','total_seconds']:
                    aggregate[name][field]+=result['meta'][field]
            record={n:old[n] for n in ['seed','location','x_observed','v_observed','b','h','u','original_mode','credits']}
            record.update(source_file=row['file'],source_sha256=row['sha256'],results=results)
            save(out/row['file'],record);files[row['file']]=sha(out/row['file'])
            rows.append(dict(seed=old['seed'],location=old['location'],file=row['file'],sha256=files[row['file']]))
            counts['states']+=1;counts['channels']+=len(CHANNELS)
            if counts['states']%5==0:print(dict(states=counts['states'],channels=counts['channels'],seconds=time.perf_counter()-begin),flush=True)
    finally:
        for obj,name,func in previous:setattr(obj,name,func)
    assert counts['states']==131 and counts['channels']==786
    for n,digest in protocol['source_sha256'].items():assert sha(root/'work/experiments'/n)==digest
    for n,digest in protocol['design_sha256'].items():assert sha(root/'outputs/ttt-pc-alm-research'/n)==digest
    for name,obj in [('rows.json',rows),('aggregate.json',aggregate)]:save(out/name,obj);files[name]=sha(out/name)
    files['protocol.json']=sha(out/'protocol.json')
    save(out/'before_geometry_manifest.json',dict(files_sha256=files,all_proposals_sealed=True,query_targets_accessed=False,geometry_accessed=False))
    result=dict(passed=True,counts=dict(counts),aggregate=aggregate,seconds=time.perf_counter()-begin,
                manifest_sha256=sha(out/'before_geometry_manifest.json'),runtime_no_global_bp_guard_passed=True,
                query_targets_accessed=False,resources_matched=False,independent_task_gain_established=False)
    save(out/'summary.json',result);print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    try:run(Path(__file__).resolve().parents[2],args.out)
    except Exception:save(args.out/'failure.json',dict(traceback=traceback.format_exc()));raise
