"""315 exact selected-formula census on every sealed 313 state."""
import argparse
from collections import Counter,defaultdict
import gzip
import json
from pathlib import Path
import time
import traceback
import numpy as np
from effective_affine_map_v1 import build,canonical,dump,evaluate,pack,digest
from solver_policy_trace_v1 import segments
from evaluate_complete_credit_mode_geometry_v1 import read,save,sha


def run(root,out):
    begin=time.perf_counter();source=root/'results/solver_policy_recurrence/development_v1'
    summary=read(source/'summary.json');assert summary['passed']
    for n,dig in summary['outputs_sha256'].items():assert sha(source/n)==dig
    tests=root/'results/effective_affine_map/tests_v1/summary.json';tested=read(tests)
    assert tested['passed'] and not tested['real_task_data_accessed']
    names={Path(__file__).name,'evaluate_complete_credit_mode_geometry_v1.py'}
    for p in [read(source/'protocol.json'),tested]:
        for n,dig in p['source_sha256'].items():assert sha(root/'work/experiments'/n)==dig;names.add(n)
    docs=root/'outputs/ttt-pc-alm-research'
    save(out/'protocol.json',dict(source_sha256={n:sha(root/'work/experiments'/n) for n in sorted(names)},
        support_summary_sha256=sha(source/'summary.json'),tests_summary_sha256=sha(tests),
        design_sha256={n:sha(docs/n) for n in ['315_effective_affine_map_design.md','315_effective_map_execution_protocol_v1.md']},
        tasks=64,states=131,trajectories=393,steps=25152,variable_order=['b','h_layer_major','u_layer_major'],
        bound=.12,trust=.01,eps=.001,float_gap_threshold=1e-10,query_targets_accessed=False,
        success_event_labels_accessed=False,resources_matched=False,new_candidates_generated=False,
        exact_arithmetic_scope='Canonical identity of executed formulas, not their optimality or future guard validity.'))
    schema=read(source/'policy_schema.json');byseed=defaultdict(list)
    for row in read(source/'rows.json'):byseed[row['seed']].append(row)
    files={};counts=Counter();taskrows=[];trajectories=[];allgaps=[]
    original={(t['seed'],t['family'],*t['location']):t for t in read(source/'trajectories.json')}
    for seed,rows in sorted(byseed.items()):
        policy_cache={};bycanonical={};maps=[];map_digests=[];seconds=Counter();subids=[{}, {}, {}]
        for row in rows:
            with np.load(source/row['file'],allow_pickle=False) as z:
                x,v=z['x_observed'],z['v_observed'];d=z['b'].shape[-1];n=len(x);r=len(z['locations'])
                cuts=[0,d,d+d*n,d+2*d*n]
                full=np.concatenate([z['policy_'+g] for g in schema],axis=2)
                ids=np.empty((64,r),dtype=np.int32);part_ids=np.empty((64,r,3),dtype=np.int32)
                gaps=np.empty((64,r,3));born_before=len(maps);built=0
                for t in range(64):
                    for i in range(r):
                        key=(row['method'],full[t,i].tobytes())
                        if key in policy_cache:
                            index=policy_cache[key];counts['cached_full_policy_calls']+=1
                        else:
                            tick=time.perf_counter()
                            builtmap=build({g:z['policy_'+g][t,i] for g in schema},schema,x,v,row['method'])
                            seconds['construction']+=time.perf_counter()-tick
                            code=canonical(builtmap);index=bycanonical.get(code)
                            if index is None:
                                index=len(maps);bycanonical[code]=index;maps.append(builtmap);map_digests.append(digest(builtmap))
                            policy_cache[key]=index;built+=1;counts['constructed_full_policy_maps']+=1
                        ids[t,i]=index;exactmap=maps[index]
                        for part in range(3):
                            code=canonical(exactmap[cuts[part]:cuts[part+1]])
                            if code not in subids[part]:subids[part][code]=len(subids[part])
                            part_ids[t,i,part]=subids[part][code]
                        tick=time.perf_counter()
                        expected=evaluate(exactmap,pack(z['b'][t,i],z['h'][t,:,i],z['u'][t,:,i]))
                        seconds['exact_evaluation']+=time.perf_counter()-tick
                        actual=np.r_[z['b'][t+1,i],z['h'][t+1,:,i].ravel(),z['u'][t+1,:,i].ravel()]
                        err=abs(np.array(expected,dtype=float)-actual)
                        gaps[t,i]=[float(np.max(err[cuts[j]:cuts[j+1]])) for j in range(3)]
                        counts['actual_state_calls']+=1
                        if max(gaps[t,i])>1e-10:
                            counts['value_discrepancies']+=1
                            allgaps.append(dict(seed=seed,family=row['family'],location=z['locations'][i].tolist(),step=t+1,gaps=gaps[t,i].tolist()))
                filename=f"{seed}_{row['family']}_maps.npz"
                np.savez_compressed(out/filename,map_ids=ids,block_map_ids=part_ids,gaps=gaps,locations=z['locations'])
                files[filename]=sha(out/filename)
                for i,loc in enumerate(z['locations']):
                    old=original[(seed,row['family'],*loc.tolist())]
                    fullchanged=np.any(full[1:,i]!=full[:-1,i],axis=1);mapchanged=ids[1:,i]!=ids[:-1,i]
                    assert not np.any(mapchanged&~fullchanged)
                    trajectories.append(dict(seed=seed,family=row['family'],location=loc.tolist(),
                        map_segments=segments(ids[:,i,None]),old_full_segments=old['full_segments'],
                        block_segments={k:segments(part_ids[:,i,j,None]) for j,k in enumerate(['b','h','u'])},
                        old_policy_changes_merged=int(np.sum(fullchanged&~mapchanged)),max_gaps=np.max(gaps[:,i],axis=0).tolist()))
                taskrows.append(dict(seed=seed,family=row['family'],file=filename,sha256=files[filename],states=r,
                    map_bank_file=f'{seed}_map_bank.json.gz',constructed_policies=built,new_unique_maps=len(maps)-born_before))
        # Exact tuple equality created map IDs; digest is a secondary integrity label.
        assert len(set(map_digests))==len(maps)
        payload=dict(seed=seed,dimension=36,variable_order=['b','h','u'],
            maps=[dict(id=i,sha256=map_digests[i],rows=dump(m)) for i,m in enumerate(maps)],
            support_x=x.tolist(),support_v=v.tolist(),seconds=dict(seconds),
            full_policy_cache_entries=len(policy_cache),unique_effective_maps=len(maps),
            unique_block_maps=[len(a) for a in subids],
            nonzero_rational_coefficients=sum(len(a.terms) for m in maps for a in m),
            resource_scope='Offline cache counts and archive size; not live/peak online memory.')
        serialized=json.dumps(payload,separators=(',',':'),ensure_ascii=False).encode()
        filename=f'{seed}_map_bank.json.gz'
        with gzip.open(out/filename,'xb') as stream:stream.write(serialized)
        files[filename]=sha(out/filename)
        counts['unique_effective_maps']+=len(maps)
        counts['uncompressed_map_bank_bytes']+=len(serialized)
        counts['compressed_map_bank_bytes']+=(out/filename).stat().st_size
        print(dict(seed=seed,processed_calls=counts['actual_state_calls'],maps=len(maps),policies=len(policy_cache),
            value_discrepancies=counts['value_discrepancies'],seconds=dict(seconds)),flush=True)
    assert counts['actual_state_calls']==25152 and len(trajectories)==393
    for name,value in [('rows.json',taskrows),('trajectories.json',trajectories),('value_discrepancies.json',allgaps)]:
        save(out/name,value);files[name]=sha(out/name)
    files['protocol.json']=sha(out/'protocol.json')
    save(out/'before_event_analysis_manifest.json',dict(files_sha256=files,all_selected_states_processed=True,
        query_targets_accessed=False,success_event_labels_accessed=False))
    final=dict(passed=counts['value_discrepancies']==0,counts=dict(counts),trajectories=393,tasks=64,
        empty_tasks=summary['empty_tasks'],max_gaps=[max(t['max_gaps'][j] for t in trajectories) for j in range(3)],
        query_targets_accessed=False,resources_matched=False,independent_task_gain_established=False,
        seconds=time.perf_counter()-begin,manifest_sha256=sha(out/'before_event_analysis_manifest.json'),outputs_sha256=files)
    save(out/'summary.json',final);print({k:v for k,v in final.items() if k not in ['outputs_sha256','empty_tasks']},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    try:run(Path(__file__).resolve().parents[2],args.out)
    except Exception:
        save(args.out/'failure.json',dict(traceback=traceback.format_exc()));raise
