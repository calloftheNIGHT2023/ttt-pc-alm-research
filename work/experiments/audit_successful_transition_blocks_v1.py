"""310: scalar replay and common geometry for every factorial alternative."""
import argparse
from collections import Counter
from pathlib import Path
import time

import numpy as np
import cold_stagnation_switch as cold
import complete_credit_mode_geometry_v1 as geometry
from diagnose_observable_trap_continuation_v1 import new_local
from diagnose_gradient_flat_split_states_v1 import mode_list
from posterior_confirmation_pipeline import discovery_box
from evaluate_complete_credit_mode_geometry_v1 import read,save,sha,certify_label


def run(root,out):
    begin=time.perf_counter()
    folder=root/'results/successful_transition_census/block_replay_v1'
    summary=read(folder/'summary.json');assert summary['passed']
    for n,digest in summary['outputs_sha256'].items():assert sha(folder/n)==digest
    for n,digest in read(folder/'protocol.json')['source_sha256'].items():assert sha(root/'work/experiments'/n)==digest
    old=root/'results/complete_credit_amplitude_events/geometry_v1'
    old_summary=read(old/'summary.json');assert old_summary['execution_passed']
    for n,digest in old_summary['outputs_sha256'].items():assert sha(old/n)==digest
    source_names=set(read(folder/'protocol.json')['source_sha256'])|set(read(old/'protocol.json')['source_sha256'])|{Path(__file__).name}
    save(out/'protocol.json',dict(source_sha256={n:sha(root/'work/experiments'/n) for n in sorted(source_names)},
        input_summary_sha256=sha(folder/'summary.json'),old_geometry_sha256=sha(old/'summary.json'),
        query_targets_accessed=False,resources_matched=False,
        purpose='Classify all counterfactual outputs, so a different target is not silently counted as infeasible.'))
    cache={};counts=Counter();records=[]
    with discovery_box(.12):
        for row in read(folder/'events.json'):
            payload=read(folder/row['file']);event=payload['event'];seed=event['seed']
            if seed not in cache:cache[seed]=read(old/f'{seed}_geometry.json')
            data=cache[seed];x=np.array(payload['x_observed']);v=np.array(payload['v_observed'])
            old_positive=set(data['original_positive_modes']);result=[]
            for p in payload['proposals']:
                mask=p['mask'];chosen={name:np.array(payload['current' if bit=='1' else 'initial'][name])
                                      for name,bit in zip(['b','h','u'],mask)}
                state=new_local(chosen['b'][None],chosen['h'][:,None],chosen['u'][:,None],
                                np.array(payload['old_best'])[None],x,v,payload['method'])
                state.step()
                for name,actual in [('b',state.b[0]),('h',state.h[:,0]),('u_after',state.u[:,0])]:
                    assert np.array_equal(actual,p['float_'+name]),(seed,row['location'],mask,name)
                    counts['scalar_batch_array_replays']+=1
                mode=mode_list(state.b,x)[0];assert mode==p['float_mode']
                if mode not in data['geometry']:
                    data['geometry'][mode]=geometry.classify_mode(x,v,mode)
                    counts['new_geometry_classifications']+=1
                g=data['geometry'][mode]
                _,aa,rr,_,_=geometry.matrices(x,v,mode)
                counts['exact_certificate_rechecks']+=certify_label(aa,rr,g)
                counts['alternative_'+g['classification']]+=1
                result.append(dict(mask=mask,mode=mode,target_reached=mode==event['mode'],
                    classification=g['classification'],new_positive=g['positive_volume_certified'] and mode not in old_positive))
            successful=[r['mask'] for r in result if r['new_positive']]
            minimals=[m for m in successful if not any(s!=m and all(int(a)<=int(b) for a,b in zip(s,m)) for s in successful)]
            records.append(dict(event=event,alternatives=result,
                successful_new_positive_masks=successful,minimal_new_positive_block_sets=minimals))
    aggregate={}
    for family in ['alm_keep','alm_reset','nodual','pc']:
        rr=[r for r in records if r['event']['family']==family]
        aggregate[family]=dict(events=len(rr),
            all_current_only_events=sum(r['successful_new_positive_masks']==['111'] for r in rr),
            old_b_any_new_positive_events=sum(any(m[0]=='0' for m in r['successful_new_positive_masks']) for r in rr),
            minimal_new_positive_block_sets=dict(Counter(s for r in rr for s in r['minimal_new_positive_block_sets'])),
            alternative_new_positive_differs_from_target=sum(a['new_positive'] and not a['target_reached'] for r in rr for a in r['alternatives']))
    for n,val in [('events.json',records),('aggregate.json',aggregate),('geometry.json',cache)]:save(out/n,val)
    final=dict(passed=True,events=len(records),counts=dict(counts),aggregate=aggregate,
        query_targets_accessed=False,resources_matched=False,independent_task_gain_established=False,
        seconds=time.perf_counter()-begin,outputs_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file()})
    save(out/'summary.json',final);print({k:v for k,v in final.items() if k!='outputs_sha256'},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False);run(Path(__file__).resolve().parents[2],args.out)
