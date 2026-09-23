"""Whole-cohort replay, independent trigger chronology, regression arithmetic."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
import torch
import run_cold_stagnation_switch as run
from analyze_recovered_online_comparison import regression_replay,forward
from run_multiplier_fixed_point_screen import sha,dump

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();src=Path(__file__).parent
    base=root/'results/cold_stagnation_switch';inp=base/'development';out=base/'audit';out.mkdir(parents=True,exist_ok=True);assert not (out/'summary.json').exists()
    torch.set_num_threads(1);torch.set_num_interop_threads(1);p=json.loads((inp/'protocol.json').read_text());summary=json.loads((inp/'summary.json').read_text());assert summary['complete']
    for n,h in p['source_sha256'].items():assert sha(src/n)==h,n
    loaded,manifest=run.meta.load(root);assert p['checkpoint_manifest']=={n:manifest[n] for n in p['checkpoint_manifest']}
    before=json.loads((inp/'before_query_manifest.json').read_text())
    assert before['protocol_sha256']==sha(inp/'protocol.json') and before['support_rows_sha256']==sha(inp/'support_rows.json')
    for group in ['arrays','geometry']:
        for n,h in before[group].items():assert sha(inp/n)==h
    rows=json.loads((inp/'support_rows.json').read_text());queries={(r['seed'],r['method']):r for r in json.loads((inp/'query_rows.json').read_text())}
    lookup={(r['seed'],r['method']):r for r in rows};assert len(lookup)==320;checks=Counter();maxreg=0.;triggers=[]
    with run.cold.frozen.original.old.core.pipeline.discovery_box(.12):
        for seed in p['seeds']:
            geometry=json.loads((inp/f'{seed}_geometry.json').read_text());seen=set()
            for cfg in p['configs']:
                name=cfg['name'];row=lookup[seed,name];a=np.load(inp/row['file']);x=a['x'];v=a['v'];q=a['q']
                assert sha(inp/row['file'])==row['file_sha256'];xx,vv=run.observations(seed);assert x.tobytes()==xx[:4].tobytes() and v.tobytes()==vv[:4].tobytes()
                replay,rr=run.fit(cfg,x,v,q,loaded,True)
                for key,value in replay.items():assert a[key].tobytes()==value.tobytes(),(seed,name,key);checks['bytewise_arrays']+=1
                checks['method_replays']+=1
                assert rr['support_max_error']==row['support_max_error']
                if cfg['family']=='regression':
                    pred=regression_replay(x,v,q,name,a,row['metadata']);gap=float(np.max(abs(pred-a['prediction'])));assert gap<1e-10
                    maxreg=max(maxreg,gap);checks['independent_regressions']+=1
                if cfg['family']=='meta':checks['frozen_meta_replays']+=1
                if cfg['family'] in ['local','bp']:
                    assert a['starts'].tobytes()==run.cold.starts().tobytes();best=a['starts'].copy();anchor=np.zeros(4)
                    for b in a['b']:run.cold.frozen.retain(best,b,x,v,anchor)
                    assert best.tobytes()==a['best_bank'].tobytes();index,b=run.cold.select(best,x,v)
                    assert index==a['selected_restart'] and b.tobytes()==a['selected_b'].tobytes();checks['support_retention_and_selection']+=1
                    keys=set()
                    for b in a['b'].reshape(-1,4):
                        key=run.cold.base.pattern(x,b).astype(np.uint8).tobytes().hex();keys.add(key)
                        if key not in seen:
                            r=run.cold.base.branch_feasibility(x,v,b);assert r['lp_status']==geometry[key]['lp_status'];seen.add(key);checks['unique_branch_lp_replays']+=1
                    assert sorted(keys)==row['mode_keys'] and sorted(k for k in keys if geometry[k]['current_branch_feasible'])==row['feasible_modes']
                if name==p['primary']:
                    # Reconstruct conditions from raw histories, without calling trigger().
                    delta=np.maximum(np.max(abs(np.diff(a['b'],axis=0)),axis=2),np.max(abs(np.diff(a['h'],axis=0)),axis=(1,3)))
                    active=np.zeros(17,dtype=bool);first=np.full(17,-1,dtype=int)
                    for t in range(1,129):
                        h=a['h'][t];b=a['b'][t];prev=np.broadcast_to(x,(17,len(x)));res=[]
                        for j in range(4):res.append(h[j]-run.cold.base.g(prev+b[:,j,None]));prev=h[j]
                        res=np.array(res);expected_u=a['u'][t-1]+np.where(active,.5,0.)[None,:,None]*res
                        assert expected_u.tobytes()==a['u'][t].tobytes();checks['causal_multiplier_updates']+=1
                        err,_=run.cold.base.score(a['best'][t],x,v,np.zeros(4))
                        proposed=np.zeros(17,dtype=bool) if t==1 else (~active)&(delta[t-2]<=1e-12)&(delta[t-1]<=1e-12)&(np.max(abs(res),axis=(0,2))>1e-6)&(err>.001001)
                        assert not np.any(a['u'][t,:,proposed]);first[proposed]=t;active|=proposed
                        assert active.tobytes()==a['active'][t].tobytes() and first.tobytes()==a['first_trigger'][t].tobytes();checks['trigger_steps']+=1
                    assert first.tolist()==row['metadata']['first_trigger']
                    for i,t in enumerate(first):
                        if t>=0:triggers.append(dict(seed=seed,restart=i,trigger_after=int(t),active_sweeps=max(128-int(t),0),selected=i==int(a['selected_restart'])))
                teacher=np.random.default_rng(seed).uniform(-.12,.12,4);truth=forward(q,teacher[None])[0];risk=float(np.mean((a['prediction']-truth)**2))
                assert risk==queries[seed,name]['mse'];checks['risk_replays']+=1
    dump(out/'trigger_events.json',triggers)
    result=dict(passed=True,source_sha256=sha(Path(__file__)),protocol_sha256=sha(inp/'protocol.json'),run_summary_sha256=sha(inp/'summary.json'),checks=checks,
        triggered_restarts=len(triggers),executed_switches=sum(t['active_sweeps']>0 for t in triggers),selected_triggered_tasks=sum(t['selected'] for t in triggers),
        trigger_range=[min(t['trigger_after'] for t in triggers),max(t['trigger_after'] for t in triggers)] if triggers else None,
        max_independent_regression_error=maxreg,events_sha256=sha(out/'trigger_events.json'))
    dump(out/'summary.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
