"""Replay all continuations and report task-balanced conditional outcomes."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
import multiplier_warm_continuation as warm
from run_multiplier_fixed_point_screen import sha,dump

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();base=root/'results/multiplier_fixed_point'
    inp=base/'continuation';out=base/'analysis';out.mkdir(parents=True,exist_ok=True);assert not (out/'summary.json').exists()
    p=json.loads((inp/'protocol.json').read_text());manifest=json.loads((inp/'before_query_manifest.json').read_text())
    for name,h in p['source_sha256'].items():assert sha(Path(__file__).parent/name)==h,name
    assert sha(inp/'protocol.json')==manifest['protocol_sha256'] and sha(inp/'support_rows.json')==manifest['support_rows_sha256']
    for group in ['arrays','geometry']:
        for name,h in manifest[group].items():assert sha(inp/name)==h,name
    rows=json.loads((inp/'support_rows.json').read_text());query=json.loads((inp/'query_rows.json').read_text());checks=Counter();taskrows=[]
    with warm.original.old.core.pipeline.discovery_box(.12):
        for seed in range(5900000,5900016):
            geom=json.loads((inp/f'{seed}_geometry.json').read_text())['modes'];validated=set()
            for name,method,steps in p['configs']:
                a=np.load(inp/f'{seed}_{name}.npz');x=a['x'];v=a['v'];starts=a['b'][0];incumbent=a['incumbent']
                if method in ['alm','nodual','pc']:expected=warm.local(starts,a['h'][0],incumbent,x,v,method,steps)
                else:expected,_=warm.baseline(starts,incumbent,x,v,method,steps)
                for key,values in expected.items():assert values.tobytes()==a[key].tobytes();checks['bytewise_arrays']+=1
                retained=incumbent.copy();anchor=np.zeros(4)
                for bank in a['b']:warm.retain(retained,bank,x,v,anchor)
                assert retained.tobytes()==a['best'][-1].tobytes();checks['retention_batches']+=1
                for i,rid in enumerate(a['restarts']):
                    row=next(r for r in rows if (r['seed'],r['method'],r['restart'])==(seed,name,rid))
                    assert sha(inp/row['file'])==row['file_sha256'];keys=set()
                    for b in np.r_[incumbent[i:i+1],a['b'][:,i]]:
                        key=warm.base.pattern(x,b).astype(np.uint8).tobytes().hex();keys.add(key)
                        if key not in validated:
                            replay=warm.base.branch_feasibility(x,v,b);assert replay['lp_status']==geom[key]['lp_status'];validated.add(key);checks['branch_lp_replays']+=1
                    assert sorted(keys)==row['mode_keys'];assert sorted(k for k in keys if geom[k]['current_branch_feasible'])==row['feasible_modes']
                    err,_=warm.base.score(a['best'][-1,i:i+1],x,v,anchor);assert err[0]==row['best_error'];checks['support_points']+=1
                    q=np.linspace(0,1,257);teacher=np.random.default_rng(seed).uniform(-.12,.12,4);truth=warm.base.forward(q,teacher)
                    risk=float(np.mean((warm.base.forward(q,a['best'][-1,i])-truth)**2));qr=next(r for r in query if (r['seed'],r['method'],r['restart'])==(seed,name,rid))
                    assert risk==qr['mse'];checks['query_replays']+=1
                sr=[r for r in rows if r['seed']==seed and r['method']==name];qr=[r for r in query if r['seed']==seed and r['method']==name]
                taskrows.append(dict(seed=seed,method=name,points=len(sr),mse=float(np.mean([r['mse'] for r in qr])),
                    initial_mse=float(np.mean([r['initial_mse'] for r in qr])),support_error=float(np.mean([r['best_error'] for r in sr])),
                    feasible=sum(r['strict_band_feasible'] for r in sr),found_any_feasible_mode=sum(bool(r['feasible_modes']) for r in sr)))
    dump(out/'task_balanced_rows.json',taskrows)
    methods={}
    for name,_,_ in p['configs']:
        rr=[r for r in taskrows if r['method']==name];sr=[r for r in rows if r['method']==name]
        methods[name]=dict(task_balanced_mse=float(np.mean([r['mse'] for r in rr])),pooled_mse=float(np.mean([r['mse'] for r in query if r['method']==name])),
            feasible=sum(r['feasible'] for r in rr),found_any_feasible_mode=sum(r['found_any_feasible_mode'] for r in rr),
            branch_changed=sum(r['first_parameter_branch_change'] is not None for r in sr),support_improved=sum(r['best_error']<r['initial_best_error']-1e-10 for r in sr),
            batch_execution_seconds=sum(next(r['batch_seconds_including_trace_and_BP_replay'] for r in sr if r['seed']==seed) for seed in range(5900000,5900016)))
    pairs={}
    for name in methods:
        if name=='alm64':continue
        delta=np.array([next(r['mse'] for r in taskrows if r['seed']==seed and r['method']=='alm64')-next(r['mse'] for r in taskrows if r['seed']==seed and r['method']==name) for seed in range(5900000,5900016)])
        pairs[name]=dict(mean_difference=float(delta.mean()),better=int(np.sum(delta< -1e-12)),same=int(np.sum(abs(delta)<=1e-12)),worse=int(np.sum(delta>1e-12)),task_differences=delta.tolist())
    screenbytes=sum(np.load(f)['b'].nbytes+np.load(f)['h'].nbytes for f in (base/'screen').glob('*.npz'))
    result=dict(passed=True,checks=checks,methods=methods,pairs=pairs,screen_history_numeric_bytes=screenbytes,
        source_sha256=sha(Path(__file__)),continuation_summary_sha256=sha(inp/'summary.json'),exact_audit_sha256=sha(base/'audit/summary.json'),
        witness_summary_sha256=sha(base/'witness/summary.json'),task_rows_sha256=sha(out/'task_balanced_rows.json'),
        inference_scope='Conditional old-development mechanism test; sixteen tasks, not69 independent tasks; descriptive counts only; no posthoc significance claim')
    dump(out/'summary.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
