"""Full20-method evidence, trigger causality, and shared feasible-mode sets."""
import argparse
import json
from pathlib import Path
import numpy as np
import cold_stagnation_switch as cold
from run_multiplier_fixed_point_screen import sha,dump

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();base=root/'results/cold_stagnation_switch';inp=base/'development'
    out=base/'analysis';out.mkdir(parents=True,exist_ok=True);assert not (out/'summary.json').exists()
    p=json.loads((inp/'protocol.json').read_text());audit=json.loads((base/'audit/summary.json').read_text());resource=json.loads((base/'resources/summary.json').read_text());assert audit['passed'] and resource['passed']
    rows=json.loads((inp/'support_rows.json').read_text());query=json.loads((inp/'query_rows.json').read_text());events=json.loads((base/'audit/trigger_events.json').read_text())
    lookup={(r['seed'],r['method']):r for r in rows};qlookup={(r['seed'],r['method']):r['mse'] for r in query};methods={};contrasts={};overlaps=[];causal=[];same=0
    for cfg in p['configs']:
        name=cfg['name'];rr=[lookup[s,name] for s in p['seeds']]
        methods[name]=dict(mean_mse=float(np.mean([qlookup[s,name] for s in p['seeds']])),support_feasible=sum(r['support_feasible'] for r in rr),
            mean_support_error=float(np.mean([r['support_max_error'] for r in rr])),feasible_mode_sum=sum(len(r.get('feasible_modes',[])) for r in rr),
            **resource['methods'][name])
        if name!=p['primary']:
            delta=np.array([qlookup[s,p['primary']]-qlookup[s,name] for s in p['seeds']]);contrasts[name]=dict(mean_difference=float(delta.mean()),
                better=int(np.sum(delta< -1e-12)),same=int(np.sum(abs(delta)<=1e-12)),worse=int(np.sum(delta>1e-12)),differences=delta.tolist())
    for seed in p['seeds']:
        a=np.load(inp/f'{seed}_{p["primary"]}.npz');b=np.load(inp/f'{seed}_nodual128.npz');x=a['x'];v=a['v'];same+=a['prediction'].tobytes()==b['prediction'].tobytes()
        bank={cfg['name']:set(lookup[seed,cfg['name']]['feasible_modes']) for cfg in p['configs'] if cfg['family'] in ['local','bp']}
        candidate=bank[p['primary']];other=set().union(*(value for name,value in bank.items() if name!=p['primary']))
        bp=set().union(*(bank[n] for n in ['adam60','adam240','gn20','gn40']))
        overlaps.append(dict(seed=seed,candidate=len(candidate),nodual=len(bank['nodual128']),alm=len(bank['alm128']),bp_union=len(bp),all_other_union=len(other),
            new_over_nodual=sorted(candidate-bank['nodual128']),new_over_alm=sorted(candidate-bank['alm128']),new_over_bp=sorted(candidate-bp),new_over_all=sorted(candidate-other),
            all_other_missing=sorted(other-candidate)))
        for event in [e for e in events if e['seed']==seed]:
            i=event['restart'];t=event['trigger_after']
            assert a['b'][:t+1,i].tobytes()==b['b'][:t+1,i].tobytes() and a['h'][:t+1,:,i].tobytes()==b['h'][:t+1,:,i].tobytes()
            ab=np.max(abs(cold.base.forward(x,a['best'][-1,i])-v));bb=np.max(abs(cold.base.forward(x,b['best'][-1,i])-v))
            after=[k for k in range(t+1,129) if a['b'][k,i].tobytes()!=b['b'][k,i].tobytes()]
            initial=cold.base.pattern(x,a['b'][t,i]);changed=[k for k in range(t+1,129) if np.any(cold.base.pattern(x,a['b'][k,i])!=initial)]
            causal.append(dict(**event,first_bias_difference=after[0] if after else None,first_forward_branch_change=changed[0] if changed else None,
                best_support_error=float(ab),nodual_best_support_error=float(bb),support_improved=bool(ab<bb-1e-10),support_worse=bool(ab>bb+1e-10)))
    dump(out/'methods.json',methods);dump(out/'contrasts.json',contrasts);dump(out/'mode_overlaps.json',overlaps);dump(out/'trigger_causality.json',causal)
    text=['| Method | Query MSE | Support feasible /16 | Feasible modes (sum) | Untraced mean seconds | Traced peak bytes (2 tasks) |','|---|---:|---:|---:|---:|---:|']
    for n,r in methods.items():text.append(f'| {n} | {r["mean_mse"]:.9f} | {r["support_feasible"]} | {r["feasible_mode_sum"]} | {r["mean_seconds"]:.6f} | {r["maximum_traced_peak_bytes"]} |')
    (out/'all_methods.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    result=dict(passed=True,source_sha256=sha(Path(__file__)),protocol_sha256=sha(inp/'protocol.json'),audit_sha256=sha(base/'audit/summary.json'),resources_sha256=sha(base/'resources/summary.json'),
        same_prediction_as_nodual_tasks=same,triggered_restarts=len(causal),triggered_with_branch_change=sum(r['first_forward_branch_change'] is not None for r in causal),
        triggered_support_improved=sum(r['support_improved'] for r in causal),triggered_support_worse=sum(r['support_worse'] for r in causal),
        trigger_median=float(np.median([r['trigger_after'] for r in causal])),
        novel_modes={kind:sum(len(r[kind]) for r in overlaps) for kind in ['new_over_nodual','new_over_alm','new_over_bp','new_over_all','all_other_missing']},
        output_sha256={n:sha(out/n) for n in ['methods.json','contrasts.json','mode_overlaps.json','trigger_causality.json','all_methods.md']},
        scope='All16 old development tasks and all20 methods; no new significance or cold-start superiority claim')
    dump(out/'summary.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
