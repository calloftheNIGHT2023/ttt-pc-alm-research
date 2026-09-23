"""Task-level paired summaries, including all controls and failed flows."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def dump(p,obj):p.write_text(json.dumps(obj,indent=2),encoding='utf-8')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();base=root/'results/band_conditioned_online';inp=base/'development'
    p=read(inp/'protocol.json');run=read(inp/'run_audit.json');audit=read(base/'audit/summary.json');assert run['execution_complete'] and audit['passed']
    rows=read(inp/'rows.json');episodes=read(inp/'episodes.json');tasks=[];summaries=[];stages=[]
    eps={(e['seed'],e['method'],e['repetition']):e for e in episodes};rr={(seed,c['name'],rep):[r for r in rows if r['seed']==seed and r['method']==c['name'] and r['repetition']==rep] for seed in p['seeds'] for c in p['configs'] for rep in range(2)}
    for seed in p['seeds']:
        for cfg in p['configs']:
            name=cfg['name'];trials=[]
            for rep in range(2):
                ep=eps[seed,name,rep];rs=sorted(rr[seed,name,rep],key=lambda r:r['n']);complete=ep['complete'];first=rs[0] if rs else None
                trials.append(dict(complete=complete,mse=float(np.mean([r['query_mse'] for r in rs])) if complete else None,time=ep['charged_seconds'] if complete else None,
                    charged_seconds=ep['charged_seconds'],first_mse=first['query_mse'] if first else None,first_time=first['total_seconds'] if first else None,
                    final_mse=rs[-1]['query_mse'] if complete else None,recoveries=ep['recovery_stages'],last_state_bytes=rs[-1]['state_bytes'] if complete else None,
                    shared_model_bytes=max((r['shared_model_bytes'] for r in rs),default=0),support_feasible_rate=float(np.mean([r['support_feasible'] for r in rs])) if complete else None,
                    first_positive_modes=first['positive_modes'] if first else None))
            metrics=[k for k in trials[0] if k!='complete'];record=dict(seed=seed,method=name,complete=all(t['complete'] for t in trials))
            for key in metrics:record[key]=None if any(t[key] is None for t in trials) else float(np.mean([t[key] for t in trials]))
            tasks.append(record)
    lookup={(r['seed'],r['method']):r for r in tasks};indices=np.random.default_rng(p['bootstrap_seed']).integers(len(p['seeds']),size=(p['bootstrap_repeats'],len(p['seeds'])))
    def contrast(left,right):
        answer=dict(left=left,right=right,sign='negative favors left for risk/time')
        for key in ['mse','time','first_mse','first_time','final_mse']:
            values=[None if lookup[s,left][key] is None or lookup[s,right][key] is None else lookup[s,left][key]-lookup[s,right][key] for s in p['seeds']]
            if any(v is None for v in values):answer[key]=None;continue
            a=np.array(values);answer[key]=dict(mean=float(a.mean()),ci95=np.quantile(a[indices].mean(1),[.025,.975]).tolist(),negative_tasks=int(np.count_nonzero(a<0)),zero_tasks=int(np.count_nonzero(a==0)),values=values)
        return answer
    for cfg in p['configs']:
        name=cfg['name'];rs=[r for r in tasks if r['method']==name];summary=dict(method=name,family=cfg['family'],learner=cfg.get('learner'),band_pool=cfg.get('band_pool'),complete_tasks=sum(r['complete'] for r in rs))
        for key in metrics:summary[key]=None if any(r[key] is None for r in rs) else float(np.mean([r[key] for r in rs]))
        summaries.append(summary)
        for n in p['stages']:
            ss=[r for r in rows if r['method']==name and r['n']==n];complete=len(ss)==32
            stages.append(dict(method=name,n=n,observed_stages=len(ss),mse=float(np.mean([r['query_mse'] for r in ss])) if complete else None,
                time=float(np.mean([r['total_seconds'] for r in ss])) if complete else None))
    primary=[contrast(p['primary'],c['name']) for c in p['configs']]
    pools=[contrast(f'band_inverse_{learner}_c5',f'{pool}_{learner}_c5') for learner in ['alm','adam60','adam240','gn20','pc','nodual','direct64'] for pool in ['prior256','prior1280']]
    gates=[contrast(c['name'],f'{c["band_pool"]}_{c["learner"]}_c5') for c in p['configs'] if c.get('primal_upper_gate')]
    out=base/'analysis';out.mkdir(parents=True,exist_ok=True);assert not (out/'summary.json').exists()
    for name,obj in [('tasks.json',tasks),('stages.json',stages),('primary_contrasts.json',primary),('pool_contrasts.json',pools),('gate_contrasts.json',gates)]:dump(out/name,obj)
    result=dict(passed=True,summaries=summaries,complete_episodes=sum(e['complete'] for e in episodes),failed_episodes=sum(not e['complete'] for e in episodes),
        primary=p['primary'],source_sha256=sha(Path(__file__)),audit_sha256=sha(base/'audit/summary.json'),protocol_sha256=sha(inp/'protocol.json'),
        output_sha256={name:sha(out/name) for name in ['tasks.json','stages.json','primary_contrasts.json','pool_contrasts.json','gate_contrasts.json']},
        statistical_scope='Old16 development tasks; repeats averaged within task; descriptive95% paired bootstrap, not confirmation or multiplicity-corrected testing')
    dump(out/'summary.json',result)
    lines=['# 243：全部39配置的新运行结果','','查询列为四阶段均值，再按两次重复和16任务平均；所有数值来自本轮。未完整结果不以成功者均值替代。','','| 方法 | 完整任务 | 查询误差 | 首阶段误差 | 完整流秒 | 首阶段秒 | 恢复次数/流 |','|---|---:|---:|---:|---:|---:|---:|']
    for r in summaries:
        cells=[r['method'],str(r['complete_tasks'])]+['未完整' if r[k] is None else f'{r[k]:.9g}' for k in ['mse','first_mse','time','first_time','recoveries']];lines.append('| '+' | '.join(cells)+' |')
    (out/'all_methods.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(json.dumps(result),flush=True)

if __name__=='__main__':main()
