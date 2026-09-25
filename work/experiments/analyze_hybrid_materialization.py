"""Independent state/prediction replay and per-task paired materialization costs."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import hybrid_rejection_materialization as memory


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def ci(values):
    a=np.array(values);rng=np.random.default_rng(481773)
    return np.quantile(a[rng.integers(len(a),size=(20000,len(a)))].mean(1),[.025,.975]).tolist()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);args=ap.parse_args();root=args.project.resolve()
    inp=root/'results/hybrid_materialization/development';out=root/'results/hybrid_materialization/analysis';curve=root/'results/posterior_state_reuse/conditional_risk'
    p=json.loads((inp/'protocol.json').read_text());rows=json.loads((inp/'rows.json').read_text());assert json.loads((inp/'run_audit.json').read_text())['complete'] and len(rows)==320
    for name,value in p['source_sha256'].items():assert sha(Path(__file__).with_name(name))==value,name
    checks=dict(state_hashes=0,full_support_checks=0,prediction_replays=0,query_risk_replays=0);max_prediction_error=0.;tasks=[];summaries=[];paired=[]
    for seed in p['seeds']:
        ca=json.loads((curve/f'audit_{seed}.json').read_text());path=curve/ca['curve_file'];assert sha(path)==ca['curve_sha256']
        with np.load(path) as z:x=z['x'];v=z['v'];q=z['q'][::4];full=np.einsum('k,rkq->rq',z['weights'],z['region_means'])[:,::4]
        truth=memory.geometry.base.forward(q,np.random.default_rng(seed).uniform(-.12,.12,4))
        for row in [r for r in rows if r['seed']==seed]:
            state=inp/row['state_file'];assert sha(state)==row['state_sha256'];checks['state_hashes']+=1
            with np.load(state) as z:points=z['points'];prediction=z['prediction'];assert np.array_equal(q,z['q'])
            assert len(points)==row['samples'] and points.nbytes==row['particle_state_bytes']
            codes,h=memory.capture.model.light.forward_many(x,points);assert np.max(abs(h[:,-1]-v))<=.001+1e-8
            assert sorted({r.tobytes().hex() for r in codes})==row['actual_positive_patterns'];checks['full_support_checks']+=len(points)
            # Different particle summation order, independent query chunks.
            replay=np.zeros(len(q))
            for start in range(0,len(points),73):
                pp=points[::-1][start:start+73]
                for startq in range(0,len(q),31):
                    qq=q[startq:startq+31];h=np.broadcast_to(qq,(len(pp),len(qq)))
                    for j in range(4):h=memory.geometry.base.g(h+pp[:,j,None])
                    replay[startq:startq+31]+=h.sum(0)/len(points)
            error=float(np.max(abs(replay-prediction)));assert error<1e-12;max_prediction_error=max(max_prediction_error,error);checks['prediction_replays']+=1
            mse=float(np.trapezoid((replay-truth)**2,x=q));ce=float(np.mean([np.trapezoid((replay-full[a])*(replay-full[b]),x=q) for a,b in [(0,1),(2,3)]]))
            assert abs(mse-row['query_mse_grid257'])<1e-13 and abs(ce-row['conditional_excess_grid257'])<1e-13;checks['query_risk_replays']+=1
        for count in p['sample_counts']:
            for method in p['methods']:
                rr=[r for r in rows if r['seed']==seed and r['method']==method and r['samples']==count];assert len(rr)==2
                fields=['total_context_to_prediction_seconds','query_mse_grid257','conditional_excess_grid257','sampling_seconds','capture_seconds','contraction_seconds','construction_seconds','screen_seconds','read_seconds','physical_proposals','fallback','geometry_fill_samples','rejection_sampling_seconds','fallback_construction_seconds','fallback_draw_seconds']
                tasks.append(dict(seed=seed,method=method,samples=count,**{k:float(np.mean([r[k] for r in rr])) for k in fields}))
        print(json.dumps(dict(seed=seed,**checks)),flush=True)
    lookup={(r['seed'],r['samples'],r['method']):r for r in tasks}
    for count in p['sample_counts']:
        for method in p['methods']:
            rr=[r for r in tasks if r['samples']==count and r['method']==method]
            summaries.append(dict(samples=count,method=method,**{k:float(np.mean([r[k] for r in rr])) for k in fields},particle_state_bytes=count*4*8))
            if method=='local':continue
            delta={k:np.array([lookup[s,count,'local'][k]-lookup[s,count,method][k] for s in p['seeds']]) for k in ['total_context_to_prediction_seconds','query_mse_grid257','conditional_excess_grid257']}
            paired.append(dict(samples=count,comparator=method,mean_delta={k:float(v.mean()) for k,v in delta.items()},descriptive_task_ci95={k:ci(v) for k,v in delta.items()},
                local_faster_tasks=int(np.sum(delta['total_context_to_prediction_seconds']<0)),local_slower_tasks=int(np.sum(delta['total_context_to_prediction_seconds']>0))))
    result=dict(complete=True,analysis_source_sha256=sha(Path(__file__)),input_sha256={n:sha(inp/n) for n in ['protocol.json','rows.json','run_audit.json']},checks=checks,
        maximum_prediction_replay_error=max_prediction_error,summaries=summaries,paired=paired,
        scope='same old discovery and equal particle count; descriptive paired task intervals, not new confirmation or same-budget distinct optimizers; no particle observations treated as tasks')
    out.mkdir(parents=True,exist_ok=True)
    for name,data in [('summary.json',result),('tasks.json',tasks)]: (out/name).write_text(json.dumps(data,indent=2),encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(12.6,4.9));colors=['#89939b','#147c80','#3873ab','#ac75aa','#dc8952']
    for ax,count in zip(axes,p['sample_counts']):
        rr=[r for r in summaries if r['samples']==count];idx=np.arange(len(rr))
        ax.bar(idx,[r['total_context_to_prediction_seconds'] for r in rr],color=colors)
        ax.set_xticks(idx,[r['method'] for r in rr],rotation=18,ha='right');ax.set_title(f'{count} accepted particles; fresh context-to-prediction')
        ax.set_ylabel('Mean seconds across 16 tasks / 2 repetitions');ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    fig.suptitle('Bounded hybrid materialization: all controls share fallback, original ALM discovery')
    fig.tight_layout();fig.savefig(out/'hybrid_materialization.png',dpi=160);plt.close(fig)
    (out/'figure_audit.json').write_text(json.dumps(dict(source_sha256=sha(Path(__file__)),summary_sha256=sha(out/'summary.json'),figure_sha256=sha(out/'hybrid_materialization.png')),indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
