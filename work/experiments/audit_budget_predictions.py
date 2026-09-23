"""Independent pre-query audit; no target generator/evaluator is imported."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
import matched_budget_suite as suite
from run_independent_hybrid_memory import observations
from analyze_recovered_online_comparison import forward
from audit_local_dual_jump_modes import modes
from run_multiplier_fixed_point_screen import sha,dump

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();src=Path(__file__).parent;base=root/'results/matched_budget_confirmation';inp=base/'confirmation';out=base/'prediction_audit';out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    assert os.environ.get('OPENBLAS_NUM_THREADS')==os.environ.get('OMP_NUM_THREADS')=='1';torch.set_num_threads(1);torch.set_num_interop_threads(1)
    p=json.loads((inp/'protocol.json').read_text());summary=json.loads((inp/'summary.json').read_text());assert summary['passed'] and summary['predictions_complete'] and not summary['query_targets_accessed'];hashes=dict(p['source_sha256']);hashes[Path(__file__).name]=sha(Path(__file__))
    for n,h in hashes.items():assert sha(src/n)==h,n
    before=json.loads((inp/'before_query_manifest.json').read_text());assert sha(inp/'before_query_manifest.json')==summary['before_query_manifest_sha256']
    for n in ['protocol','rows','memory']:assert sha(inp/f'{n}.json')==before[f'{n}_sha256']
    dump(out/'protocol.json',dict(source_sha256=hashes,prediction_summary_sha256=sha(inp/'summary.json'),scope='all inputs, independent forward/readouts and support particles; all point heads and all64 primary pipelines freshly replayed',query_targets_accessed=False))
    rows=json.loads((inp/'rows.json').read_text());cfgs={c['name']:c for c in p['configs']};assert len(rows)==64*len(cfgs);assert len({(r['seed'],r['method']) for r in rows})==len(rows);assert {(r['seed'],r['method']) for r in rows}=={(s,m) for s in p['seeds'] for m in cfgs}
    loaded,manifest=suite.oldfit.meta.load(root);assert manifest==p['checkpoint_manifest'];counts=Counter();gaps=[];maxgap=0.;maxsupport=0.;begin=time.perf_counter();q=np.linspace(0,1,257)
    with suite.cold.frozen.original.old.core.pipeline.discovery_box(.12):
        for row in rows:
            path=inp/row['file'];assert sha(path)==row['sha256']==before['prediction_files'][row['file']];x,v=observations(row['seed']);x=x[:4];v=v[:4];m=row['metadata'];cfg=cfgs[row['method']]
            with np.load(path) as a:
                assert x.tobytes()==a['x_observed'].tobytes() and v.tobytes()==a['v_observed'].tobytes() and q.tobytes()==a['q_observed'].tobytes();counts['input_files']+=1
                assert row['seconds']==m['charged_complete_seconds']>0
                for k in a.files:
                    if np.issubdtype(a[k].dtype,np.number):assert np.isfinite(a[k]).all()
                if m['execution_failed']:
                    expected=forward(q,np.zeros((1,4)))[0];assert a['selected_b'].tobytes()==np.zeros(4).tobytes();assert m['failure_type'] in p['recoverable_exceptions'];counts['fixed_fallbacks']+=1
                elif row['readout']=='mode':
                    points=a['points'];expected=forward(q,points).mean(0);counts['independent_mode_predictions']+=1
                    if m['positive_modes']:
                        assert len(points)==2048;support=float(np.max(abs(forward(x,points)-v)));assert support<=.001+1e-7;maxsupport=max(maxsupport,support)
                        observed=modes(x,points);assert set(observed)<=set(m['positive_modes']);counts['support_particles']+=len(points);counts['sample_cell_memberships']+=len(points)
                        assert set(m['positive_modes'])<=set(m['feasible_modes'])
                        if 'visited_modes' in m:assert set(m['feasible_modes'])<=set(m['visited_modes']) and m['lp_calls']==len(m['visited_modes'])
                        else:assert len(m['feasible_modes'])<=m['observed_modes']==m['lp_calls']
                    else:
                        assert len(points)==1 and points[0].tobytes()==a['selected_b'].tobytes();counts['empty_pool_fallbacks']+=1
                    assert np.max(abs(forward(q,a['selected_b'][None])[0]-a['point_prediction']))<1e-12;counts['independent_point_parameters']+=1
                else:
                    aa,mm=suite.guarded_fit(cfg,x,v,q,row['seed'],loaded);assert not mm['execution_failed'];expected=aa['prediction'];assert expected.tobytes()==a['prediction'].tobytes();counts['fresh_point_head_replays']+=1
                gap=float(np.max(abs(expected-a['prediction'])));assert gap<1e-12;maxgap=max(maxgap,gap);gaps.append(dict(seed=row['seed'],method=row['method'],gap=gap))
                if row['method']==p['primary']:
                    aa,mm=suite.guarded_fit(cfg,x,v,q,row['seed'],loaded);assert mm['execution_failed']==m['execution_failed']
                    for k,value in aa.items():assert value.tobytes()==a[k].tobytes(),(row['seed'],k);counts['fresh_primary_arrays']+=1
                    counts['fresh_primary_pipelines']+=1
            if counts['input_files']%len(cfgs)==0:print(json.dumps(dict(audited=counts['input_files'],total=len(rows),seconds=time.perf_counter()-begin)),flush=True)
    assert counts['fresh_primary_pipelines']==64 and counts['fixed_fallbacks']==summary['failures'];dump(out/'prediction_gaps.json',gaps)
    ans=dict(passed=True,counts=counts,max_independent_prediction_gap=maxgap,max_support_particle_error=maxsupport,seconds=time.perf_counter()-begin,
        protocol_sha256=sha(out/'protocol.json'),prediction_gaps_sha256=sha(out/'prediction_gaps.json'),query_targets_accessed=False)
    dump(out/'summary.json',ans);print(json.dumps(ans),flush=True)

if __name__=='__main__':main()
