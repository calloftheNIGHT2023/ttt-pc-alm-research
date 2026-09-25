"""320 support-only proposals; all 131 states and six channels before geometry."""
import argparse
from collections import Counter
from pathlib import Path
import time
import traceback
import numpy as np
import scipy.optimize as opt
import batched_bp_discovery as bp
import factorized_dual_branch_search_v1 as search
from evaluate_complete_credit_mode_geometry_v1 import read,save,sha

CHANNELS=['dual','dual_plus_residual','residual','bp','random_sign','zero']


def forward(b,x):
    h=x.copy();codes=[];slopes=[]
    for bias in b:
        z=h+bias;codes.append(np.searchsorted([0.,.5,1.],z,side='right').astype(np.uint8))
        slopes.append(np.where((z>0)&(z<.5),2.,np.where((z>.5)&(z<1),-2.,0.)))
        h=np.maximum(0.,1.-np.abs(2.*z-1.))
    return np.array(codes),np.array(slopes),h


def bp_control(b,x,v):
    _,slopes,prediction=forward(b,x);raw=prediction-v
    residual=np.sign(raw)*np.maximum(abs(raw)-.001,0.)
    credit=np.empty_like(slopes);credit[-1]=-residual
    for j in range(len(b)-2,-1,-1):credit[j]=slopes[j+1]*credit[j+1]
    _,rr,jac,_=bp.evaluate(b[None],x,v)
    expected=np.einsum('rni,rn->ri',jac,rr,optimize=False)[0]/len(x)
    actual=-(slopes*credit).sum(1)/len(x)
    gap=float(np.max(abs(actual-expected)));assert gap<1e-12
    return credit,gap


def signals(b,h,u,x,seed,location):
    previous=x;residual=[]
    for j,bias in enumerate(b):
        residual.append(h[j]-np.maximum(0.,1.-np.abs(2.*(previous+bias)-1.)));previous=h[j]
    rr=np.array(residual)
    rng=np.random.default_rng(np.random.SeedSequence([320071,int(seed),*map(int,location)]))
    return dict(dual=u.copy(),dual_plus_residual=u+rr,residual=rr,
                random_sign=rng.choice([-1.,1.],size=u.shape),zero=np.zeros_like(u))


def run(root,out):
    begin=time.perf_counter();source=root/'results/solver_policy_recurrence/development_v1'
    tested=read(root/'results/factorized_dual_branch_search/tests_v1/summary.json');assert tested['passed']
    for name,digest in tested['source_sha256'].items():assert sha(root/'work/experiments'/name)==digest
    summary=read(source/'summary.json');assert summary['passed'] and summary['states']==131
    source_rows=[r for r in read(source/'rows.json') if r['family']=='alm_keep']
    for row in source_rows:assert sha(source/row['file'])==row['sha256']
    names=[Path(__file__).name,'factorized_dual_branch_search_v1.py','test_factorized_dual_branch_search_v1.py',
           'batched_bp_discovery.py','local_branch_memory.py','streaming_branch_projection.py','evaluate_complete_credit_mode_geometry_v1.py']
    design=root/'outputs/ttt-pc-alm-research/320_factorized_dual_branch_search_protocol_v1.md'
    protocol=dict(source_sha256={n:sha(root/'work/experiments'/n) for n in names},design_sha256=sha(design),
                  input_summary_sha256=sha(source/'summary.json'),input_rows_sha256=sha(source/'rows.json'),
                  input_files_sha256={r['file']:r['sha256'] for r in source_rows},tests_sha256=sha(root/'results/factorized_dual_branch_search/tests_v1/summary.json'),
                  tasks=64,states=131,channels=CHANNELS,k=8,max_modes_per_state_channel=24,
                  empty_tasks=summary['empty_tasks'],query_targets_accessed=False,geometry_accessed=False,
                  resources_matched=False,bp_scope='Explicit matched-state control only; never supplied to candidate.')
    save(out/'protocol.json',protocol);rows=[];files={};counts=Counter();aggregate={n:Counter() for n in CHANNELS}
    def forbid(*args,**kwargs):raise AssertionError('Global BP/LP/optimizer entered branch selector')
    for old in source_rows:
        with np.load(source/old['file'],allow_pickle=False) as z:
            x,v=z['x_observed'],z['v_observed'];bs=z['b'][0];hs=z['h'][0];us=z['u'][0];locations=z['locations']
        for i,location in enumerate(locations):
            b,h,u=bs[i],hs[:,i],us[:,i];original,_,_=forward(b,x)
            credits=signals(b,h,u,x,old['seed'],location);results={};credit_seconds={}
            for name in CHANNELS:
                if name=='bp':
                    started=time.perf_counter();credits[name],gap=bp_control(b,x,v);credit_seconds[name]=time.perf_counter()-started
                previous=[]
                try:
                    for obj,attr in [(bp,'evaluate'),(bp,'refine'),(opt,'linprog'),(opt,'minimize')]:
                        previous.append((obj,attr,getattr(obj,attr)));setattr(obj,attr,forbid)
                    result=search.propose(x,v,credits[name],original,k=8)
                finally:
                    for obj,attr,func in previous:setattr(obj,attr,func)
                results[name]=result;aggregate[name]['cases']+=1
                aggregate[name]['current_certified_infeasible']+=int(result['current_certified_infeasible'])
                aggregate[name]['proposals']+=len(result['proposals'])
                for field in ['table_seconds','selection_seconds','total_seconds','layer_rows_evaluated','dp_combinations']:
                    aggregate[name][field]+=result['meta'][field]
            record=dict(seed=old['seed'],location=location.tolist(),source_file=old['file'],source_index=i,
                        x_observed=x.tolist(),v_observed=v.tolist(),b=b.tolist(),h=h.tolist(),u=u.tolist(),
                        original_mode=original.tobytes().hex(),credits={n:a.tolist() for n,a in credits.items()},
                        results=results,bp_gradient_check_gap=gap,bp_credit_seconds=credit_seconds['bp'])
            filename=f"{old['seed']}_{i}.json";save(out/filename,record);files[filename]=sha(out/filename)
            rows.append(dict(seed=old['seed'],location=location.tolist(),file=filename,sha256=files[filename]))
            counts['states']+=1;counts['channels']+=len(CHANNELS)
            if counts['states']%5==0:print(dict(states=counts['states'],channels=counts['channels'],seconds=time.perf_counter()-begin),flush=True)
    assert counts['states']==131 and counts['channels']==786
    assert sha(design)==protocol['design_sha256']
    for n,digest in protocol['source_sha256'].items():assert sha(root/'work/experiments'/n)==digest
    for n,obj in [('rows.json',rows),('aggregate.json',aggregate)]:save(out/n,obj);files[n]=sha(out/n)
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
