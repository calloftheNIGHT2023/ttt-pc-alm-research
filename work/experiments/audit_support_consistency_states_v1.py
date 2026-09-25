"""324 independent full-trajectory trigger audit using piecewise rational maps."""
import argparse
from collections import Counter
from fractions import Fraction as F
from pathlib import Path
import time
import traceback
import numpy as np
from evaluate_complete_credit_mode_geometry_v1 import read,save,sha


def exact(b,x,v):
    a=[F(float(y)) for y in x];codes=[]
    for bias in b:
        nexta=[];row=[]
        for h in a:
            z=h+F(float(bias))
            if z<0:r,w=0,F(0)
            elif z<F(1,2):r,w=1,2*z
            elif z<1:r,w=2,2-2*z
            else:r,w=3,F(0)
            row.append(r);nexta.append(w)
        codes.extend(row);a=nexta
    loss=sum((max(F(0),abs(y-F(float(target)))-F(.001))**2 for y,target in zip(a,v)),F(0))/(2*len(v))
    return loss,bytes(codes).hex()


def run(root,out):
    started=time.perf_counter();folder=root/'results/support_consistency_trigger/states_v1'
    summary=read(folder/'summary.json');assert summary['passed'];m=read(folder/'before_search_manifest.json')
    assert sha(folder/'before_search_manifest.json')==summary['manifest_sha256']
    for n,digest in m['files_sha256'].items():assert sha(folder/n)==digest
    records={}
    for row in read(folder/'rows.json'):records.setdefault(row['seed'],{})[row['policy']]=read(folder/row['file'])
    fields=['prefix_b','prefix_h','prefix_u','anchor_b','anchor_h','anchor_u','x_observed','v_observed']
    counts=Counter();facts=[];max_credit_gap=0.
    for seed,policies in records.items():
        ref=policies['first_fit'];path=root/'results/certificate_activity_attribution/development'/ref['source_file']
        assert sha(path)==ref['source_sha256']
        with np.load(path,allow_pickle=False) as z:a={n:z[n] for n in fields}
        entries=[]
        for index in range(1088):
            if index<1056:phase,t,r=0,index//33+1,index%33
            else:phase,t,r=1,index-1055,0
            prefix='prefix' if phase==0 else 'anchor'
            loss,mode=exact(a[prefix+'_b'][t,r],a['x_observed'],a['v_observed'])
            entries.append((loss,[phase,t,r],mode));counts['exact_trajectory_states']+=1
        fitting=[i for i,e in enumerate(entries) if e[0]==0]
        chosen=fitting[0] if fitting else min(range(1088),key=lambda i:(entries[i][0],i))
        assert ref['trajectory_index']==chosen and ref['fallback']==(not fitting)
        assert ref['exact_selection_forward_calls']==(chosen+1 if fitting else 1088)
        rng=np.random.default_rng(np.random.SeedSequence([324071,seed]));assert policies['uniform_state']['trajectory_index']==int(rng.integers(1088))
        for policy,rec in policies.items():
            i=rec['trajectory_index'];loss,location,mode=entries[i];phase,t,r=location;prefix='prefix' if phase==0 else 'anchor'
            b=a[prefix+'_b'][t,r];h=a[prefix+'_h'][t,:,r];u=a[prefix+'_u'][t,:,r]
            assert rec['location']==location and rec['original_mode']==mode and rec['exact_support_loss']==str(loss)
            assert rec['support_fit']==(loss==0)
            for field,value in [('b',b),('h',h),('u',u),('x_observed',a['x_observed']),('v_observed',a['v_observed'])]:
                assert np.array_equal(np.array(rec[field]),value);counts['array_fields']+=1
            prediction=a['x_observed'].copy();slopes=[];rr=[];previous=a['x_observed']
            for j,bias in enumerate(b):
                z=prediction+bias;slopes.append(np.where((z>0)&(z<.5),2.,np.where((z>.5)&(z<1.),-2.,0.)))
                prediction=np.maximum(0.,1.-np.abs(2*z-1))
                rr.append(h[j]-np.maximum(0.,1.-np.abs(2*(previous+bias)-1.)));previous=h[j]
            raw=prediction-a['v_observed'];band=np.sign(raw)*np.maximum(abs(raw)-.001,0.)
            bp=np.empty_like(h);bp[-1]=-band
            for j in [2,1,0]:bp[j]=slopes[j+1]*bp[j+1]
            rr=np.array(rr);rng=np.random.default_rng(np.random.SeedSequence([320071,seed,*location]))
            expected=dict(dual=u,dual_plus_residual=u+rr,residual=rr,bp=bp,random_sign=rng.choice([-1.,1.],size=u.shape),zero=np.zeros_like(u))
            for name,value in expected.items():
                gap=float(np.max(abs(np.array(rec['credits'][name])-value)));assert gap==0.;max_credit_gap=max(max_credit_gap,gap)
                counts['credit_arrays']+=1
            facts.append(dict(seed=seed,policy=policy,support_fit=loss==0,dual_norm=float(np.linalg.norm(u)),
                              bp_credit_norm=float(np.linalg.norm(bp)),bp_exactly_zero=not bool(np.any(bp))))
        counts['tasks']+=1
    assert counts['tasks']==64 and counts['credit_arrays']==768
    save(out/'facts.json',facts)
    result=dict(passed=True,counts=dict(counts),max_credit_gap=max_credit_gap,seconds=time.perf_counter()-started,
                input_summary_sha256=sha(folder/'summary.json'),facts_sha256=sha(out/'facts.json'),
                source_sha256=sha(Path(__file__)),query_targets_accessed=False,
                fitted_states=sum(f['support_fit'] for f in facts),
                fitted_states_with_nonzero_dual=sum(f['support_fit'] and f['dual_norm']>0 for f in facts),
                fitted_states_with_zero_bp=sum(f['support_fit'] and f['bp_exactly_zero'] for f in facts))
    save(out/'summary.json',result);print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    try:run(Path(__file__).resolve().parents[2],a.out)
    except Exception:save(a.out/'failure.json',dict(traceback=traceback.format_exc()));raise
