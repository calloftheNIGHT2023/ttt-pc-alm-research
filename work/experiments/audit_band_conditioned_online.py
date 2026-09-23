"""Independent arithmetic, state, initializer, and exact certificate audit."""
import argparse
from collections import Counter
from fractions import Fraction as F
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import linprog
import band_conditioned_online as model
import band_conditioned_meta as meta_model
import band_conditioned_initialization as init
from verify_band_conditioned_online import validate
from analyze_online_primal_gate import forward,regression_replay
import online_interval_h2 as certificate
from audit_light_h2_certificates_v2 import relaxed_lp

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def dump(p,obj):p.write_text(json.dumps(obj,indent=2),encoding='utf-8')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();src=Path(__file__).parent
    assert os.environ.get('OPENBLAS_NUM_THREADS')==os.environ.get('OMP_NUM_THREADS')=='1';torch.set_num_threads(1);torch.set_num_interop_threads(1)
    base=root/'results/band_conditioned_online';inp=base/'development';p=read(inp/'protocol.json');run=read(inp/'run_audit.json');assert run['execution_complete']
    for name,h in p['source_sha256'].items():assert sha(src/name)==h,name
    for name,h in run['input_sha256'].items():assert sha(inp/name)==h,name
    rows=read(inp/'rows.json');episodes=read(inp/'episodes.json');failures=read(inp/'failures.json')
    assert len(episodes)==run['episodes']==1248 and len(rows)==run['stages']==sum(e['stages'] for e in episodes) and len(failures)==run['failures']
    lookup={(r['seed'],r['method'],r['repetition'],r['n']):r for r in rows};eps={(e['seed'],e['method'],e['repetition']):e for e in episodes};assert len(lookup)==len(rows) and len(eps)==len(episodes)
    loaded,manifest=meta_model.load(root);assert manifest==p['checkpoint_manifest'];checks=Counter();maximum=0.;maximum_lp=0.;generation_records=[]
    old=root/'results/online_primal_gate/development_v2';oldrows={(r['seed'],r['method'],r['repetition'],r['n']):r for r in read(old/'rows.json')}
    oldmap={f'prior256_{c}_c5':f'{c}_c5' for c in ['alm','adam60','pc','nodual']}
    oldmap['prior256_alm_native_gate']='alm_native_full_c5_interval_endpoints_primal_gate';oldmap['direct4096_c5']='direct4096_c5'
    oldmap.update({c['name']:c['name'] for c in p['configs'] if c['family']=='regression'})
    for seed in p['seeds']:
        rng=np.random.default_rng(seed);teacher=rng.uniform(-.12,.12,4);xx=rng.uniform(0,1,24);vv=forward(xx,teacher[None])[0]+np.random.default_rng(seed+19000000).uniform(-.001,.001,24)
        q=np.linspace(0,1,257);truth=forward(q,teacher[None])[0];points,records,gm=init.generate(xx[:4],vv[:4]);checks.update(validate(xx[:4],vv[:4],points,records,gm,4));generation_records.append(dict(seed=seed,metadata=gm))
        initializations={};details={}
        for cfg in p['configs']:
            name=cfg['name']
            for rep in range(2):
                ep=eps[seed,name,rep];previous=None;fast=None;rr=[]
                for n in p['stages'][:ep['stages']]:
                    row=lookup[seed,name,rep,n];path=inp/row['state_file'];assert sha(path)==row['state_sha256'];checks['states']+=1
                    with np.load(path) as z:
                        assert xx[:n].tobytes()==z['x'].tobytes() and vv[:n].tobytes()==z['v'].tobytes() and q.tobytes()==z['q'].tobytes();checks['observations_exact']+=1
                        if cfg['family']=='h2':
                            pp=z['points'];anchor=z['anchor'];assert pp.shape==(2048,4) and anchor.tobytes()==pp[0].tobytes()
                            assert np.max(abs(pp))<=.12+1e-12 and np.max(abs(forward(xx[:n],pp)-vv[:n]))<=.001+1e-8
                            digest=hashlib.sha256(anchor.tobytes()+pp.tobytes()).hexdigest();assert row['state_digest']==digest and row['previous_state_digest']==previous;previous=digest;checks['own_state_links']+=int(n>4)
                            replay=np.zeros(len(q))
                            for start in range(0,len(pp),193):replay+=forward(q,pp[::-1][start:start+193]).sum(0)/len(pp)
                            checks['posterior_replays']+=1;checks['support_particles']+=len(pp)
                            events=row['recovery']['attempts'];assert events[-1]['success'] and all(not e['success'] for e in events[:-1]);assert sum(e['seconds'] for e in events)<=row['write_seconds']+1e-6
                            if row['recovery']['triggered']:
                                assert [e['features'] for e in events[1:]]==[4096,16384][:len(events)-1];checks['recoveries']+=1
                            checks['charged_attempts']+=len(events)
                        elif cfg['family']=='meta':
                            with torch.no_grad():fast=loaded[name].adapt(torch.as_tensor(xx[None,:n]),torch.as_tensor(vv[None,:n]),fast)
                            actual=meta_model.arrays(fast);assert all(z[k].tobytes()==a.tobytes() for k,a in actual.items())
                            with torch.no_grad():replay=loaded[name].predict(fast,torch.as_tensor(q[None])).clamp(0,1)[0].numpy()
                            checks['frozen_meta_state_replays']+=1
                        else:replay=regression_replay(xx[:n],vv[:n],q,name,z,row);checks['independent_regression_replays']+=1
                        error=float(np.max(abs(replay-z['prediction'])));assert error<1e-8,(seed,name,n,error);maximum=max(maximum,error)
                        risk=float(np.trapezoid((replay-truth)**2,x=q));assert abs(risk-row['query_mse'])<1e-9;checks['risk_replays']+=1
                        if name in oldmap:
                            reference=oldrows[seed,oldmap[name],rep,n];assert sha(old/reference['state_file'])==reference['state_sha256']
                            with np.load(old/reference['state_file']) as ref:assert all(z[k].tobytes()==ref[k].tobytes() for k in ref.files),(seed,name,n)
                            checks['unchanged_old_control_states']+=1
                        if n==4 and cfg.get('band_pool'):
                            kind=cfg['band_pool'];pool=np.random.default_rng(731).uniform(-.12,.12,(1280 if kind=='prior1280' else 256,4))
                            if kind=='band_inverse':pool=np.vstack([pool,points])
                            starts,_=model.interface.select_pool(xx[:4],vv[:4],np.zeros(4),pool,64)
                            assert starts.tobytes()==z['initial_starts'].tobytes();bm=row['initialization'];assert hashlib.sha256(pool.tobytes()).hexdigest()==bm['pool_sha256']
                            initializations.setdefault(kind,set()).add(starts.tobytes());checks['shared_start_replays']+=1
                    if 'detail_file' in row:
                        assert sha(inp/row['detail_file'])==row['detail_sha256'];detail=read(inp/row['detail_file']);details[name,rep]=detail;checks['details']+=1
                        if cfg.get('band_pool'):
                            attempts=detail['conditioning_attempts'];assert len(attempts)==1
                            if cfg['band_pool']=='band_inverse':assert attempts[0]['inverse_records']==records;checks['inverse_source_records_exact']+=1
                    rr.append(row)
                charged=sum(r['total_seconds'] for r in rr)
                if ep['complete']:assert len(rr)==4 and abs(charged-ep['charged_seconds'])<1e-9
                else:
                    fail=next(f for f in failures if (f['seed'],f['method'],f['repetition'])==(seed,name,rep));assert abs(charged+fail['failed_seconds']-ep['charged_seconds'])<1e-9
                checks['episodes_accounted']+=1
        assert all(len(s)==1 for s in initializations.values())
        for cfg in [c for c in p['configs'] if c.get('primal_upper_gate')]:
            name=cfg['name'];control=f'{cfg["band_pool"]}_{cfg["learner"]}_c5'
            for rep in range(2):
                assert eps[seed,name,rep]['complete']==eps[seed,control,rep]['complete']
                for n in p['stages'][:eps[seed,name,rep]['stages']]:
                    a=lookup[seed,name,rep,n];b=lookup[seed,control,rep,n]
                    with np.load(inp/a['state_file']) as za,np.load(inp/b['state_file']) as zb:assert all(za[k].tobytes()==zb[k].tobytes() for k in za.files)
                    checks['gated_c5_states_bytewise']+=1
            d=details[name,0];assert d['credit_bank']==details[name,1]['credit_bank'] and d['credit_proofs']==details[name,1]['credit_proofs'];checks['credit_repetitions_exact']+=1
            bank=np.array(d['credit_bank'])
            for proof in d['credit_proofs']:
                reg=np.frombuffer(bytes.fromhex(proof['pattern']),np.uint8).reshape(4,4);a=bank[proof['direction']]
                exact=certificate.interval.original.exact_optimum(xx[:4],vv[:4],reg,a);assert exact['positive'];checks['fraction_certificates']+=1
                if proof['kind']=='interval':
                    if proof['empty_shared_bias']:assert 'empty_layer' in exact
                    else:
                        value=F(int(exact['numerator']),int(exact['denominator']));assert F(proof['lower'])<=value<=F(proof['upper']) and proof['lower']>0
                    checks['outward_interval_enclosures']+=1
                else:assert proof['exact']==exact;checks['rational_fallbacks']+=1
                _,_,matrix,rhs=model.neighbor.pattern_matrix(xx[:4],vv[:4],reg)
                lp=linprog(np.zeros(4),A_ub=matrix,b_ub=rhs,bounds=[(-.12,.12)]*4,options={'primal_feasibility_tolerance':1e-9});assert lp.status==2;checks['parameter_infeasible_lp']+=1
                lp,value=relaxed_lp(xx[:4],vv[:4],reg,a)
                if 'empty_layer' in exact:assert lp.status==2
                else:
                    assert lp.success;error=abs(exact['value']-value);assert error<1e-8;maximum_lp=max(maximum_lp,error)
                checks['independent_relaxed_lp']+=1
        print(json.dumps(dict(seed=seed,checks=checks)),flush=True)
    result=dict(passed=True,checks=dict(checks),max_prediction_replay_error=maximum,max_relaxed_lp_error=maximum_lp,
        source_sha256=sha(Path(__file__)),input_sha256={name:sha(inp/name) for name in ['protocol.json','rows.json','episodes.json','failures.json','run_audit.json']},
        scope='Full old-task arithmetic/state/shared-start audit; every deployed first-repetition proof independently exact and LP checked; no confirmation claim')
    out=base/'audit';out.mkdir(parents=True,exist_ok=True);assert not (out/'summary.json').exists();dump(out/'generations.json',generation_records);result['generation_sha256']=sha(out/'generations.json');dump(out/'summary.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
