"""Global BP-credit controls only; never part of the local candidate path."""
import argparse,hashlib,json
from pathlib import Path
from fractions import Fraction
import numpy as np
from scipy.optimize import linprog
import credit_cached_mode_memory as model
import local_region_screen as screen
from certified_branch_solver import exact_certificate


class Observer:
    def __init__(self,x,v,evaluate):
        self.x=x;self.v=v;self.evaluate=evaluate;self.last=None;self.current=None;self.history=None
        self.forward={};self.split={};self.cuts={};self.max_gradient_error=0.;self.gradient_checks=0

    def ensure(self,b):
        if self.last is not None and np.array_equal(b,self.last):return
        r,d=b.shape;n=len(self.x);h=np.broadcast_to(self.x,(r,n));deriv=[];codes=[]
        for j in range(d):
            z=h+b[:,j,None];deriv.append(model.base.derivative(z))
            codes.append(np.searchsorted(model.base.KNOTS,z,side='right').astype(np.uint8));h=model.base.g(z)
        raw=h-self.v;res=np.sign(raw)*np.maximum(np.abs(raw)-model.base.EPS,0)
        alpha=np.empty((r,d,n));alpha[:,-1]=-res;deriv=np.array(deriv).transpose(1,0,2)
        for j in range(d-2,-1,-1):alpha[:,j]=deriv[:,j+1]*alpha[:,j+1]
        _,expected_res,jac,_=self.evaluate(b,self.x,self.v)
        gradient=np.einsum('rni,rn->ri',jac,expected_res,optimize=False)/n
        recovered=-(deriv*alpha).sum(2)/n;gap=float(np.max(np.abs(gradient-recovered)))
        assert gap<1e-12,gap;self.max_gradient_error=max(self.max_gradient_error,gap);self.gradient_checks+=1
        if self.last is None:self.history=np.zeros_like(alpha)
        else:self.history+=.5*self.current
        self.last=b.copy();self.current=alpha;self.codes=np.array(codes).transpose(1,0,2)

    def parameter(self,b,step=0):
        self.ensure(b)
        for reg in self.codes:self.forward.setdefault(reg.tobytes(),None)

    def activity(self,b,h,u=None,step=0,phase=''):
        self.ensure(b);r,d=b.shape;prev=np.broadcast_to(self.x,(r,len(self.x)));codes=[]
        for j in range(d):
            codes.append(np.searchsorted(model.base.KNOTS,prev+b[:,j,None],side='right').astype(np.uint8));prev=h[j]
        split=np.array(codes).transpose(1,0,2)
        for reg in self.codes:self.forward.setdefault(reg.tobytes(),None)
        for reg in split:self.split.setdefault(reg.tobytes(),None)
        for regs in [self.codes,split]:
            for label,a in [('current_bp',self.current),('history_bp',self.history),('current_plus_history_bp',self.current+self.history)]:
                p=model.base.SLOPES[regs]*a;rough=screen.float_bound(self.x,self.v,regs,p,a)
                scale=1+np.abs(p).sum((1,2))+np.abs(a).sum((1,2));ids=np.flatnonzero(rough>1e-10*scale)
                if not len(ids):continue
                lower=screen.certified_lower_bound(self.x,self.v,regs[ids],p[ids],a[ids])
                for i,value in zip(ids,lower):
                    if value<=0:continue
                    key=(label,regs[i].tobytes())
                    if key not in self.cuts or value>self.cuts[key]['lower']:
                        self.cuts[key]=dict(lower=float(value),p=p[i].copy(),a=a[i].copy())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--project',type=Path,required=True);a=parser.parse_args()
    out=a.project/'results/bp_credit_control/diagnostic';refroot=a.project/'results/posterior_state_reuse'
    inherited=json.loads((a.project/'results/credit_cached_memory/development/protocol.json').read_text());hashes=inherited['source_sha256'].copy()
    for name,h in hashes.items():assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()==h,name
    hashes[Path(__file__).name]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    localpath=a.project/'results/reused_local_credit/diagnostic/proofs.json';respath=a.project/'results/reused_local_credit/history_increment/proofs.json'
    local=json.loads(localpath.read_text());residual=json.loads(respath.read_text())
    configs=[dict(name='alm16',generator='alm',sweeps=16),dict(name='adam8',generator='adam',steps=8),dict(name='adam16',generator='adam',steps=16)]
    protocol=dict(seeds=list(range(5900000,5900016)),configs=configs,source_sha256=hashes,
        local_proofs_sha256=hashlib.sha256(localpath.read_bytes()).hexdigest(),residual_proofs_sha256=hashlib.sha256(respath.read_bytes()).hexdigest(),
        endpoint='on identical ALM paths, local-credit cuts beyond contractor20 + residual + current/history BP cuts',
        history='.5 times sum of previous negative unnormalized BP adjoints; update only when parameter array changes',
        scope='support-only diagnostic using global BP in a read-only comparator; no candidate implementation or speed/quality claim')
    out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8');rows=[];proofs=[]
    refs={r['seed']:r for r in json.loads((refroot/'first_write_reference/coverage.json').read_text())}
    for seed in protocol['seeds']:
        path=refroot/'first_write_reference'/refs[seed]['reference_file'];assert hashlib.sha256(path.read_bytes()).hexdigest()==refs[seed]['reference_sha256']
        data=json.loads(path.read_text());x=np.array(data['x']);v=np.array(data['v']);anchor=np.zeros(4)
        pool=model.old.interface.make_pool(4,'prior256');starts,_=model.old.interface.select_pool(x,v,anchor,pool,64)
        positive={r['pattern'] for r in data['positive_regions']}
        for cfg in configs:
            evaluate=model.old.core.batched.evaluate;obs=Observer(x,v,evaluate)
            with model.old.core.pipeline.discovery_box(.12):
                if cfg['generator']=='alm':
                    actual=model.trace.refine(starts,x,v,anchor,'alm',cfg['sweeps'],obs)
                    expected=model.trace.original(starts,x,v,anchor,'alm',cfg['sweeps'])
                else:
                    def wrapped(b,*args,**kwargs):
                        result=evaluate(b,*args,**kwargs);obs.parameter(b)
                        obs.activity(b,model.previous.one_pass_relaxation(b,x,v));return result
                    try:
                        model.old.core.batched.evaluate=wrapped
                        actual,_=model.old.core.batched.refine(starts,x,v,anchor,solver='adam',steps=cfg['steps'],lr=.003)
                    finally:model.old.core.batched.evaluate=evaluate
                    expected,_=model.old.core.batched.refine(starts,x,v,anchor,solver='adam',steps=cfg['steps'],lr=.003)
            assert np.array_equal(actual,expected)
            _,expected_regs,_=model.previous.discover(x,v,dict(**cfg,restarts=64))
            keys=sorted(set(obs.forward)|set(obs.split));assert set(keys)=={r.tobytes() for r in expected_regs}
            regs=np.array([np.frombuffer(k,np.uint8).reshape(4,4) for k in keys]);contract20=screen.contract(x,v,regs,20)
            contracted={k.hex() for k,c in zip(keys,contract20) if c};bpcuts={label:set() for label in ['current_bp','history_bp','current_plus_history_bp']}
            for (label,key),cut in obs.cuts.items():
                reg=np.frombuffer(key,np.uint8).reshape(4,4)
                with model.old.core.pipeline.discovery_box(.12):exact=exact_certificate(x,v,reg,cut['p'],cut['a'])
                value=Fraction(int(exact['numerator']),int(exact['denominator']));assert value>0 and Fraction(cut['lower'])<=value
                _,_,g,rhs=model.previous.neighbor.pattern_matrix(x,v,reg);lp=linprog(np.zeros(4),A_ub=g,b_ub=rhs,bounds=[(-.12,.12)]*4)
                assert lp.status==2 and key.hex() not in positive;bpcuts[label].add(key.hex())
                proofs.append(dict(seed=seed,method=cfg['name'],label=label,pattern=key.hex(),lower=cut['lower'],p=cut['p'].tolist(),a=cut['a'].tolist(),exact=exact,lp_status=int(lp.status)))
            union=set().union(*bpcuts.values());local_cuts={r['pattern'] for r in local if r['seed']==seed and r['method']=='alm16'}
            residual_cuts={r['pattern'] for r in residual if r['seed']==seed and r['method']=='alm16'}
            extra=local_cuts-residual_cuts-contracted-union if cfg['generator']=='alm' else set()
            row=dict(seed=seed,method=cfg['name'],patterns=len(keys),original_final_best_bitwise=True,original_pattern_universe_exact=True,
                gradient_checks=obs.gradient_checks,maximum_bp_gradient_error=obs.max_gradient_error,
                current_certificates=len(bpcuts['current_bp']),history_certificates=len(bpcuts['history_bp']),combined_certificates=len(bpcuts['current_plus_history_bp']),
                union_certificates=len(union),bp_union_beyond_contract20=len(union-contracted),
                local_history_beyond_contract20_residual_and_all_bp=len(extra),remaining_local_patterns=sorted(extra))
            rows.append(row)
        (out/'audits.json').write_text(json.dumps(rows,indent=2),encoding='utf-8');(out/'proofs.json').write_text(json.dumps(proofs,indent=2),encoding='utf-8')
        print(json.dumps(dict(completed=seed-protocol['seeds'][0]+1,total=16,proofs=len(proofs))),flush=True)
    summary=[]
    for cfg in configs:
        rr=[r for r in rows if r['method']==cfg['name']]
        summary.append(dict(method=cfg['name'],**{k:sum(r[k] for r in rr) for k in ['patterns','gradient_checks','current_certificates','history_certificates','combined_certificates','union_certificates','bp_union_beyond_contract20','local_history_beyond_contract20_residual_and_all_bp']},
            maximum_bp_gradient_error=max(r['maximum_bp_gradient_error'] for r in rr)))
    result=dict(summary=summary,exact_rational_and_lp_proofs=len(proofs),exact_original_paths=len(rows),source_hashes=len(hashes))
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
