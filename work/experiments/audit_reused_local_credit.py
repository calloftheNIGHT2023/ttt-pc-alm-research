"""No additional optimization: reuse discovery multipliers as safe cell cuts."""
import argparse,hashlib,json
from pathlib import Path
from fractions import Fraction
import numpy as np
from scipy.optimize import linprog
import split_activity_mode_trace as trace
import local_region_screen as screen
import neighbor_mode_memory as model
from certified_branch_solver import exact_certificate


class Observer:
    def __init__(self,x,v):
        self.x=x;self.v=v;self.keys=set();self.cuts={};self.tests=0
    def parameter(self,b,step):pass
    def activity(self,b,h,u,step,phase):
        d,r,n=h.shape;prev=np.broadcast_to(self.x,(r,n));codes=[];res=[]
        for j in range(d):
            z=prev+b[:,j,None];codes.append(np.searchsorted(model.base.KNOTS,z,side='right').astype(np.uint8))
            res.append(h[j]-model.base.g(z));prev=h[j]
        split=np.array(codes).transpose(1,0,2);forward=trace.old.interface.archived.signatures(self.x,b)
        residual=np.array(res).transpose(1,0,2);raw=u.transpose(1,0,2)
        for origin,regs in [('forward',forward),('split',split)]:
            self.keys.update(reg.tobytes() for reg in regs)
            for label,credit in [('raw_dual',raw),('augmented_credit',raw+residual)]:
                p=model.base.SLOPES[regs]*credit
                rough=screen.float_bound(self.x,self.v,regs,p,credit);scale=1+np.abs(p).sum((1,2))+np.abs(credit).sum((1,2))
                ids=np.flatnonzero(rough>1e-10*scale);self.tests+=len(regs)
                if not len(ids):continue
                lower=screen.certified_lower_bound(self.x,self.v,regs[ids],p[ids],credit[ids])
                for i,value in zip(ids,lower):
                    if value<=0:continue
                    key=(label,regs[i].tobytes())
                    if key in self.cuts and self.cuts[key]['lower']>=value:continue
                    self.cuts[key]=dict(lower=float(value),p=p[i].copy(),a=credit[i].copy(),step=step,phase=phase,origin=origin)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--project',type=Path,required=True);a=parser.parse_args()
    refroot=a.project/'results/posterior_state_reuse';out=a.project/'results/reused_local_credit/diagnostic'
    old=json.loads((a.project/'results/split_activity_modes/development/protocol.json').read_text());hashes=old['source_sha256'].copy()
    for name,h in hashes.items():assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()==h,name
    for name in [Path(__file__).name,'certified_branch_solver.py']:
        hashes[name]=hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
    configs=[dict(name='alm16',method='alm',sweeps=16),dict(name='nodual16',method='nodual',sweeps=16),dict(name='pc80',method='pc',sweeps=80)]
    protocol=dict(seeds=list(range(5900000,5900016)),configs=configs,source_sha256=hashes,
        predeclared='raw a=u / augmented a=u+r; p=s_R*a; both forward and split codes; only strict outward-rounded lower bound may reject',
        controls='shared-bias interval contractor 5/20; exact Fraction + independent global LP on every accepted cut',
        scope='support-only old-task diagnostic; no additional dual optimization and no online speed claim')
    out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8');rows=[];proofs=[]
    refs={r['seed']:r for r in json.loads((refroot/'first_write_reference/coverage.json').read_text())}
    for seed in protocol['seeds']:
        path=refroot/'first_write_reference'/refs[seed]['reference_file'];assert hashlib.sha256(path.read_bytes()).hexdigest()==refs[seed]['reference_sha256']
        data=json.loads(path.read_text());x=np.array(data['x']);v=np.array(data['v']);anchor=np.zeros(4)
        pool=trace.old.interface.make_pool(4,'prior256');starts,_=trace.old.interface.select_pool(x,v,anchor,pool,64)
        for cfg in configs:
            obs=Observer(x,v)
            with trace.old.core.pipeline.discovery_box(.12):
                actual=trace.refine(starts,x,v,anchor,cfg['method'],cfg['sweeps'],obs)
                original=trace.original(starts,x,v,anchor,cfg['method'],cfg['sweeps'])
            assert np.array_equal(actual,original)
            keys=sorted(obs.keys);regs=np.array([np.frombuffer(k,np.uint8).reshape(4,4) for k in keys]);indices={k:i for i,k in enumerate(keys)}
            contract5=screen.contract(x,v,regs,5);contract20=screen.contract(x,v,regs,20)
            positive={r['pattern'] for r in data['positive_regions']};assert not any(contract5[i] or contract20[i] for i,k in enumerate(keys) if k.hex() in positive)
            for label in ['raw_dual','augmented_credit']:
                cuts=[(k,c) for (kind,k),c in obs.cuts.items() if kind==label]
                for key,cut in cuts:
                    reg=np.frombuffer(key,np.uint8).reshape(4,4)
                    with trace.old.core.pipeline.discovery_box(.12):exact=exact_certificate(x,v,reg,cut['p'],cut['a'])
                    value=Fraction(int(exact['numerator']),int(exact['denominator']))
                    assert value>0 and Fraction(cut['lower'])<=value
                    _,_,g,rhs=model.pattern_matrix(x,v,reg)
                    lp=linprog(np.zeros(4),A_ub=g,b_ub=rhs,bounds=[(-.12,.12)]*4)
                    assert lp.status==2 and key.hex() not in positive
                    proofs.append(dict(seed=seed,method=cfg['name'],credit=label,pattern=key.hex(),lower=cut['lower'],
                        p=cut['p'].tolist(),a=cut['a'].tolist(),exact=exact,lp_status=int(lp.status),origin=cut['origin'],step=cut['step'],phase=cut['phase']))
                rows.append(dict(seed=seed,method=cfg['name'],credit=label,patterns=len(keys),candidate_row_tests=obs.tests,
                    original_path_bitwise=True,certified_patterns=len(cuts),contract5_rejects=int(contract5.sum()),contract20_rejects=int(contract20.sum()),
                    extra_over_contract5=sum(not contract5[indices[k]] for k,c in cuts),extra_over_contract20=sum(not contract20[indices[k]] for k,c in cuts)))
        (out/'audits.json').write_text(json.dumps(rows,indent=2),encoding='utf-8');(out/'proofs.json').write_text(json.dumps(proofs,indent=2),encoding='utf-8')
        print(json.dumps(dict(completed=seed-protocol['seeds'][0]+1,total=16,exact_proofs=len(proofs))),flush=True)
    summary=[]
    for cfg in configs:
        for label in ['raw_dual','augmented_credit']:
            rr=[r for r in rows if r['method']==cfg['name'] and r['credit']==label]
            summary.append(dict(method=cfg['name'],credit=label,**{k:sum(r[k] for r in rr) for k in
                ['patterns','certified_patterns','contract5_rejects','contract20_rejects','extra_over_contract5','extra_over_contract20']}))
    result=dict(summary=summary,source_hashes=len(hashes),exact_rational_and_lp_proofs=len(proofs),original_paths_exact=48,
        scope='candidate extra screening utility only; no runtime advantage established')
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
