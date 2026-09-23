"""252: audit all artifacts; independently minimize all visited bias blocks.

Float scalar interval enumeration audits every non-PC bias point; exact
three-point interpolation additionally audits all dual/activity-only points.
"""
import argparse
from collections import Counter
from fractions import Fraction as F
import json
from pathlib import Path
import time
import numpy as np
import local_dual_jump_transfer as transfer
from audit_local_dual_jump import g
from run_multiplier_fixed_point_screen import sha,dump

def optimum(previous,target,old,rational):
    conv=(lambda t:F(float(t))) if rational else float
    prev=list(map(conv,previous));target=list(map(conv,target));old=conv(old);bound=conv(.12);trust=conv(.01)
    knots=[conv(0),conv(.5),conv(1)];breaks=sorted({-bound,bound}|{k-p for k in knots for p in prev if -bound<k-p<bound})
    fun=g if rational else lambda z:max(0.,1-abs(2*z-1))
    def energy(z):return sum((fun(p+z)-y)**2 for p,y in zip(prev,target))/len(prev)+trust*(z-old)**2
    best=None
    for lo,hi in zip(breaks[:-1],breaks[1:]):
        if rational:
            f0,fm,f1=[energy(t) for t in [lo,(lo+hi)/2,hi]];aa=2*(f1+f0-2*fm);cc=f1-f0-aa;assert aa>0
            t=min(F(1),max(F(0),-cc/(2*aa)));z=lo+t*(hi-lo)
        else:
            # Direct independent scalar sums, not the cumulative-event solver.
            mid=(lo+hi)/2;slopes=[0. if p+mid<0 or p+mid>1 else 2. if p+mid<.5 else -2. for p in prev]
            offsets=[fun(p+mid)-s*mid for p,s in zip(prev,slopes)]
            aa=sum(s*s for s in slopes)/len(prev)+trust
            bb=sum(s*(y-c) for s,y,c in zip(slopes,target,offsets))/len(prev)+trust*old
            z=min(hi,max(lo,bb/aa))
        for zz in [lo,z,hi]:
            value=energy(zz)
            if best is None or value<best:best=value
    return energy,best,len(breaks)-1

def key(x,b):return transfer.base.pattern(x,b).astype(np.uint8).tobytes().hex()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();src=Path(__file__).parent
    parent=root/'results/local_dual_jump/transfer';out=root/'results/local_dual_jump/transfer_audit';out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    p=json.loads((parent/'protocol.json').read_text());summary=json.loads((parent/'summary.json').read_text());rows=json.loads((parent/'support_rows.json').read_text())
    for name,h in p['source_sha256'].items():assert sha(src/name)==h,name
    for name in ['protocol','support_rows','geometry_files']:assert sha(parent/f'{name}.json')==summary[f'{name}_sha256']
    hashes={**p['source_sha256'],Path(__file__).name:sha(Path(__file__))}
    dump(out/'protocol.json',dict(source_sha256=hashes,transfer_summary_sha256=sha(parent/'summary.json'),query_targets_accessed=False))
    geometry={}
    for item in json.loads((parent/'geometry_files.json').read_text()):
        assert sha(parent/item['file'])==item['sha256'];geometry[int(item['file'].split('_')[0])]=json.loads((parent/item['file']).read_text())
    counts=Counter();maxgap=0.;maxexactgap=0.;seen=set();begin=time.perf_counter();exact_rows=[]
    with transfer.cold.frozen.original.old.core.pipeline.discovery_box(.12):
        for row in rows:
            assert sha(parent/row['file'])==row['sha256'];a=np.load(parent/row['file']);x,v,b,h,old=[a[k] for k in ['x','v','initial_b','initial_h','incumbent']]
            actual,meta=transfer.run(b,h,old,x,v,row['event'],row['method'])
            for name,value in actual.items():assert value.tobytes()==a[name].tobytes(),(row['file'],name);counts['replayed_arrays']+=1
            # Independent support archive selection, including every branch trial.
            bank=np.r_[old[None],b[None],a['trial_b']];errors=[]
            for bb in bank:
                yy=x.copy()
                for bias in bb:yy=np.maximum(0.,1.-np.abs(2.*(yy+bias)-1.))
                errors.append(float(np.max(abs(yy-v))))
            order=sorted(range(len(bank)),key=lambda k:(errors[k]>.001001,.5*float(bank[k]@bank[k]) if errors[k]<=.001001 else errors[k],k))
            assert bank[order[0]].tobytes()==a['best'].tobytes();counts['archive_checks']+=1
            if row['method'] in ['alm1','nodual1','pc1']:
                hist=transfer.cold.frozen.local(b[None],h[:,None],old[None],x,v,row['method'][:-1],1)
                for name in ['b','h','u','best']:
                    expected=hist[name][-1,0] if name in ['b','best'] else hist[name][-1,:,0]
                    assert expected.tobytes()==a[name].tobytes();counts['frozen_control_arrays']+=1
            for idx,bb in enumerate(a['trial_b']):
                k=key(x,bb);pair=(row['seed'],k)
                if pair not in seen:
                    result=transfer.base.branch_feasibility(x,v,bb);assert result['current_branch_feasible']==geometry[row['seed']][k]['current_branch_feasible'];seen.add(pair);counts['lp_checks']+=1
                if row['method'] in ['pc1','unchanged'] or (row['method']=='branch_probe' and idx==0):continue
                hh=a['trial_h'][:,idx];uu=np.zeros_like(hh) if row['method'] in ['alm1','nodual1'] else a['trial_u'][:,idx]
                for j in range(4):
                    prev=x if j==0 else hh[j-1];target=hh[j]+uu[j]
                    energy,minimum,intervals=optimum(prev,target,b[j],False);gap=energy(float(bb[j]))-minimum;maxgap=max(maxgap,gap)
                    assert -.120000000000001<=bb[j]<=.120000000000001;assert gap<=2e-12,(row['file'],idx,j,gap)
                    counts['float_bias_blocks']+=1;counts['float_bias_intervals']+=intervals
                    if row['method'] in ['dual_jump','activity_only']:
                        energy,minimum,intervals=optimum(prev,target,b[j],True);gap=energy(F(float(bb[j])))-minimum;maxexactgap=max(maxexactgap,float(gap))
                        assert F(0)<=gap<=F(2e-12),(row['file'],j,float(gap));counts['exact_bias_blocks']+=1;counts['exact_bias_intervals']+=intervals
                        exact_rows.append(dict(seed=row['seed'],restart=row['restart'],method=row['method'],layer=j,objective_gap=[str(gap.numerator),str(gap.denominator)]))
            counts['method_states']+=1
            if counts['method_states']%153==0:print(json.dumps(dict(seed=row['seed'],counts=counts,seconds=time.perf_counter()-begin)),flush=True)
    dump(out/'exact_bias_gaps.json',exact_rows)
    result=dict(passed=True,counts=counts,max_float_bias_objective_gap=maxgap,max_exact_bias_objective_gap=maxexactgap,seconds=time.perf_counter()-begin,
        protocol_sha256=sha(out/'protocol.json'),exact_bias_gaps_sha256=sha(out/'exact_bias_gaps.json'),query_targets_accessed=False)
    dump(out/'summary.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
