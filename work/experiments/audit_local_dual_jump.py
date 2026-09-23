"""Independent exact audit: interpolate each conditional quadratic at three points.

Does not call the candidate's stationary-point formula or classifier.
All comparisons include rejected trials and the actual binary writes.
"""
import argparse
from collections import Counter
from fractions import Fraction as F
import json
from pathlib import Path
import time
import numpy as np
from run_multiplier_fixed_point_screen import sha, dump

def g(z): return max(F(0), F(1)-abs(2*z-1))
def branch(z): return sum(z>=k for k in [F(0),F(1,2),F(1)])
def unpack(z): return F(int(z[0]),int(z[1]))

def energy(s,u,z):
    prev,bias,nb,before,nexth=s
    return (z-g(prev+bias)+u[0])**2+(nexth-g(z+nb)+u[1])**2+F(.01)*(z-before)**2

def minima(s,u):
    nb=s[2];ans=[];knots=[F(0),F(1,2),F(1)]
    for k in range(4):
        lo=F(0) if k==0 else max(F(0),knots[k-1]-nb)
        hi=F(1) if k==3 else min(F(1),knots[k]-nb)
        if lo>hi:continue
        if lo==hi:z=lo
        else:
            mid=(lo+hi)/2;fl,fm,fh=[energy(s,u,t) for t in [lo,mid,hi]]
            half=(hi-lo)/2
            aa=(fl+fh-2*fm)/(2*half**2);bb=(fh-fl)/(2*half)
            assert aa>0
            z=min(hi,max(lo,mid-bb/(2*aa)))
        ans.append(dict(k=k,lo=lo,hi=hi,z=z,energy=energy(s,u,z),actual_branch=branch(z+nb)))
    return ans

def decision(zero,test,k0):
    own=next((r for r in zero if r['k']==k0),None);other=[r for r in zero if r['k']!=k0]
    if own is None:return dict(accepted=False,reason='current branch outside activity domain')
    if not other:return dict(accepted=False,reason='no competing branch')
    if any(own['energy']>=r['energy'] for r in other):return dict(accepted=False,reason='zero dual not strict same-branch optimum')
    same=next(r for r in test if r['k']==k0);winner=min((r for r in test if r['k']!=k0),key=lambda r:(r['energy'],r['k']))
    gap=same['energy']-winner['energy']
    return dict(accepted=gap>0,reason='strict different-branch minimum' if gap>0 else 'no strict branch separation',
        original_branch=k0,competing_branch=winner['k'],actual_branch=winner['actual_branch'],margin=gap,z=winner['z'])

def compare(actual,saved,counts):
    assert len(actual)==len(saved)
    for a,b in zip(actual,saved):
        for key,value in a.items(): assert value==(b[key] if key in ['k','actual_branch'] else unpack(b[key])),key
        counts['branch_quadratics']+=1

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();src=Path(__file__).parent
    screen=root/'results/local_dual_jump/screen_v2';out=root/'results/local_dual_jump/audit';out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    p=json.loads((screen/'protocol.json').read_text());summary=json.loads((screen/'summary.json').read_text());files=json.loads((screen/'files.json').read_text())
    for name,h in p['source_sha256'].items():assert sha(src/name)==h,name
    for name in ['protocol','files','selected']:assert sha(screen/f'{name}.json')==summary[f'{name}_sha256']
    dump(out/'protocol.json',dict(screen_summary_sha256=sha(screen/'summary.json'),source_sha256={**p['source_sha256'],Path(__file__).name:sha(Path(__file__))},
        oracle='three exact objective values, quadratic interpolation, clipped vertex',query_targets_accessed=False))
    counts=Counter();selected=[];begin=time.perf_counter()
    for file in files:
        assert sha(screen/file['file'])==file['sha256'];source=root/'results/cold_stagnation_switch/development'/file['source_file'];assert sha(source)==file['source_sha256']
        with np.load(source) as a:
            x=a['x'];bank=a['b'][16];activities=a['h'][16]
        for record in json.loads((screen/file['file']).read_text()):
            restart=record['restart'];b=bank[restart];h=activities[:,restart];candidates=[];counts['states']+=1
            for block in record['blocks']:
                j,i=block['j'],block['i'];s=[F(float(t)) for t in [x[i] if j==0 else h[j-1,i],b[j],b[j+1],h[j,i],h[j+1,i]]]
                k0=branch(s[3]+s[2]);res=[s[3]-g(s[0]+s[1]),s[4]-g(s[3]+s[2])];zero=minima(s,[F(0),F(0)])
                assert k0==block['exact_zero']['current_branch'];assert res==list(map(unpack,block['exact_zero']['residual']))
                compare(zero,block['exact_zero']['branches'],counts);counts['blocks']+=1
                for trial in block['trials']:
                    tau=F(trial['tau']);u=[tau*r for r in res];test=minima(s,u);compare(test,trial['exact_values']['branches'],counts)
                    assert u==list(map(unpack,trial['exact_values']['u']));dec=decision(zero,test,k0)
                    for key,value in dec.items():assert value==(unpack(trial['exact_decision'][key]) if key in ['margin','z'] else trial['exact_decision'][key]),key
                    rounded_ok=False
                    if trial['float_decision']['accepted']:
                        u_round=[F(float(t)) for t in trial['float_values']['u']];z=F(trial['float_decision']['z']);physical=minima(s,u_round)
                        own=next(r for r in physical if r['k']==k0);margin=own['energy']-energy(s,u_round,z)
                        rounded_ok=F(0)<=z<=F(1) and branch(z+s[2])!=k0 and margin>0
                        rr=trial['rounded_write'];assert margin==unpack(rr['margin']);assert rounded_ok==rr['accepted'];assert z==unpack(rr['z']);assert u_round==list(map(unpack,rr['u']))
                        counts['rounded_writes']+=1
                    accept=bool(trial['float_decision']['accepted'] and dec['accepted'] and rounded_ok);assert accept==trial['accepted'];counts['tau_trials']+=1;counts['accepted_trials']+=accept
                    if accept:candidates.append((float(tau),j,i,trial['float_decision']['competing_branch']))
            if candidates:
                want=min(candidates);got=record['selected'];assert want==(got['tau'],got['j'],got['i'],got['k']);counts['selected_states']+=1
                selected.append(dict(seed=file['seed'],restart=restart,**got))
            else:assert record['selected'] is None
        print(json.dumps(dict(seed=file['seed'],states=counts['states'])),flush=True)
    assert selected==json.loads((screen/'selected.json').read_text())
    for key in ['states','blocks','tau_trials','accepted_trials','selected_states']:assert counts[key]==summary['counts'][key]
    result=dict(passed=True,counts=counts,seconds=time.perf_counter()-begin,protocol_sha256=sha(out/'protocol.json'),query_targets_accessed=False)
    dump(out/'summary.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
