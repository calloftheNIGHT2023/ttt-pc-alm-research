"""Certified finite-prior two-edit unreachable feasible modes; supports only."""
import argparse
import hashlib
import json
from fractions import Fraction as F
from pathlib import Path
import numpy as np
import neighbor_mode_memory as model


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def down(a):return np.nextafter(a,-np.inf)
def up(a):return np.nextafter(a,np.inf)


def interval_patterns(x,bank):
    lo=np.broadcast_to(x,(len(bank),len(x))).copy();hi=lo.copy();codes_lo=[];codes_hi=[]
    for j in range(4):
        zl=down(lo+bank[:,j,None]);zu=up(hi+bank[:,j,None])
        codes_lo.append(np.searchsorted([0.,.5,1.],zl,side='right'))
        codes_hi.append(np.searchsorted([0.,.5,1.],zu,side='right'))
        # g(z)=max(0, 1-abs(2*z-1)); outward at every arithmetic operation.
        al=down(down(2*zl)-1);au=up(up(2*zu)-1)
        amin=np.where((al<=0)&(au>=0),0,np.minimum(abs(al),abs(au)))
        amax=np.maximum(abs(al),abs(au))
        lo=np.maximum(0,down(1-amax));hi=np.maximum(0,up(1-amin))
    return np.stack(codes_lo,axis=1).astype(np.int16),np.stack(codes_hi,axis=1).astype(np.int16)


def exact_forward(x,b):
    h=[F(float(a)) for a in x];codes=[]
    for bias in b:
        z=[a+F(float(bias)) for a in h]
        codes.append([sum(a>=k for k in [F(0),F(1,2),F(1)]) for a in z])
        h=[max(F(0),F(1)-abs(2*a-1)) for a in z]
    return np.array(codes,np.int16),h


def feasible_witness(x,v,reg,points):
    for b in points:
        code,out=exact_forward(x,b)
        if not np.array_equal(reg,code):continue
        error=max(abs(a-F(float(y))) for a,y in zip(out,v))
        if error<=F(float(.001)) and all(abs(F(float(a)))<=F(float(.12)) for a in b):
            return dict(biases=b.tolist(),max_error_numerator=str(error.numerator),max_error_denominator=str(error.denominator))
    return None


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve()
    inp=root/'results/context_block_scaling/development';parent=json.loads((inp/'protocol.json').read_text())
    assert json.loads((inp/'run_audit.json').read_text())['execution_complete']
    hashes=dict(parent['source_sha256'])
    for name,value in hashes.items():assert sha(Path(__file__).with_name(name))==value,name
    hashes[Path(__file__).name]=sha(Path(__file__))
    out=inp.parent/'reachability';out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    p=dict(source_sha256=hashes,parent_protocol_sha256=sha(inp/'protocol.json'),rows_sha256=sha(inp/'rows.json'),
        names=['alm_c5','adam60_c5','pc_c5','nodual_c5','direct4096_c5'],prior_sizes=[256,4096,16384],radius=2,
        input_policy='Only support, pattern and state-file fields from frozen rows; no teacher or query-risk field accessed')
    (out/'protocol.json').write_text(json.dumps(p,indent=2),encoding='utf-8')
    # Discard all evaluation fields immediately; do not use them in this audit.
    fields=['seed','method','n','repetition','complete','state_file','state_sha256','positive_mode_keys']
    raw=json.loads((inp/'rows.json').read_text())
    rows=[{k:r[k] for k in fields if k in r} for r in raw if r['repetition']==0 and r['method'] in p['names']];del raw
    lookup={(r['seed'],r['method'],r['n']):r for r in rows};bank=np.vstack([np.zeros(4),np.random.default_rng(731).uniform(-.12,.12,(16384,4))])
    results=[];checks=dict(tasks=0,contexts=0,exact_prior_enclosures=0,positive_patterns=0,exact_feasible_witnesses=0,geometry_fallbacks=0)
    for seed in parent['seeds']:
        for n in parent['block_sizes']:
            successful=[lookup[seed,name,n] for name in p['names'] if lookup[seed,name,n]['complete']]
            if not successful:
                results.append(dict(seed=seed,n=n,all_frozen_methods_exhausted=True,modes=[]));checks['contexts']+=1;continue
            with np.load(inp/successful[0]['state_file']) as z:x=z['x'];v=z['v']
            lower,upper=interval_patterns(x,bank)
            for index in [0,1,2,17,255,256,4095,4096,16383,16384]:
                code,_=exact_forward(x,bank[index]);assert np.all(lower[index]<=code) and np.all(code<=upper[index]);checks['exact_prior_enclosures']+=1
            keys=sorted(set(k for r in successful for k in r['positive_mode_keys']));modes=[]
            states={}
            for row in successful:
                path=inp/row['state_file'];assert sha(path)==row['state_sha256']
                with np.load(path) as z:states[row['method']]=z['points']
            for key in keys:
                reg=np.frombuffer(bytes.fromhex(key),np.uint8).reshape(4,n)
                distances=np.maximum(np.maximum(lower-reg,reg-upper),0).sum(axis=(1,2))
                bounds={str(k):int(distances[:k+1].min()) for k in p['prior_sizes']}
                found=[r['method'] for r in successful if key in r['positive_mode_keys']]
                witness=None
                for name in found:
                    candidates=states[name];codes=np.stack([model.base.pattern(x,b) for b in candidates])
                    candidates=candidates[np.all(codes==reg,axis=(1,2))][:32]
                    witness=feasible_witness(x,v,reg,candidates)
                    if witness is not None:break
                if witness is None:
                    _,_,matrix,rhs=model.pattern_matrix(x,v,reg);poly,_=model.posterior.polytope(matrix,rhs)
                    assert poly is not None
                    point=poly['center']+poly['scale']*poly['interior']
                    witness=feasible_witness(x,v,reg,point[None]);checks['geometry_fallbacks']+=1
                assert witness is not None,(seed,n,key)
                checks['exact_feasible_witnesses']+=1;checks['positive_patterns']+=1
                modes.append(dict(pattern=key,found_by=found,prior_edit_distance_lower_bounds=bounds,
                    certified_outside_two_edit_full16384=bool(bounds['16384']>2),exact_witness=witness))
            results.append(dict(seed=seed,n=n,all_frozen_methods_exhausted=False,
                uncertain_prior_codes=int((lower!=upper).sum()),modes=modes));checks['contexts']+=1
        checks['tasks']+=1;print(json.dumps(dict(seed=seed,**checks)),flush=True)
    summaries=[]
    for n in parent['block_sizes']:
        for name in p['names']:
            rr=[r for r in results if r['n']==n];owned=[m for r in rr for m in r['modes'] if name in m['found_by']]
            beyond=[m for m in owned if m['certified_outside_two_edit_full16384']]
            summaries.append(dict(n=n,method=name,complete_tasks=sum(lookup[s,name,n]['complete'] for s in parent['seeds']),
                found_modes=len(owned),certified_beyond16384_two_edits=len(beyond),
                tasks_with_beyond_modes=sum(any(name in m['found_by'] and m['certified_outside_two_edit_full16384'] for m in r['modes']) for r in rr),
                modes_absent_direct=sum('direct4096_c5' not in m['found_by'] for m in owned)))
    result=dict(passed=True,checks=checks,summaries=summaries,source_sha256=sha(Path(__file__)),
        protocol_sha256=sha(out/'protocol.json'),scope='Finite supplied-prior and two-edit search-family separation, NOT all closed-form/BP algorithms')
    for name,value in [('modes.json',results),('summary.json',result)]:
        (out/name).write_text(json.dumps(value,indent=2),encoding='utf-8')
    print(json.dumps(dict(passed=True,checks=checks)),flush=True)


if __name__=='__main__':main()
