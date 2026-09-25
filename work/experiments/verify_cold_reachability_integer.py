"""Independent exact integer replay of ALL finite prior paths and witnesses."""
import argparse
import hashlib
import json
from fractions import Fraction as F
from pathlib import Path
import numpy as np


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def exponent(a):return max(float(v).as_integer_ratio()[1].bit_length()-1 for v in np.asarray(a).ravel())


def integers(a,k):
    def one(v):
        num,den=float(v).as_integer_ratio();return num << (k-(den.bit_length()-1))
    a=np.asarray(a);return np.array([one(v) for v in a.ravel()],dtype=object).reshape(a.shape)


def exact_codes(x,bank,k):
    scale=1<<k;xb=integers(x,k);bb=integers(bank,k);answer=np.empty((len(bank),4,len(x)),np.uint8)
    for start in range(0,len(bank),256):
        biases=bb[start:start+256];h=np.broadcast_to(xb,(len(biases),len(x))).copy()
        for j in range(4):
            z=h+biases[:,j,None]
            answer[start:start+len(biases),j]=np.asarray((z>=0),np.uint8)+np.asarray((z>=scale//2),np.uint8)+np.asarray((z>=scale),np.uint8)
            h=np.maximum(0,scale-abs(2*z-scale))
    return answer


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve()
    inp=root/'results/context_block_scaling/development';ref=inp.parent/'reachability'
    p=json.loads((ref/'protocol.json').read_text());summary=json.loads((ref/'summary.json').read_text());assert summary['passed']
    for name,value in p['source_sha256'].items():assert sha(Path(__file__).with_name(name))==value,name
    assert summary['source_sha256']==sha(Path(__file__).with_name('audit_cold_pattern_reachability.py'))
    # Use support inputs only, including the regression saved state if all posterior searches failed.
    rows=json.loads((inp/'rows.json').read_text());supports={r['seed']:(r['state_file'],r['state_sha256']) for r in rows if r['method']=='linear_ls' and r['n']==24 and r['repetition']==0}
    del rows
    modes=json.loads((ref/'modes.json').read_text());bank=np.vstack([np.zeros(4),np.random.default_rng(731).uniform(-.12,.12,(16384,4))])
    checks=dict(prior_parameters=0,exact_prior_activation_codes=0,mode_distance_bounds=0,exact_witnesses=0,strict_two_edit_certificates=0)
    differences=[];certificates=[]
    for seed,(filename,digest) in supports.items():
        path=inp/filename;assert sha(path)==digest
        with np.load(path) as z:x=z['x'];v=z['v']
        k=max(exponent(x),exponent(bank));codes=exact_codes(x,bank,k)
        checks['prior_parameters']+=len(bank);checks['exact_prior_activation_codes']+=codes.size
        for row in [r for r in modes if r['seed']==seed]:
            n=row['n']
            for mode in row['modes']:
                reg=np.frombuffer(bytes.fromhex(mode['pattern']),np.uint8).reshape(4,n)
                distances=abs(codes[:,:,:n].astype(np.int16)-reg.astype(np.int16)).sum(axis=(1,2))
                minima={str(size):int(distances[:size+1].min()) for size in p['prior_sizes']}
                for size,value in minima.items():
                    assert value>=mode['prior_edit_distance_lower_bounds'][size];checks['mode_distance_bounds']+=1
                    if value!=mode['prior_edit_distance_lower_bounds'][size]:differences.append(dict(seed=seed,n=n,pattern=mode['pattern'],size=size,exact=value,lower=mode['prior_edit_distance_lower_bounds'][size]))
                if mode['certified_outside_two_edit_full16384']:
                    assert minima['16384']>2;checks['strict_two_edit_certificates']+=1
                    certificates.append(dict(seed=seed,n=n,pattern=mode['pattern'],exact_minimum_edits=minima,found_by=mode['found_by']))
                b=np.asarray(mode['exact_witness']['biases']);w=max(exponent(x[:n]),exponent(v[:n]),exponent(b),exponent([.001,.12]))
                scale=1<<w;h=integers(x[:n],w);bias=integers(b,w);observed=integers(v[:n],w);epsilon=integers([.001],w)[0]
                witness_codes=[]
                for shift in bias:
                    z=h+shift;witness_codes.append(np.array([(int(a>=0)+int(a>=scale//2)+int(a>=scale)) for a in z],np.uint8))
                    h=np.maximum(0,scale-abs(2*z-scale))
                assert np.array_equal(np.array(witness_codes),reg)
                err=int(max(abs(h-observed)));assert err<=epsilon
                exact=F(err,scale);record=mode['exact_witness']
                assert exact==F(int(record['max_error_numerator']),int(record['max_error_denominator']))
                assert all(abs(a)<=integers([.12],w)[0] for a in bias);checks['exact_witnesses']+=1
        print(json.dumps(dict(seed=seed,**checks)),flush=True)
    assert checks['exact_witnesses']==summary['checks']['exact_feasible_witnesses']
    result=dict(passed=True,checks=checks,strict_certificates=certificates,conservative_bound_differences=differences,
        source_sha256=sha(Path(__file__)),protocol_sha256=sha(ref/'protocol.json'),modes_sha256=sha(ref/'modes.json'),
        scope='Every supplied prior path exactly recomputed by common-denominator integer arithmetic; finite-search boundary only')
    (ref/'integer_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(dict(passed=True,checks=checks)),flush=True)


if __name__=='__main__':main()
