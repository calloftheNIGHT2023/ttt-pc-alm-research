"""325 independent scalar risk and main-selector fixed-value witness replay."""
import argparse
from collections import Counter,defaultdict
from fractions import Fraction as F
from pathlib import Path
import math
import time
import numpy as np
from branch_image_chain_v1 import fixed_value
from evaluate_complete_credit_mode_geometry_v1 import read,save,sha


def run(root,out):
    start=time.perf_counter();base=root/'results/support_consistency_trigger';folder=base/'exclusive_value_v1';ss=read(folder/'summary.json');assert ss['passed']
    for n,d in ss['outputs_sha256'].items():assert sha(folder/n)==d
    scores=read(folder/'scores.json');lookup={(r['seed'],r['method'],r['grid']):r for r in scores}
    mom=root/'results/confirmation_conditional_risk/moments';index={r['seed']:r for r in read(mom/'tasks.json')}
    counts=Counter();maxgap=0.
    def close(a,b):
        nonlocal maxgap
        gap=abs(a-b);maxgap=max(maxgap,gap);assert gap<2e-13;counts['numeric_fields']+=1
    for t in read(folder/'pools.json'):
        seed=t['seed'];g=read(base/'geometry_v1'/f'{seed}_geometry.json')
        controls=set(g['prior_positive_modes'])
        for name,value in g['methods'].items():
            if name.split('/')[1] not in ['dual','dual_plus_residual']:controls.update(value['positive_modes'])
        assert sorted(controls)==t['control'];mt=index[seed];p=mom/mt['file'];assert sha(p)==mt['sha256']
        with np.load(p,allow_pickle=False) as z:vol,means=z['volumes'],z['means']
        keys=mt['keys'];w=[float(v)/math.fsum(map(float,vol)) for v in vol]
        def mixture(pool):
            ids=[i for i,k in enumerate(keys) if k in pool];mass=math.fsum(w[i] for i in ids)
            return mass,np.array([[math.fsum(w[i]*float(means[i,b,q]) for i in ids)/mass for q in range(257)] for b in range(4)])
        _,full=mixture(set(keys));cmass,cmu=mixture(controls)
        for name,pool in t['candidates'].items():
            assert set(pool)==controls|set(g['methods'][name]['positive_modes']);mass,mu=mixture(set(pool))
            for grid,qs in [(257,range(257)),(129,range(0,257,2))]:
                s=lookup[seed,name,grid];assert s['added_modes']==sorted(set(pool)-controls);close(mass-cmass,s['added_mass'])
                differences=[]
                for a,b in [(0,1),(2,3)]:
                    before=math.fsum(float(cmu[a,q]-full[a,q])*float(cmu[b,q]-full[b,q]) for q in qs)/grid
                    after=math.fsum(float(mu[a,q]-full[a,q])*float(mu[b,q]-full[b,q]) for q in qs)/grid
                    differences.append(after-before)
                for got,want in zip(differences,s['delta_pairs']):close(got,want)
                close(math.fsum(differences)/2,s['delta']);counts['risk_rows']+=1
        counts['tasks']+=1
    for w in read(folder/'witnesses.json'):
        policy=w['method'].split('/')[0];rec=read(base/'development_v1'/f"{w['seed']}_{policy}.json")
        pattern=np.array(list(bytes.fromhex(w['mode']))).reshape(4,4)
        for channel,note in w['credits'].items():
            value=fixed_value(np.array(rec['x_observed']),np.array(rec['v_observed']),np.array(rec['credits'][channel]),pattern)
            assert value==F(note['lower']) and value<=0;result=rec['results'][channel]
            match=[p for p in result['proposals'] if p['mode']==w['mode']]
            if note['status']=='selected':assert len(match)==1 and match[0]['rank']==note['selected_rank']
            elif note['status']=='ranked_out':
                assert not match and (value,w['mode'])>(F(note['cutoff_lower']),note['cutoff_mode'])
            else:assert note['status']=='outside_window' and not match
            counts['independent_fixed_credit_bounds']+=1
    groups=defaultdict(list)
    for r in scores:groups[r['method'],r['grid']].append(r)
    for a in ss['aggregate']:
        group=groups[a['method'],a['grid']];top=max(group,key=lambda r:abs(r['delta']));denom=math.fsum(abs(r['delta']) for r in group)
        close(math.fsum(r['delta'] for r in group)/64,a['mean_delta'])
        close(abs(top['delta'])/denom if denom else 0.,a['largest_absolute_contribution_fraction'])
        close(math.fsum(r['delta'] for r in group if r['seed']!=top['seed'])/63,a['leave_largest_out_mean'])
        assert a['added_mode_pairs']==sum(len(r['added_modes']) for r in group)
    result=dict(passed=True,counts=dict(counts),max_numeric_gap=maxgap,seconds=time.perf_counter()-start,
                input_summary_sha256=sha(folder/'summary.json'),source_sha256=sha(Path(__file__)),query_targets_accessed=False,
                resources_matched=False,independent_task_gain_established=False)
    save(out/'summary.json',result);print(result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    run(Path(__file__).resolve().parents[2],a.out)
