"""372 archive audit: independent rows, every certificate, fixed sample selection."""
from collections import Counter,defaultdict
from fractions import Fraction as F
from pathlib import Path
import math,time
import numpy as np
from budget_reinvestment_suite_v1 import read,sha,save,complete
import prefix_obstruction_v1 as model
import run_prefix_obstruction_v2 as runner


def independent_rows(x,v,reg):
    # Observation-major scalar arithmetic; no candidate matrix/box functions.
    a=[];rhs=[];d,n=reg.shape;eps=np.nextafter(.001+.000001,np.inf)
    for i in range(n):
        c=F(float(x[i]));coeff=[F(0)]*d
        for j in range(d):
            coeff=coeff.copy();coeff[j]+=1;code=int(reg[j,i])
            if code<3:a.append(coeff.copy());rhs.append([F(0),F(1,2),F(1)][code]-c)
            if code>0:a.append([-t for t in coeff]);rhs.append(c-[F(0),F(1,2),F(1)][code-1])
            slope=[0,2,-2,0][code];c=slope*c+[0,0,2,0][code];coeff=[slope*t for t in coeff]
        lower=F(float(max(0.,np.nextafter(v[i]-eps,-np.inf))));upper=F(float(min(1.,np.nextafter(v[i]+eps,np.inf))))
        a.extend([coeff.copy(),[-t for t in coeff]]);rhs.extend([upper-c,c-lower])
    for j in range(d):
        row=[F(int(t==j)) for t in range(d)];a.extend([row,[-t for t in row]]);rhs.extend([F(.12),F(.12)])
    return a,rhs


def canonical(a,r):return Counter((tuple(row),rhs) for row,rhs in zip(a,r))


def run(root,out):
    begin=time.perf_counter();base=root/runner.BASE;summary=complete(base/'development_v2');complete(base/'selection_v2')
    p=read(base/'selection_v2/protocol.json');assert p['source_sha256']==runner.hashes(root)
    rows=read(base/'development_v2/rows.json');seal=read(base/'development_v2/screen_seal.json');assert seal['all_screens_completed'] and not seal['lp_labels_accessed']
    by={(r['seed'],r['method']):r for r in rows};counts=Counter();aggregate=defaultdict(lambda:dict(cases=0,samples=0,infeasible=0,positive=0,zero=0,unresolved=0,rates=[],c100=0,pdhg=0,c100_seconds=0.,pdhg_seconds=0.))
    for c in p['cases']:
        row=by[c['seed'],c['method']];source=root/c['source_directory'];selection=root/c['directory']/'selection.npz';assert sha(selection)==c['selection_sha256']
        for file,h in c['source_files'].items():assert sha(source/file)==h
        meta=read(source/'metadata.json');assert meta['complete_stages']==c['k'] and meta['completed']==c['source_completed']
        with np.load(source/'arrays.npz',allow_pickle=False) as z, np.load(selection,allow_pickle=False) as s:
            pool=z['prefix_'+str(c['k'])];indices=np.sort(np.random.default_rng(np.random.SeedSequence([372171,c['seed'],runner.METHODS.index(c['method'])])).choice(len(pool),min(64,len(pool)),replace=False))
            assert np.array_equal(s['source_indices'],indices) and np.array_equal(pool[indices],s['regions'])
            assert np.array_equal(s['observation_ids'],z['support_order'][:c['k']])
            assert np.array_equal(s['x'],z['x_observed'][:c['k']]) and np.array_equal(s['v'],z['v_observed'][:c['k']])
            x=s['x'];v=s['v'];regs=s['regions']
        directory=root/row['directory'];classification=read(directory/'classification.json');sm=read(directory/'screens.json')
        with np.load(directory/'screens.npz',allow_pickle=False) as z:maskc=z['c100_reject'];maskp=z['pdhg128_reject']
        assert row['samples']==len(regs)==len(classification) and row['source_count']==len(pool)
        expected=Counter();screen_counts=Counter()
        for index,(reg,result) in enumerate(zip(regs,classification)):
            assert result['sample_index']==index
            a,r=model.matrices(x,v,reg);aa,rr=independent_rows(x,v,reg);assert canonical(a,r)==canonical(aa,rr);counts['independent_constraint_rows']+=len(a)
            counts['exact_certificates']+=model.proof.verify_certificates(a,r,result)
            status=result['classification'];expected[status]+=1
            if status=='positive_volume':
                cert=next(c for c in result['certificates'] if c['type']=='exact_interior_cube');point=[F(t) for t in cert['point']]
                assert all(sum((b*q for b,q in zip(point,line)),F(0))<=rhs for line,rhs in zip(aa,rr));counts['independent_positive_points']+=1
            if maskc[index]:screen_counts['c100_'+status]+=1;assert status!='positive_volume'
            if maskp[index]:
                cert=sm['pdhg']['certificates'][str(index)];val=model.exact_box_bound(x,v,reg,np.array(cert['p']),np.array(cert['a']))
                assert F(cert['lower'])<=val and val>0 and status!='positive_volume';counts['pdhg_wide_domain_exact']+=1
                screen_counts['pdhg_'+status]+=1
        assert dict(expected)==row['counts'] and dict(screen_counts)==row['screening_counts']
        assert int(maskc.sum())==row['c100_rejected'] and int(maskp.sum())==row['pdhg_rejected'];counts['sampled_frontiers']+=1
        key=(c['method'],'complete' if c['source_completed'] else 'budget_exhausted');g=aggregate[key]
        g['cases']+=1;g['samples']+=len(regs);g['infeasible']+=expected['infeasible'];g['positive']+=expected['positive_volume'];g['zero']+=expected['zero_volume_or_empty'];g['unresolved']+=expected['unresolved']
        g['rates'].append(expected['infeasible']/len(regs));g['c100']+=int(maskc.sum());g['pdhg']+=int(maskp.sum());g['c100_seconds']+=row['c100_seconds'];g['pdhg_seconds']+=row['pdhg_seconds']
    groups=[]
    for (method,status),g in aggregate.items():
        rates=g.pop('rates');groups.append(dict(method=method,source_status=status,**g,mean_frontier_infeasible_fraction=math.fsum(rates)/len(rates)))
    failed=[r for r in rows if not r['source_completed']]
    mean=math.fsum(r['counts'].get('infeasible',0)/r['samples'] for r in failed)/len(failed)
    save(out/'groups.json',groups)
    result=dict(passed=True,checks=dict(counts),seconds=time.perf_counter()-begin,budget_frontiers=len(failed),
        mean_budget_frontier_infeasible_fraction=mean,predeclared_75_percent_obstruction_gate=mean>=.75,
        source_sha256=sha(Path(__file__)),development_summary_sha256=sha(base/'development_v2/summary.json'),query_targets_accessed=False,
        outputs_sha256={'groups.json':sha(out/'groups.json')})
    save(out/'summary.json',result);print(result,flush=True)


if __name__=='__main__':
    root=Path(__file__).resolve().parents[2];out=root/runner.BASE/'audit_v2';out.mkdir(parents=True,exist_ok=False);run(root,out)
