"""Independent rational cube interiors, all particle draws/predictions, marginal identity."""
import argparse
from collections import Counter
from fractions import Fraction as F
from itertools import product
import json
from pathlib import Path
import numpy as np
import cold_stagnation_switch as cold
import shared_mode_readout as shared
from analyze_recovered_online_comparison import forward
from run_multiplier_fixed_point_screen import sha,dump

def check_cube(x,v,key,proof):
    pattern=np.frombuffer(bytes.fromhex(key),dtype=np.uint8).reshape(4,len(x));point=[shared.exact.unpack(t) for t in proof['point']]
    radius=shared.exact.unpack(proof['minimum_nonconstant_slack'])/64;assert radius>0
    # Check16 rational corners by actual scalar composition, not the
    # construction's constraint coefficients. A fixed-pattern cell is convex.
    for signs in product([-1,1],repeat=4):
        b=[a+s*radius for a,s in zip(point,signs)];assert all(abs(t)<shared.exact.B for t in b);h=[F(float(t)) for t in x]
        for j in range(4):
            for i in range(len(x)):
                z=h[i]+b[j];k=int(pattern[j,i]);lo=None if k==0 else shared.exact.K[k-1];hi=None if k==3 else shared.exact.K[k]
                assert (lo is None or z>lo) and (hi is None or z<hi);h[i]=max(F(0),1-abs(2*z-1))
        assert all(abs(t-F(float(y)))<shared.exact.EPS for t,y in zip(h,v))
    return dict(radius=shared.exact.pack(radius),corners=16)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();src=Path(__file__).parent
    base=root/'results/shared_mode_readout';inp=base/'development';out=base/'audit';out.mkdir(parents=True,exist_ok=True);assert not (out/'summary.json').exists()
    p=json.loads((inp/'protocol.json').read_text());run=json.loads((inp/'summary.json').read_text());assert run['complete']
    for n,h in p['source_sha256'].items():assert sha(src/n)==h,n
    manifest=json.loads((inp/'before_query_manifest.json').read_text());assert manifest['protocol_sha256']==sha(inp/'protocol.json') and manifest['rows_sha256']==sha(inp/'rows.json')
    for group in ['arrays','geometry']:
        for n,h in manifest[group].items():assert sha(inp/n)==h,n
    rows=json.loads((inp/'rows.json').read_text());queries={(r['seed'],r['method'],r['repetition']):r for r in json.loads((inp/'query_rows.json').read_text())}
    checks=Counter();cubes=[];maxprediction=0.;maxsupport=0.
    for seed in p['seeds']:
        gg=json.loads((inp/f'{seed}_geometry.json').read_text());polys={};anyarray=np.load(inp/next(r['file'] for r in rows if r['seed']==seed));x=anyarray['x'];v=anyarray['v'];q=anyarray['q']
        for key,r in gg['regions'].items():
            if 'file' not in r:continue
            assert sha(inp/r['file'])==r['sha256'];z=np.load(inp/r['file']);poly={n:z[n] for n in z.files};polys[key]=poly
            cube=check_cube(x,v,key,r['proof']);cubes.append(dict(seed=seed,key=key,**cube));checks['positive_volume_rational_cubes']+=1;checks['exact_corners']+=16
            facets=poly['facets'];weights=np.abs(np.linalg.det(facets-poly['interior']))/24;total=weights.sum()*np.prod(poly['scale'])
            assert np.isclose(total,float(poly['volume']),rtol=1e-12,atol=1e-24);checks['volume_replays']+=1
        truth=forward(q,np.random.default_rng(seed).uniform(-.12,.12,(1,4)))[0]
        for row in [r for r in rows if r['seed']==seed]:
            a=np.load(inp/row['file']);assert sha(inp/row['file'])==row['sha256'];keys=row['mode_keys']
            if keys:
                points,allocation=shared.draw(polys,keys,p['sample_count'],np.random.default_rng(np.random.SeedSequence([p['repetition_seeds'][row['repetition']],seed,2048])))
                assert points.tobytes()==a['points'].tobytes() and allocation.tobytes()==a['allocation'].tobytes();checks['particle_draws']+=1
                support=float(np.max(abs(forward(x,points)-v)));assert support<=.001+1e-7;maxsupport=max(maxsupport,support);checks['support_particles']+=len(points)
            else:
                original=np.load(root/'results/cold_stagnation_switch/development'/f'{seed}_{row["method"]}.npz');assert a['points'][0].tobytes()==original['selected_b'].tobytes();checks['unchanged_fallbacks']+=1
            pred=forward(q,a['points']).mean(0);gap=float(np.max(abs(pred-a['prediction'])));maxprediction=max(maxprediction,gap);assert gap<1e-12
            risk=float(np.mean((a['prediction']-truth)**2));assert risk==queries[seed,row['method'],row['repetition']]['mse'];checks['predictions_and_risks']+=1
    marginal=json.loads((inp/'marginal_rows.json').read_text());evaluations=json.loads((inp/'marginal_evaluation.json').read_text())
    for row,e in zip(marginal,evaluations):
        a=np.load(inp/row['file']);q=a['q'];means=[]
        for i,r in enumerate(row['pointfiles']):
            z=np.load(inp/r['file']);assert sha(inp/r['file'])==r['sha256'];polyref=json.loads((inp/f'{row["seed"]}_geometry.json').read_text())['regions'][r['key']]['file'];poly=np.load(inp/polyref)
            points=shared.geometry.sample(poly,p['marginal_per_mode_count'],np.random.default_rng(np.random.SeedSequence([p['repetition_seeds'][row['repetition']],row['seed'],i,8192])))
            assert points.tobytes()==z['points'].tobytes();pred=forward(q,points).mean(0);assert np.max(abs(pred-z['prediction']))<1e-12
            means.append(z['prediction']);checks['marginal_mode_draws']+=1
        assert np.array(means).tobytes()==a['mode_prediction'].tobytes();mask=np.array([r['key'] in row['new_keys'] for r in row['pointfiles']]);w=float(a['weights'][mask].sum()/a['weights'].sum())
        prior=np.average(a['mode_prediction'][~mask],weights=a['weights'][~mask],axis=0);new=np.average(a['mode_prediction'][mask],weights=a['weights'][mask],axis=0)
        assert prior.tobytes()==a['prior'].tobytes() and new.tobytes()==a['new'].tobytes() and abs(w-float(a['w']))<1e-15
        truth=forward(q,np.random.default_rng(row['seed']).uniform(-.12,.12,(1,4)))[0];cross=2*w*np.mean((prior-truth)*(new-prior));square=w*w*np.mean((new-prior)**2)
        assert abs(cross-e['cross_term'])<1e-14 and abs(square-e['positive_square_term'])<1e-14 and abs(cross+square-e['difference'])<1e-14;checks['marginal_risk_identities']+=1
    dump(out/'rational_cubes.json',cubes)
    result=dict(passed=True,source_sha256=sha(Path(__file__)),protocol_sha256=sha(inp/'protocol.json'),run_summary_sha256=sha(inp/'summary.json'),checks=checks,
        max_independent_prediction_difference=maxprediction,max_particle_support_error=maxsupport,cubes_sha256=sha(out/'rational_cubes.json'))
    dump(out/'summary.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
