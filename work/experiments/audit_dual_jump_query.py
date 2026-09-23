"""256 independent prediction/geometry checks, including all frozen references."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
import shared_mode_readout as shared
from audit_shared_mode_readout import check_cube
from analyze_recovered_online_comparison import forward
from run_multiplier_fixed_point_screen import sha,dump

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();src=Path(__file__).parent
    base=root/'results/dual_jump_query';inp=base/'development';out=base/'audit';out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    p=json.loads((inp/'protocol.json').read_text());run=json.loads((inp/'summary.json').read_text());assert run['passed'] and run['complete']
    for name,h in p['source_sha256'].items():assert sha(src/name)==h,name
    hashes={**p['source_sha256'],Path(__file__).name:sha(Path(__file__))};dump(out/'protocol.json',dict(source_sha256=hashes,run_summary_sha256=sha(inp/'summary.json')))
    before=json.loads((inp/'before_query_manifest.json').read_text());assert sha(inp/'before_query_manifest.json')==run['before_query_manifest_sha256']
    for name in ['protocol','rows','reference_rows','representative_files','geometry_files']:assert sha(inp/f'{name}.json')==before[f'{name}_sha256']
    for group in ['arrays','geometry']:
        for name,h in before[group].items():assert sha(inp/name)==h,name
    assert sha(inp/'query_rows.json')==run['query_rows_sha256'];rows=json.loads((inp/'rows.json').read_text());references=json.loads((inp/'reference_rows.json').read_text())
    query={(r['seed'],r['method'],r['readout'],r['repetition']):r for r in json.loads((inp/'query_rows.json').read_text())}
    counts=Counter();cubes=[];max_prediction_gap=0.;max_support_error=0.;max_risk_gap=0.
    for seed in p['seeds']:
        with np.load(inp/f'point_{seed}_{p["primary"]}.npz') as a:x=a['x'];v=a['v'];q=a['q']
        geometry=json.loads((inp/f'{seed}_geometry.json').read_text());polys={}
        for key,row in geometry['regions'].items():
            if 'file' not in row:continue
            with np.load(inp/row['file']) as a:poly={n:a[n] for n in a.files}
            polys[key]=poly;cube=check_cube(x,v,key,row['proof']);cubes.append(dict(seed=seed,key=key,**cube));counts['exact_corners']+=16;counts['positive_volume_cubes']+=1
            weights=np.abs(np.linalg.det(poly['facets']-poly['interior']))/24;volume=weights.sum()*np.prod(poly['scale'])
            assert np.isclose(volume,float(poly['volume']),rtol=1e-12,atol=1e-24);counts['volume_replays']+=1
        truth=forward(q,np.random.default_rng(seed).uniform(-.12,.12,(1,4)))[0]
        for row in [r for r in rows+references if r['seed']==seed]:
            ref=row['method'].startswith('cold__');path=root/row['file'] if ref else inp/row['file'];assert sha(path)==row['sha256']
            with np.load(path) as a:data={n:a[n] for n in a.files if n in ['x','v','q','prediction','points','allocation','b']}
            assert np.array_equal(data['x'],x) and np.array_equal(data['v'],v) and np.array_equal(data['q'],q)
            if not ref:
                if row['readout']=='point':pred=forward(q,data['b'][None])[0];counts['point_predictions']+=1
                else:
                    keys=row['mode_keys']
                    if keys:
                        points,allocation=shared.draw(polys,keys,p['sample_count'],np.random.default_rng(np.random.SeedSequence([p['repetition_seeds'][row['repetition']],seed,2048])))
                        assert points.tobytes()==data['points'].tobytes() and allocation.tobytes()==data['allocation'].tobytes();counts['particle_draws']+=1
                        support=float(np.max(abs(forward(x,points)-v)));max_support_error=max(max_support_error,support);assert support<=.001+1e-7;counts['support_particles']+=len(points)
                    else:
                        with np.load(inp/f'point_{seed}_{row["method"]}.npz') as a:assert data['points'][0].tobytes()==a['b'].tobytes()
                        counts['fallbacks']+=1
                    pred=forward(q,data['points']).mean(0);counts['mode_predictions']+=1
                gap=float(np.max(abs(pred-data['prediction'])));max_prediction_gap=max(max_prediction_gap,gap);assert gap<1e-12
            else:counts['reference_predictors']+=1
            risk=float(np.mean((data['prediction']-truth)**2));gap=abs(risk-query[seed,row['method'],row['readout'],row['repetition']]['mse']);assert gap<1e-14;max_risk_gap=max(max_risk_gap,gap);counts['risk_checks']+=1
        print(json.dumps(dict(seed=seed,checks=counts['risk_checks'])),flush=True)
    # All reused references must retain their old evaluated results too.
    for folder,readout in [('cold_stagnation_switch','point'),('shared_mode_readout','mode')]:
        for row in json.loads((root/f'results/{folder}/development/query_rows.json').read_text()):
            key=(row['seed'],'cold__'+row['method'],readout,row.get('repetition',0))
            if key in query:assert query[key]['mse']==row['mse'];counts['unchanged_reference_risks']+=1
    dump(out/'rational_cubes.json',cubes)
    result=dict(passed=True,counts=counts,max_independent_prediction_gap=max_prediction_gap,max_independent_risk_gap=max_risk_gap,max_particle_support_error=max_support_error,
        protocol_sha256=sha(out/'protocol.json'),rational_cubes_sha256=sha(out/'rational_cubes.json'))
    dump(out/'summary.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
