"""279 fixed132-state exact response-path primitive; no query targets."""
import argparse,json,time,os
from collections import Counter
from pathlib import Path
from fractions import Fraction as F
import numpy as np
import actual_credit_response as curve
import cold_stagnation_switch as cold
from posterior_confirmation_pipeline import discovery_box
from run_multiplier_fixed_point_screen import sha,dump


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();src=Path(__file__).parent
    out=root/'results/actual_credit_response/primitive';out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    assert os.environ.get('OPENBLAS_NUM_THREADS')==os.environ.get('OMP_NUM_THREADS')=='1'
    parent=root/'results/round_278_audit.json';pa=json.loads(parent.read_text());assert pa['passed'];hashes=dict(pa['source_sha256'])
    for name in ['actual_credit_response.py',Path(__file__).name]:hashes[name]=sha(src/name)
    for n,h in hashes.items():assert sha(src/n)==h,n
    design='279_actual_credit_response_protocol.md';assert sha(root/'outputs/ttt-pc-alm-research'/design)==pa['report_sha256'][design]
    unit=curve.verify();assert unit['passed']
    original=root/'results/forward_credit_transfer/diagnosis';osummary=json.loads((original/'summary.json').read_text());assert osummary['passed']
    assert sha(original/'files.json')==osummary['outputs_sha256']['files.json'];originalfiles=json.loads((original/'files.json').read_text())
    reference=root/'results/confirmation_conditional_risk/reference';refs={r['seed']:r for r in json.loads((reference/'coverage.json').read_text())}
    p=dict(source_sha256=hashes,parent_audit_sha256=sha(parent),design_sha256=pa['report_sha256'][design],seeds=[5910000,5910001,5910053,5910063],restarts=33,
        maximum_layer_intersection_tests=10000,maximum_state_forward_segments=10000,endpoint_tolerance=1e-10,dense_grid_points=257,
        reference_files_sha256=sha(reference/'coverage.json'),original_files_sha256=sha(original/'files.json'),unit_tests=unit,
        phase_accesses_query_targets=False,phase_accesses_posterior_moments=False,
        scope='Mathematical canonical argmin response on certified open cells; isolated breakpoints excluded, exact and frozen endpoints included equally in both pools; not an online performance claim')
    dump(out/'protocol.json',p);rows=[];counts=Counter();begin=time.perf_counter();files={};maxendpoint=0.;maxdense=0.
    starts=np.r_[np.zeros((1,4)),np.random.default_rng(731).uniform(-.12,.12,(32,4))]
    with discovery_box(.12):
        for seed in p['seeds']:
            filename=f'{seed}_atomic_states.npz';assert sha(original/filename)==originalfiles[filename]
            with np.load(original/filename) as z:x=z['x'].copy();v=z['v'].copy();hbank=z['D_h'].copy();ubank=z['D_u'].copy();bD=z['D_b'].copy();bA=z['A_b'].copy()
            ref=refs[seed];assert sha(reference/ref['file'])==ref['sha256'];positive={r['pattern'] for r in json.loads((reference/ref['file']).read_text())['reference']['positive_regions']}
            prep=cold.Local(starts,x,v,'nodual')
            for _ in range(16):prep.step()
            for r in range(33):
                start=time.perf_counter();row=dict(seed=seed,restart=r,status='pending');h=hbank[:,r];u=ubank[:,r];old=prep.b[r]
                try:
                    layers=[curve.envelope(x if j==0 else h[j-1],h[j],u[j],old[j]) for j in range(4)]
                    endpoints=[];endpoint_ties=0;endpoint_gap=0.;canonical_gap=0.
                    for t,saved in [(F(0),bA[r]),(F(1),bD[r])]:
                        exact=[]
                        for j,layer in enumerate(layers):
                            value,alternatives=curve.at(layer,t);exact.append(value);endpoint_ties+=len(alternatives)>1
                            distance=min(abs(float(z)-saved[j]) for z in alternatives);endpoint_gap=max(endpoint_gap,distance);canonical_gap=max(canonical_gap,abs(float(value)-saved[j]));assert distance<=p['endpoint_tolerance'],(seed,r,j,t,distance)
                        endpoints.append(exact)
                    maxendpoint=max(maxendpoint,endpoint_gap);response=curve.response_path(x,layers);chordbias=[(a,b-a) for a,b in zip(*endpoints)]
                    chord=curve.true_forward_piece(x,chordbias,curve.rational(0),curve.rational(1))
                    endpoint_modes={curve.exact_mode(x,b) for b in endpoints}|{curve.exact_mode(x,list(map(curve.fraction,b))) for b in [bA[r],bD[r]]}
                    response_modes={s['mode'] for s in response}|endpoint_modes;chord_modes={s['mode'] for s in chord}|endpoint_modes
                    response_cuts=curve.ordered([z for s in response for z in [s['lo'],s['hi']]]);chord_cuts=curve.ordered([z for s in chord for z in [s['lo'],s['hi']]])
                    dense_gap=0.;dense_ties=0;boundary_probes=0;mode_probes=0
                    tt=np.linspace(0,1,257);bb=[]
                    for j in range(4):
                        prev=x if j==0 else h[j-1];target=h[j][None]+tt[:,None]*u[j][None]
                        bb.append(cold.base.bias_solve(np.tile(prev,(257,1)),target,np.full(257,old[j]),0.,float('inf'),.01))
                    floatbias=np.stack(bb,axis=1)
                    for ti,tfloat in enumerate(tt):
                        t=curve.fraction(tfloat);params=[]
                        for j,layer in enumerate(layers):
                            val,alternatives=curve.at(layer,t);params.append(val);gap=min(abs(float(z)-floatbias[ti,j]) for z in alternatives);dense_gap=max(dense_gap,gap);dense_ties+=len(alternatives)>1
                            assert gap<=p['endpoint_tolerance'],(seed,r,j,t,gap)
                        mode=curve.exact_mode(x,params)
                        if any(z.compare_rational(t)==0 for z in response_cuts):boundary_probes+=1
                        else:assert mode in response_modes;mode_probes+=1
                        params=[a+b*t for a,b in chordbias];mode=curve.exact_mode(x,params)
                        if not any(z.compare_rational(t)==0 for z in chord_cuts):assert mode in chord_modes;mode_probes+=1
                    maxdense=max(maxdense,dense_gap)
                    output=dict(seed=seed,restart=r,x=x.tolist(),v=v.tolist(),old_bias=old.tolist(),activity=h.tolist(),dual=u.tolist(),frozen_A=bA[r].tolist(),frozen_D=bD[r].tolist(),
                        layers=layers,mathematical_endpoints=endpoints,response_segments=response,chord_segments=chord,
                        isolated_response_breakpoints=response_cuts,isolated_chord_breakpoints=chord_cuts,endpoint_modes=sorted(endpoint_modes),
                        response_modes=sorted(response_modes),chord_modes=sorted(chord_modes))
                    fn=f'{seed}_r{r:02d}_curve.json';dump(out/fn,curve.encode(output));files[fn]=sha(out/fn)
                    rp=response_modes&positive;cp=chord_modes&positive;ep=endpoint_modes&positive
                    row.update(status='complete',file=fn,sha256=files[fn],endpoint_maximum_gap=endpoint_gap,canonical_endpoint_maximum_gap=canonical_gap,endpoint_nonunique_layers=endpoint_ties,
                        dense_maximum_gap=dense_gap,dense_nonunique_layers=dense_ties,dense_boundary_probes=boundary_probes,dense_open_mode_probes=mode_probes,
                        candidates=sum(len(l['candidates']) for l in layers),intersection_tests=sum(l['pair_tests'] for l in layers),envelope_cells=sum(len(l['cells']) for l in layers),
                        envelope_inequality_checks=sum(l['inequality_checks'] for l in layers),response_segments=len(response),chord_segments=len(chord),
                        endpoint_positive=sorted(ep),response_positive=sorted(rp),chord_positive=sorted(cp),response_only_positive=sorted(rp-cp),chord_only_positive=sorted(cp-rp))
                    for key in ['candidates','intersection_tests','envelope_cells','envelope_inequality_checks','response_segments','chord_segments','dense_open_mode_probes','dense_boundary_probes','endpoint_nonunique_layers']:counts[key]+=row[key]
                    counts['complete_states']+=1
                except (ArithmeticError,OverflowError) as exc:
                    row.update(status='unresolved',error_type=type(exc).__name__,error=str(exc));counts['unresolved_states']+=1
                row['diagnostic_seconds']=time.perf_counter()-start;rows.append(row);dump(out/'rows.json',rows)
                if len(rows)%11==0:print(json.dumps(dict(states=len(rows),total=132,complete=counts['complete_states'],unresolved=counts['unresolved_states'],seconds=time.perf_counter()-begin)),flush=True)
    dump(out/'files.json',files);ans=dict(passed=counts['unresolved_states']==0,counts=counts,states=len(rows),maximum_endpoint_gap=maxendpoint,maximum_dense_gap=maxdense,seconds=time.perf_counter()-begin,
        outputs_sha256={n:sha(out/n) for n in ['protocol.json','rows.json','files.json']},phase_accesses_query_targets=False,
        next='Independent direct-cost interpolation, global minimum certificates and full-forward interval audit; no expansion to2112 before mechanism review')
    dump(out/'summary.json',ans);print(json.dumps(ans),flush=True)


if __name__=='__main__':main()
