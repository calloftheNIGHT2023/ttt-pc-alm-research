"""277: exact residual-to-forward credit transfer on2112 frozen support states."""
import argparse,json,time
from collections import Counter
from fractions import Fraction as F
from pathlib import Path
import numpy as np
import cold_stagnation_switch as cold
import local_dual_jump_short as jump
import minimum_sufficient_dual as minimum
import local_dual_jump_transfer as transfer
from posterior_confirmation_pipeline import discovery_box
from audit_local_dual_jump_modes import modes
from run_multiplier_fixed_point_screen import sha,dump


def frac(x):return F(float(x))
def g(z):return max(F(0),F(1)-abs(2*z-1))
def branch(z):return sum(z>=t for t in [F(0),F(1,2),F(1)])
def pack(x):return str(x)


def exact_measure(x,v,b,h,event,positive):
    bb=list(map(frac,b));hh=[[frac(z) for z in layer] for layer in h];xx=list(map(frac,x));vv=list(map(frac,v))
    actual=xx[:];previous=xx[:];bounds=[F(0)]*len(x);residuals=[];errors=[];errorbounds=[];preacts=[];codes=[]
    for l in range(4):
        p=[z+bb[l] for z in actual];actual=[g(z) for z in p];codes.extend(branch(z) for z in p);preacts.append(p)
        residual=[hh[l][i]-g(previous[i]+bb[l]) for i in range(len(x))];residuals.append(residual)
        bounds=[abs(residual[i])+2*bounds[i] for i in range(len(x))];err=[abs(actual[i]-hh[l][i]) for i in range(len(x))]
        assert all(a<=c for a,c in zip(err,bounds));errors.append(err);errorbounds.append(bounds[:]);previous=hh[l]
    exact_key=bytes(codes).hex();float_key=next(iter(modes(x,b[None])));support_error=max(abs(a-c) for a,c in zip(actual,vv))
    result=dict(residuals=[[pack(z) for z in r] for r in residuals],actual_errors=[[pack(z) for z in r] for r in errors],
        error_bounds=[[pack(z) for z in r] for r in errorbounds],exact_mode=exact_key,float_mode=float_key,
        exact_float_mode_equal=exact_key==float_key,positive_mode=exact_key in positive,float_positive_mode=float_key in positive,
        atomic_support_max_error=pack(support_error),atomic_support_feasible=support_error<=frac(.001),branch_event=None)
    if event is not None:
        j,i,k=event['j'],event['i'],event['k'];center=hh[j][i]+bb[j+1];radius=errorbounds[j][i];lo=center-radius;hi=center+radius
        lower=[None,F(0),F(1,2),F(1)][k];upper=[F(0),F(1,2),F(1),None][k]
        certificate=(lower is None or lo>lower) and (upper is None or hi<upper);true=preacts[j+1][i];hit=branch(true)==k
        assert not certificate or hit
        result['branch_event']=dict(layer=j,observation=i,target=k,center=pack(center),radius=pack(radius),interval=[pack(lo),pack(hi)],
            true_preactivation=pack(true),true_branch=branch(true),split_branch=branch(center),split_target_hit=branch(center)==k,
            actual_target_hit=hit,strict_certificate=certificate,actual_error=pack(errors[j][i]))
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();src=Path(__file__).parent
    out=root/'results/forward_credit_transfer/diagnosis';out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    parent=root/'results/round_276_audit.json';pa=json.loads(parent.read_text());assert pa['passed'];hashes=dict(pa['source_sha256']);hashes[Path(__file__).name]=sha(Path(__file__))
    for n,h in hashes.items():assert sha(src/n)==h,n
    design='277_forward_credit_transfer_protocol.md';assert sha(root/'outputs/ttt-pc-alm-research'/design)==pa['report_sha256'][design]
    directories=dict(D=root/'results/distinct_prior_credit/development',A=root/'results/anchor_preserving_fork/development',rootD=root/'results/matched_budget_confirmation/conditioned_confirmation')
    records={key:{(r['seed'],r['method']):r for r in json.loads((path/'rows.json').read_text())} for key,path in directories.items()}
    reference=root/'results/confirmation_conditional_risk/reference';refs={r['seed']:r for r in json.loads((reference/'coverage.json').read_text())}
    p=dict(source_sha256=hashes,parent_audit_sha256=sha(parent),design_sha256=pa['report_sha256'][design],seeds=list(range(5910000,5910064)),
        source_rows_sha256={k:sha(d/'rows.json') for k,d in directories.items()},reference_coverage_sha256=sha(reference/'coverage.json'),
        phase_accesses_query_targets=False,phase_accesses_posterior_moments=False,scope='Support-only exact binary-rational transfer diagnosis, not a new online method or risk result')
    dump(out/'protocol.json',p);rows=[];files={};counts=Counter();groups=Counter();begin=time.perf_counter()
    starts=np.r_[np.zeros((1,4)),np.random.default_rng(731).uniform(-.12,.12,(32,4))]
    with discovery_box(.12):
        for seed in p['seeds']:
            atoms={};sourcefiles={}
            for label,name in [('D','distinct_D32_A64'),('A','anchor_A33_32_A64'),('rootD','minimum_dual_alm64')]:
                r=records[label][seed,name];path=directories[label]/r['file'];assert sha(path)==r['sha256'];sourcefiles[label]=dict(path=str(path.relative_to(root)),sha256=r['sha256'])
                with np.load(path) as z:
                    if label=='A':x=z['x_observed'].copy();v=z['v_observed'].copy()
                    atoms[label]={k:z['initial_'+k].copy() for k in ['b','h','u']}
            for key in ['b','h','u']:
                if key=='b':atoms['D'][key][0]=atoms['rootD'][key][0]
                else:atoms['D'][key][:,0]=atoms['rootD'][key][:,0]
            assert atoms['D']['h'].tobytes()==atoms['A']['h'].tobytes()
            ref=refs[seed];assert sha(reference/ref['file'])==ref['sha256'];data=json.loads((reference/ref['file']).read_text());positive={r['pattern'] for r in data['reference']['positive_regions']}
            assert np.array(data['x_observed']).tobytes()==x.tobytes() and np.array(data['v_observed']).tobytes()==v.tobytes()
            prep=cold.Local(starts,x,v,'nodual')
            for _ in range(16):prep.step()
            for r in range(33):
                scan=jump.scan(x,prep.b[r],prep.h[:,r]);event=minimum.minimum_event(x,prep.b[r],prep.h[:,r],scan['selected'])['selected'];result={}
                for action,method in [('D','dual_jump'),('A','activity_only')]:
                    recreated,_=transfer.run(prep.b[r],prep.h[:,r],prep.best[r],x,v,event,method)
                    for key in ['b','h','u']:
                        saved=atoms[action][key][r] if key=='b' else atoms[action][key][:,r]
                        assert recreated[key].tobytes()==saved.tobytes(),(seed,r,action,key);counts['exact_atomic_arrays']+=1
                    result[action]=exact_measure(x,v,recreated['b'],recreated['h'],event,positive);counts['exact_forward_states']+=1;counts['layer_observation_bounds']+=4*len(x)
                    for field in ['positive_mode','float_positive_mode','atomic_support_feasible','exact_float_mode_equal']:counts[action+'_'+field]+=int(result[action][field])
                    if event is not None:
                        for field in ['split_target_hit','actual_target_hit','strict_certificate']:counts[action+'_'+field]+=int(result[action]['branch_event'][field])
                if event is None:counts['states_without_event']+=1
                else:
                    counts['states_with_event']+=1
                    for field in ['actual_target_hit','strict_certificate']:
                        left=result['D']['branch_event'][field];right=result['A']['branch_event'][field];groups[field+'_'+('both' if left and right else 'only_D' if left else 'only_A' if right else 'neither')]+=1
                for field in ['positive_mode','atomic_support_feasible']:
                    left=result['D'][field];right=result['A'][field];groups[field+'_'+('both' if left and right else 'only_D' if left else 'only_A' if right else 'neither')]+=1
                rows.append(dict(seed=seed,restart=r,event=event,source_files=sourcefiles,**result))
            filename=f'{seed}_atomic_states.npz';np.savez_compressed(out/filename,x=x,v=v,D_b=atoms['D']['b'],D_h=atoms['D']['h'],D_u=atoms['D']['u'],A_b=atoms['A']['b'],A_h=atoms['A']['h'],A_u=atoms['A']['u']);files[filename]=sha(out/filename)
            if (seed-p['seeds'][0]+1)%8==0:print(json.dumps(dict(tasks=seed-p['seeds'][0]+1,total=64,seconds=time.perf_counter()-begin)),flush=True)
    assert len(rows)==2112 and counts['exact_forward_states']==4224
    for n,data in [('rows.json',rows),('files.json',files)]:dump(out/n,data)
    ans=dict(passed=True,counts=counts,groups=groups,states=len(rows),seconds=time.perf_counter()-begin,
        outputs_sha256={n:sha(out/n) for n in ['protocol.json','rows.json','files.json']},phase_accesses_query_targets=False,phase_accesses_posterior_moments=False,
        next='Independent rational branch/bound audit, then classify true transfer loss versus loose sufficient bounds before proposing any intervention')
    dump(out/'summary.json',ans);print(json.dumps(ans),flush=True)


if __name__=='__main__':main()
