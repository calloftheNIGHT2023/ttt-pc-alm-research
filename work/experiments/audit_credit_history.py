"""Unchanged trajectory replay, all exact certificates and history expansion."""
import argparse
import hashlib
import json
from fractions import Fraction as F
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
import credit_history_capture as history
import credit_history_expansion as expansion
import common_pool_credit as common
import optimized_branch_dual as cert
import neighbor_mode_memory as neighbor
from audit_joint_credit_minimax import exact_mixture,rational_optimum


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();base=root/'results/credit_history';inp=base/'development'
    p=read(inp/'protocol.json');s=read(inp/'summary.json');assert s['execution_complete']
    for name,value in p['source_sha256'].items():assert sha(Path(__file__).with_name(name))==value,name
    for name in ['protocol','banks','captures','paired']:assert s[name+'_sha256']==sha(inp/(name+'.json'))
    parent=root/'results/common_pool_credit/development';oldbanks={(r['seed'],r['method']):r for r in read(parent/'banks.json')}
    assert p['parent_banks_sha256']==sha(parent/'banks.json') and p['parent_pools_sha256']==sha(parent/'pools.json')
    grouped={r['seed']:r for r in read(root/'results/common_pool_credit/audit/grouped_sets.json')}
    records={(r['seed'],r['method']):r for r in read(inp/'banks.json')};captures=read(inp/'captures.json')
    checks=dict(tasks=0,trajectory_events=0,shadow_steps=0,banks=0,replayed_regions=0,replayed_arrays=0,native_prior_arrays=0,old_exact=0,new_exact=0,
        convex_exact=0,rounding_exact=0,method_exclusions=0,unique_lp_infeasible=0,expanded=0,expanded_positive=0,expanded_lower_positive=0)
    proofrows=[];expansions=[];duals=[];paired=[];examples=[]
    for cap in captures:
        seed=cap['seed'];assert sha(inp/cap['history_file'])==cap['history_sha256'] and sha(parent/cap['pool_file'])==cap['pool_sha256']
        with np.load(inp/cap['history_file']) as z:saved={key:z[key] for key in z.files}
        with np.load(parent/cap['pool_file']) as z:x=z['x'];v=z['v'];regs=z['regs']
        actual=history.capture(x,v,True);rebuilt,meta_banks=history.build(actual[6])
        for key,value in {**actual[6],**rebuilt}.items():assert value.tobytes()==saved[key].tobytes()
        for key in ['trajectory_sha256','best_sha256','events','shadow_checks']:assert actual[7][key]==cap['captured'][key]
        assert meta_banks=={k:dict(v,steps=tuple(v['steps'])) for k,v in cap['archive'].items()}
        checks['trajectory_events']+=actual[7]['events'];checks['shadow_steps']+=actual[7]['shadow_checks']
        raw=expansion.rational_history(saved);duals.append(dict(seed=seed,**expansion.shadow_error(saved,raw)))
        sets={};lp_cache={}
        for method in p['methods']:
            rec=records[seed,method];assert sha(inp/rec['arrays_file'])==rec['arrays_sha256'] and sha(inp/rec['meta_file'])==rec['meta_sha256']
            with np.load(inp/rec['arrays_file']) as z:arrays={key:z[key] for key in z.files}
            meta=read(inp/rec['meta_file']);bank=saved['native' if method=='alm_native' else method]
            again,am=common.solve(x,v,regs,bank)
            for key,value in arrays.items():assert value.tobytes()==again[key].tobytes();checks['replayed_arrays']+=1
            assert meta['old_proofs']==am['old_proofs'] and meta['proofs']==am['proofs'];checks['replayed_regions']+=len(arrays['indices'])
            assert np.array_equal(arrays['indices'],np.flatnonzero(~arrays['old_positive']))
            if method=='alm_native':
                old=oldbanks[seed,method];assert sha(parent/old['arrays_file'])==old['arrays_sha256']
                with np.load(parent/old['arrays_file']) as z:
                    for key,value in arrays.items():assert value.tobytes()==z[key].tobytes();checks['native_prior_arrays']+=1
                om=read(parent/old['meta_file']);assert meta['old_proofs']==om['old_proofs'] and meta['proofs']==om['proofs']
            lookup={reg.tobytes().hex():i for i,reg in enumerate(regs)};todo=[]
            for pr in meta['old_proofs']:
                index=lookup[pr['pattern']];exact=cert.exact_optimum(x,v,regs[index],bank[pr['direction']]);assert exact['positive'];checks['old_exact']+=1
                if method=='alm_native':
                    weights=np.zeros(len(bank));weights[pr['direction']]=1;todo.append((index,bank[pr['direction']],weights,'individual'))
            rows=[]
            for pr in meta['proofs']:
                q=pr['index'];index=int(arrays['indices'][q]);reg=regs[index];a=arrays['credit'][q];w=arrays['weights'][q]
                exact=cert.exact_optimum(x,v,reg,a);assert exact==pr['exact'] and exact['positive'];checks['new_exact']+=1
                ex=exact_mixture(bank,w);value=rational_optimum(x,v,reg,ex);assert value>0;checks['convex_exact']+=1
                error=sum(abs(ex[j][i]-F(float(a[j,i]))) for j in range(reg.shape[0]) for i in range(reg.shape[1]))
                lower=F(int(exact['numerator']),int(exact['denominator']))-error;assert 0<lower<=value;checks['rounding_exact']+=1
                rows.append(dict(index=index,numerator=str(value.numerator),denominator=str(value.denominator),rounding_l1=float(error),lower=float(lower)))
                if method=='alm_native':todo.append((index,a,w,'joint'))
            for index,a,w,kind in todo:
                result=expansion.expand(x,v,regs[index],a,w,saved,saved,raw)
                expansions.append(dict(seed=seed,index=index,pattern=regs[index].tobytes().hex(),kind=kind,**result));checks['expanded']+=1
                checks['expanded_positive']+=result['positive'];checks['expanded_lower_positive']+=result['lower_positive']
            ids=np.flatnonzero(arrays['accepted']);sets[method]={regs[i].tobytes().hex() for i in ids}
            assert sets[method]=={pr['pattern'] for pr in meta['old_proofs']}|{pr['pattern'] for pr in meta['proofs']}
            for index in ids:
                if int(index) not in lp_cache:
                    _,_,mat,rhs=neighbor.pattern_matrix(x,v,regs[index])
                    lp=linprog(np.zeros(regs.shape[1]),A_ub=mat,b_ub=rhs,bounds=[(-.12,.12)]*regs.shape[1],options={'primal_feasibility_tolerance':1e-9})
                    assert lp.status==2;lp_cache[int(index)]=True;checks['unique_lp_infeasible']+=1
            assert len(ids)==rec['total_positive'];checks['method_exclusions']+=len(ids);checks['banks']+=1
            proofrows.append(dict(seed=seed,method=method,proofs=rows))
        for method in history.VARIANTS:
            a=sets['alm_native'];b=sets[method];paired.append(dict(seed=seed,method=method,common=len(a&b),only_native=len(a-b),only_history=len(b-a),
                only_native_patterns=sorted(a-b),only_history_patterns=sorted(b-a)))
        for pattern in grouped[seed]['alm_only_vs_all_other_patterns']:
            assert pattern in sets['alm_native'];examples.append(dict(seed=seed,pattern=pattern,covered={m:pattern in sets[m] for m in history.VARIANTS}))
        checks['tasks']+=1;print(json.dumps(dict(seed=seed,**checks)),flush=True)
    assert paired==read(inp/'paired.json') and len(examples)==6
    for row in s['summaries']:
        rr=[r for r in records.values() if r['method']==row['method']]
        for key,value in row.items():
            if key=='method':continue
            if key=='prefix_total':assert value=={str(t):sum(r[key][str(t)] for r in rr) for t in [1,4,16,64,128]}
            elif key.startswith('mean_') and key!='mean_evaluations':assert value==float(np.mean([r[key[5:]] for r in rr]))
            else:assert value==sum(r[key] for r in rr)
    assert checks['expanded']==263
    out=base/'audit';out.mkdir(parents=True,exist_ok=True);assert not (out/'summary.json').exists()
    for name,value in [('proofs',proofrows),('expansions',expansions),('dual_rounding',duals),('six_examples',examples)]:
        (out/(name+'.json')).write_text(json.dumps(value,indent=2),encoding='utf-8')
    result=dict(passed=True,checks=checks,comparisons=s['comparisons'],six_example_coverage={m:sum(r['covered'][m] for r in examples) for m in history.VARIANTS},
        max_ideal_dual_l1=max(r['max_ideal_dual_l1'] for r in duals),max_expansion_bank_l1=max(r['bank_l1_error'] for r in expansions),
        min_expansion_normalized_lower=min(r['normalized_lower'] for r in expansions),expanded_above_delta=sum(r['full_history_value']>.001001 for r in expansions),
        source_sha256=sha(Path(__file__)),input_sha256={name:sha(inp/name) for name in ['protocol.json','summary.json','banks.json','captures.json','paired.json']},
        output_sha256={name:sha(out/name) for name in ['proofs.json','expansions.json','dual_rounding.json','six_examples.json']},
        scope='All original ALM certificate expansions checked offline; none supplied to history solvers')
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
