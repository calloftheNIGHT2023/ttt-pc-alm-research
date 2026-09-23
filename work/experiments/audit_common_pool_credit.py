"""Rebuild common pools, replay paths, verify exact mixtures and all exclusions."""
import argparse
import hashlib
import json
from fractions import Fraction as F
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
import common_pool_credit as model
import optimized_branch_dual as cert
import neighbor_mode_memory as neighbor
from audit_joint_credit_minimax import exact_mixture,rational_optimum


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();base=root/'results/common_pool_credit';inp=base/'development'
    p=read(inp/'protocol.json');s=read(inp/'summary.json');assert s['execution_complete']
    for name,value in p['source_sha256'].items():assert sha(Path(__file__).with_name(name))==value,name
    assert s['protocol_sha256']==sha(inp/'protocol.json') and s['banks_sha256']==sha(inp/'banks.json')
    assert s['pools_sha256']==sha(inp/'pools.json') and s['paired_sha256']==sha(inp/'paired.json')
    assert p['design_sha256']==sha(root/'outputs/ttt-pc-alm-research/229_common_pool_credit_protocol.md') and p['primitive_sha256']==sha(base/'primitive/summary.json')
    source=root/'results/light_h2_credit/full_bank_ceiling';assert p['records_sha256']==sha(source/'records.json')
    original={(r['seed'],r['method']):r for r in read(source/'records.json')};banks={(r['seed'],r['method']):r for r in read(inp/'banks.json')}
    pools=read(inp/'pools.json');checks=dict(tasks=0,banks=0,common_regions=0,replayed_regions=0,replayed_arrays=0,old_exact=0,new_exact=0,
        convex_exact=0,rounding_exact=0,method_exclusions=0,unique_parameter_lp_infeasible=0,paired_sets=0)
    paired=[];proofrows=[];grouped=[];tasks=[]
    for pool in pools:
        seed=pool['seed'];items={}
        for method in model.METHODS:
            r=original[seed,method];assert sha(source/r['data_file'])==r['data_sha256']
            with np.load(source/r['data_file']) as z:items[method]={key:z[key] for key in ['x','v','regs','bank']}
        x,v,regs,membership=model.assemble(items);assert sha(inp/pool['arrays_file'])==pool['arrays_sha256']
        with np.load(inp/pool['arrays_file']) as z:
            for name,value in [('x',x),('v',v),('regs',regs),('membership',membership)]:assert z[name].tobytes()==value.tobytes()
        assert pool['pattern_sha256']==hashlib.sha256(regs.tobytes()).hexdigest();sets={};rowsets={};lp_cache={}
        for method in model.METHODS:
            r=banks[seed,method];assert r['pool_regions']==len(regs) and r['pool_sha256']==pool['arrays_sha256']
            assert sha(inp/r['arrays_file'])==r['arrays_sha256'] and sha(inp/r['meta_file'])==r['meta_sha256']
            with np.load(inp/r['arrays_file']) as z:arrays={key:z[key] for key in z.files}
            meta=read(inp/r['meta_file']);bank=items[method]['bank'];again,am=model.solve(x,v,regs,bank)
            for key,value in arrays.items():assert value.tobytes()==again[key].tobytes();checks['replayed_arrays']+=1
            assert meta['old_proofs']==am['old_proofs'] and meta['proofs']==am['proofs'];checks['replayed_regions']+=len(arrays['indices'])
            assert np.array_equal(arrays['indices'],np.flatnonzero(~arrays['old_positive']))
            assert int(arrays['accepted'].sum())==r['total_positive']==r['old_count']+r['positive']
            lookup={reg.tobytes().hex():i for i,reg in enumerate(regs)}
            assert {pr['pattern'] for pr in meta['old_proofs']}=={regs[i].tobytes().hex() for i in np.flatnonzero(arrays['old_positive'])}
            for pr in meta['old_proofs']:
                index=lookup[pr['pattern']];exact=cert.exact_optimum(x,v,regs[index],bank[pr['direction']]);assert exact['positive'];checks['old_exact']+=1
            rows=[]
            for pr in meta['proofs']:
                q=pr['index'];index=int(arrays['indices'][q]);reg=regs[index];a=arrays['credit'][q];w=arrays['weights'][q]
                assert pr['pattern']==reg.tobytes().hex();exact=cert.exact_optimum(x,v,reg,a);assert exact==pr['exact'] and exact['positive'];checks['new_exact']+=1
                exacta=exact_mixture(bank,w);value=rational_optimum(x,v,reg,exacta);assert value>0;checks['convex_exact']+=1
                error=sum(abs(exacta[j][i]-F(float(a[j,i]))) for j in range(reg.shape[0]) for i in range(reg.shape[1]))
                lower=F(int(exact['numerator']),int(exact['denominator']))-error;assert 0<lower<=value;checks['rounding_exact']+=1
                rows.append(dict(index=index,step=pr['step'],numerator=str(value.numerator),denominator=str(value.denominator),rounding_l1=float(error),lower=float(lower)))
            selected=set(np.flatnonzero(arrays['accepted']).tolist());rowsets[method]=selected;sets[method]={regs[i].tobytes().hex() for i in selected}
            assert sets[method]=={pr['pattern'] for pr in meta['old_proofs']}|{pr['pattern'] for pr in meta['proofs']}
            for index in sorted(selected):
                if index not in lp_cache:
                    _,_,mat,rhs=neighbor.pattern_matrix(x,v,regs[index])
                    lp=linprog(np.zeros(regs.shape[1]),A_ub=mat,b_ub=rhs,bounds=[(-.12,.12)]*regs.shape[1],options={'primal_feasibility_tolerance':1e-9})
                    assert lp.status==2;lp_cache[index]=True;checks['unique_parameter_lp_infeasible']+=1
            checks['method_exclusions']+=len(selected);checks['banks']+=1;proofrows.append(dict(seed=seed,method=method,new_proofs=rows))
        for i,m1 in enumerate(model.METHODS):
            for m2 in model.METHODS[i+1:]:
                a=sets[m1];b=sets[m2];paired.append(dict(seed=seed,method_a=m1,method_b=m2,common=len(a&b),only_a=len(a-b),only_b=len(b-a),
                    only_a_patterns=sorted(a-b),only_b_patterns=sorted(b-a)));checks['paired_sets']+=1
        nonalm=set().union(*(sets[m] for m in ['adam60_native','adam60_residual','pc_native','nodual_native']))
        allother=nonalm|sets['alm_residual'];alm=sets['alm_native']
        grouped.append(dict(seed=seed,alm=len(alm),nonalm_union=len(nonalm),alm_only_vs_nonalm=len(alm-nonalm),nonalm_only=len(nonalm-alm),
            alm_only_vs_all_other=len(alm-allother),alm_only_vs_nonalm_patterns=sorted(alm-nonalm),alm_only_vs_all_other_patterns=sorted(alm-allother)))
        tasks.append(dict(seed=seed,common_regions=len(regs),accepted_by_at_least_one=len(lp_cache),totals={k:len(v) for k,v in sets.items()}))
        checks['tasks']+=1;checks['common_regions']+=len(regs);print(json.dumps(dict(seed=seed,**checks)),flush=True)
    assert paired==read(inp/'paired.json') and checks['common_regions']==s['unique_task_regions'] and checks['banks']==96
    assert checks['old_exact']==sum(r['old_count'] for r in s['summaries']) and checks['new_exact']==sum(r['positive'] for r in s['summaries'])
    comparisons=[]
    for c in s['comparisons']:
        rr=[r for r in paired if r['method_a']==c['method_a'] and r['method_b']==c['method_b']]
        actual=dict(method_a=c['method_a'],method_b=c['method_b'],**{k:sum(r[k] for r in rr) for k in ['common','only_a','only_b']});assert actual==c;comparisons.append(actual)
    for row in s['summaries']:
        rr=[r for r in banks.values() if r['method']==row['method']]
        for key,value in row.items():
            if key=='method':continue
            if key=='prefix_total':assert value=={str(t):sum(r[key][str(t)] for r in rr) for t in [1,4,16,64,128]}
            elif key.startswith('mean_') and key not in ['mean_evaluations']:assert value==float(np.mean([r[key[5:]] for r in rr]))
            else:assert value==sum(r[key] for r in rr)
    out=base/'audit';out.mkdir(parents=True,exist_ok=True);assert not (out/'summary.json').exists()
    (out/'proofs.json').write_text(json.dumps(proofrows,indent=2),encoding='utf-8');(out/'tasks.json').write_text(json.dumps(tasks,indent=2),encoding='utf-8')
    (out/'grouped_sets.json').write_text(json.dumps(grouped,indent=2),encoding='utf-8')
    result=dict(passed=True,checks=checks,comparisons=comparisons,
        grouped={k:sum(r[k] for r in grouped) for k in ['alm','nonalm_union','alm_only_vs_nonalm','nonalm_only','alm_only_vs_all_other']},
        source_sha256=sha(Path(__file__)),input_sha256={name:sha(inp/name) for name in ['protocol.json','summary.json','banks.json','pools.json','paired.json']},
        proofs_sha256=sha(out/'proofs.json'),tasks_sha256=sha(out/'tasks.json'),grouped_sets_sha256=sha(out/'grouped_sets.json'),
        scope='Same-region finite-path complementarity, not mathematical inability of any unproved credit hull or online task superiority')
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
