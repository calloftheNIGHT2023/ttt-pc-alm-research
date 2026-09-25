"""362 independent scheduling/resource/Fraction audit; post-seal geometry only."""
from collections import Counter
from fractions import Fraction
from pathlib import Path
import time
import numpy as np
import budget_reinvestment_suite_v1 as io
import branch_image_chain_v1 as exact


def audit_one(x,v,regs,a,m):
    counts=Counter();cut=a['base_first_step'].copy();disabled=a['base_disabled'].copy();visits=a['base_response_counts'].copy()
    assert int(visits.sum())==m['base_response_pairs']<=m['budget_limit']==len(regs)*m['steps']
    by={p['index']:p for p in m['proofs']};assert len(by)==len(m['proofs'])==int((a['first_step']>0).sum())
    base_indices=set()
    for call in m['base_calls']:
        for p in call['metadata']['proofs']:
            i=call['begin']+p['index'];base_indices.add(i)
            assert by[i]['stage']=='base' and by[i]['step']==p['step']==int(cut[i])
    assert base_indices==set(np.flatnonzero(cut>0))
    for round_number,item in enumerate(m['trace'],1):
        candidates=[i for i,s in enumerate(cut) if s==0]
        if m['strategy']=='frontier':candidates=candidates[:m['geometry_budget']]
        candidates=[i for i in candidates if not disabled[i]][:m['budget_limit']-int(visits.sum())]
        assert item['round']==round_number and item['indices']==candidates and candidates
        for i in candidates:visits[i]+=1
        assert item['response_pairs']==int(visits.sum())<=m['budget_limit']
        assert not set(item['certified'])&set(item['disabled'])
        for i in item['certified']:
            assert i in candidates and cut[i]==0 and by[i]['stage']=='continued' and by[i]['round']==round_number
            assert by[i]['step']==int(visits[i]);cut[i]=int(visits[i])
        for i in item['disabled']:
            assert i in candidates and cut[i]==0;disabled[i]=True
        counts['rounds']+=1
    assert cut.tobytes()==a['first_step'].tobytes() and disabled.tobytes()==a['disabled'].tobytes()
    assert visits.tobytes()==a['response_counts'].tobytes() and int(visits.sum())==m['total_response_pairs']
    assert m['continuation_rounds']==len(m['trace'])
    selected=[i for i,s in enumerate(cut) if s==0][:m['geometry_budget']]
    assert selected==a['selected_indices'].tolist()
    if int(visits.sum())<m['budget_limit']:
        active=[i for i,s in enumerate(cut) if s==0]
        if m['strategy']=='frontier':active=active[:m['geometry_budget']]
        assert not [i for i in active if not disabled[i]]
    for i,p in by.items():
        value=exact.fixed_value(x,v,a['proof_credit'][i],regs[i]);assert (p['structural_empty'] and value is None) or value==Fraction(p['lower'])>0
        assert p['step']==int(cut[i]) and p['mode']==regs[i].tobytes().hex();counts['fraction_certificates']+=1
    counts['budget_and_selection_checks']+=1
    return counts


def run(root,out):
    begin=time.perf_counter();source=root/'results/frontier_reallocation/development_v1';io.complete(source)
    parent=root/'results/cross_region_credit/development_v1';geometry=root/'results/cross_region_credit/audit_v1';io.complete(geometry)
    p=io.read(source/'protocol.json');counts=Counter();totals={c['name']:Counter() for c in p['configs']};rows=[]
    for name,digest in p['source_sha256'].items():assert io.sha(root/name)==digest
    assert io.sha(parent/'tasks.json')==p['parent_tasks_sha256']
    config={c['name']:c for c in p['configs']}
    for seal in io.read(source/'input_seals.json'):
        for name,digest in seal['source_files'].items():assert io.sha(parent/str(seal['seed'])/name)==digest
        assert io.sha(source/str(seal['seed'])/'commit.json')==seal['commit_sha256']
    for row in io.read(source/'rows.json'):
        seed,name=row['seed'],row['method'];directory=source/str(seed);src=parent/str(seed)
        for ext,digest in row['files'].items():assert io.sha(directory/(name+ext))==digest
        inp=io.read(src/'input.json');x,v=np.array(inp['x_observed']),np.array(inp['v_observed'])
        with np.load(src/'pool.npz',allow_pickle=False) as z:regs=z['regions']
        with np.load(directory/(name+'.npz'),allow_pickle=False) as z:a={k:z[k] for k in z.files}
        m=io.read(directory/(name+'.json'));counts.update(audit_one(x,v,regs,a,m))
        cfg=config[name];assert m['strategy']==cfg['strategy']
        with np.load(src/(cfg['channel']+'_independent.npz'),allow_pickle=False) as z:
            for key,old in [('base_first_step','first_step'),('base_disabled','disabled'),('base_final_credit','final_credit')]:
                assert a[key][:m['base_processed']].tobytes()==z[old][:m['base_processed']].tobytes();counts['base_bitwise_arrays']+=1
            before=np.flatnonzero(z['first_step']==0)[:8];assert before.tolist()==a['base_selected'].tolist()
        notes=io.read(geometry/(str(seed)+'_geometry.json'));labels=[notes[r.tobytes().hex()]['classification'] for r in regs]
        assert all(labels[i]=='infeasible' for i in np.flatnonzero(a['first_step']>0))
        oldpositive={int(i) for i in before if labels[i]=='positive_volume'}
        newpositive={int(i) for i in a['selected_indices'] if labels[i]=='positive_volume'}
        assert oldpositive<=newpositive;counts['positive_pool_inclusions']+=1
        t=totals[name]
        for key in ['total_response_pairs','base_response_pairs','continuation_rounds','exact_calls','total_seconds','budget_limit']:
            t[key]+=m[key]
        t['rejected']+=int((a['first_step']>0).sum());t['selected_positive']+=len(newpositive);t['new_positive']+=len(newpositive-oldpositive)
        rows.append(dict(seed=seed,method=name,base_positive=sorted(oldpositive),selected_positive=sorted(newpositive),
                         selected_indices=a['selected_indices'].tolist(),budget_limit=m['budget_limit'],response_pairs=m['total_response_pairs']))
    io.save(out/'rows.json',rows)
    result=dict(passed=True,counts=dict(counts),totals={k:dict(v) for k,v in totals.items()},seconds=time.perf_counter()-begin,
        source_summary_sha256=io.sha(source/'summary.json'),geometry_summary_sha256=io.sha(geometry/'summary.json'),
        query_targets_accessed=False,query_risk_evaluated=False,core_research_goal_complete=False,
        outputs_sha256={'rows.json':io.sha(out/'rows.json')})
    io.save(out/'summary.json',result);print(result,flush=True)


if __name__=='__main__':
    root=Path(__file__).resolve().parents[2];out=root/'results/frontier_reallocation/audit_v1';out.mkdir(parents=True,exist_ok=False);run(root,out)
