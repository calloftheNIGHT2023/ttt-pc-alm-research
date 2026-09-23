"""Independently rebuild every diagnostic union and all grouped contributions."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time
import numpy as np
from run_multiplier_fixed_point_screen import sha,dump


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True)
    root=ap.parse_args().project.resolve();src=Path(__file__).parent;base=root/'results/confirmation_conditional_risk'
    inp=base/'complementarity';out=base/'complementarity_audit';out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    summary=json.loads((inp/'summary.json').read_text());assert summary['passed']
    for n,h in summary['outputs_sha256'].items():assert sha(inp/n)==h
    p=json.loads((inp/'protocol.json').read_text());hashes=dict(p['source_sha256']);hashes[Path(__file__).name]=sha(Path(__file__))
    for n,h in hashes.items():assert sha(src/n)==h,n
    assert sha(root/'outputs/ttt-pc-alm-research/269_complementary_mode_accounting_protocol.md')==p['design_sha256']
    assert sha(root/'results/round_268_audit.json')==p['parent_audit_sha256']
    dump(out/'protocol.json',dict(source_sha256=hashes,input_summary_sha256=sha(inp/'summary.json'),phase_accesses_query_targets=False,
        scope='Independent mixture construction and expanded risk polynomial; all groups and210 task-bootstrap intervals'))
    files=json.loads((inp/'files.json').read_text());tasks=json.loads((base/'moments/tasks.json').read_text())
    saved={(r['seed'],r['control'],r['grid']):r for r in json.loads((inp/'rows.json').read_text())}
    oldp=json.loads((base/'decomposition/protocol.json').read_text());names=oldp['methods'];pj=names.index(p['primary'])
    oldfiles=json.loads((base/'decomposition/files.json').read_text());replay={};counts=Counter();maximum=0.;begin=time.perf_counter()

    def check(a,b):
        nonlocal maximum
        gap=float(np.max(np.abs(np.asarray(a)-np.asarray(b))));maximum=max(maximum,gap);assert gap<1e-12,gap
        counts['numeric_checks']+=1

    for task in tasks:
        seed=task['seed'];path=base/'moments'/task['file'];assert sha(path)==task['sha256']
        fn=f'{seed}_curves.npz';assert sha(base/'decomposition'/fn)==oldfiles[fn]
        ufn=f'{seed}_union_means.npz';assert sha(inp/ufn)==files[ufn]
        with np.load(path) as z,np.load(base/'decomposition'/fn) as old,np.load(inp/ufn) as unions:
            w=z['volumes']/np.sum(z['volumes']);mu=z['means'];full=np.sum(w[:,None,None]*mu,axis=0)
            assert unions['methods'].tolist()==p['methods']
            smask=old['masks'][pj]
            primary=np.sum((w[smask]/w[smask].sum())[:,None,None]*mu[smask],axis=0) if smask.any() else np.tile(old['predictions'][pj],(4,1))
            for j,name in enumerate(p['methods']):
                cj=names.index(name);cmask=old['masks'][cj];mask=np.logical_or(smask,cmask)
                c=np.sum((w[cmask]/w[cmask].sum())[:,None,None]*mu[cmask],axis=0) if cmask.any() else np.tile(old['predictions'][cj],(4,1))
                u=np.sum((w[mask]/w[mask].sum())[:,None,None]*mu[mask],axis=0) if mask.any() else c.copy()
                check(unions['means'][j],u);assert np.array_equal(unions['masks'][j],mask)
                group=('both_nonempty' if cmask.any() else 'primary_nonempty_control_empty') if smask.any() else ('primary_empty_control_nonempty' if cmask.any() else 'both_empty')
                for grid in p['grids']:
                    sl=slice(None) if grid==257 else slice(None,None,2);row=saved[seed,name,grid];assert row['group']==group
                    for prefix,mm in [('primary',smask),('control',cmask),('union',mask),('primary_unique',smask&~cmask),('control_unique',cmask&~smask)]:
                        assert row[prefix+'_modes']==int(mm.sum());check(row[prefix+'_mass'],w[mm].sum())
                    pair=[]
                    for i,(a,b) in enumerate(p['batch_pairs']):
                        ma,mb=full[a,sl],full[b,sl]
                        values=[]
                        for strategy in [primary,c,u]:
                            sa,sb=strategy[a,sl],strategy[b,sl]
                            values.append(float(np.average(sa*sb-sa*mb-sb*ma+ma*mb)))
                        es,ec,eu=values
                        rr=dict(primary_excess=es,control_excess=ec,union_excess=eu,addition=eu-ec,discarding=es-eu,difference=es-ec)
                        for f in p['fields']:check(rr[f],row['pairs'][i][f]);counts['pair_metric_replays']+=1
                        check(rr['difference'],rr['addition']+rr['discarding']);pair.append(rr)
                    mean={f:float(np.average([rr[f] for rr in pair])) for f in p['fields']}
                    for f in p['fields']:check(mean[f],row[f])
                    replay[seed,name,grid]=dict(group=group,**mean)
                counts['union_curves']+=1
    for r in json.loads((inp/'methods.json').read_text()):
        rr=[replay[s,r['control'],r['grid']] for s in p['seeds']]
        for f in p['fields']:check(r[f],np.mean([v[f] for v in rr]))
        for g in r['groups']:
            group=[v for v in rr if v['group']==g['group']];assert len(group)==g['tasks']
            for f in p['fields']:check(g['contribution_to_all64_mean'][f],sum(v[f] for v in group)/64)
            counts['task_group_summaries']+=1
        originals=[saved[s,r['control'],r['grid']] for s in p['seeds']]
        for key in ['primary_unique_modes','control_unique_modes','union_mass']:check(r['mean_'+key],np.mean([v[key] for v in originals]))
        counts['method_grid_summaries']+=1
    ids=np.random.default_rng(262193).integers(0,64,(20000,64))
    for r in json.loads((inp/'paired.json').read_text()):
        delta=np.array([replay[s,r['control'],r['grid']][r['metric']] for s in p['seeds']])
        check(r['mean_difference'],delta.mean());check(r['descriptive_ci95'],np.quantile(delta[ids].mean(1),[.025,.975]))
        assert r['lower_tasks']==int((delta<-1e-12).sum()) and r['equal_tasks']==int((abs(delta)<=1e-12).sum()) and r['higher_tasks']==int((delta>1e-12).sum())
        counts['paired_intervals']+=1
    assert counts['paired_intervals']==210 and len(replay)==4480
    ans=dict(passed=True,counts=counts,maximum_numeric_replay_gap=maximum,seconds=time.perf_counter()-begin,
        protocol_sha256=sha(out/'protocol.json'),phase_accesses_query_targets=False)
    dump(out/'summary.json',ans);print(json.dumps(ans),flush=True)


if __name__=='__main__':main()
