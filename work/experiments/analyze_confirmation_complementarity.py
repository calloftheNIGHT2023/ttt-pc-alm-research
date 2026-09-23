"""269: fixed-pool union accounting, not a new online method or free ensemble."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from run_multiplier_fixed_point_screen import sha,dump

FIELDS=['primary_excess','control_excess','union_excess','addition','discarding','difference']
DIFFERENCES=['addition','discarding','difference']
GROUPS=['primary_empty_control_nonempty','primary_nonempty_control_empty','both_empty','both_nonempty']


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True)
    root=ap.parse_args().project.resolve();src=Path(__file__).parent;base=root/'results/confirmation_conditional_risk'
    inp=base/'decomposition';out=base/'complementarity';out.mkdir(parents=True,exist_ok=True)
    assert not (out/'protocol.json').exists()
    parent=root/'results/round_268_audit.json';aa=json.loads(parent.read_text());assert aa['passed']
    hashes=dict(aa['source_sha256']);hashes[Path(__file__).name]=sha(Path(__file__))
    for n,h in hashes.items():assert sha(src/n)==h,n
    design='269_complementary_mode_accounting_protocol.md';assert sha(root/'outputs/ttt-pc-alm-research'/design)==aa['report_sha256'][design]
    ss=json.loads((inp/'summary.json').read_text())
    for n,h in ss['outputs_sha256'].items():assert sha(inp/n)==h
    p0=json.loads((inp/'protocol.json').read_text());names=p0['methods'];primary=p0['primary'];pj=names.index(primary)
    risks={(r['seed'],r['method'],r['grid']):r for r in json.loads((inp/'rows.json').read_text())}
    mode_names=[n for n in names if risks[p0['seeds'][0],n,257]['kind']!='fixed_prediction_head'];assert len(mode_names)==35
    files0=json.loads((inp/'files.json').read_text());tasks=json.loads((base/'moments/tasks.json').read_text())
    p=dict(source_sha256=hashes,parent_audit_sha256=sha(parent),design_sha256=aa['report_sha256'][design],
        seeds=p0['seeds'],methods=mode_names,primary=primary,grids=[257,129],batch_pairs=[[0,1],[2,3]],
        fields=FIELDS,differences=DIFFERENCES,groups=GROUPS,bootstrap_seed=262193,bootstrap_replicates=20000,
        phase_accesses_query_targets=False,scope='Fixed-pool post-result capacity accounting only; neither online selector nor matched-cost union algorithm')
    dump(out/'protocol.json',p);begin=time.perf_counter();rows=[];files={};maxidentity=0.;maxold=0.
    for task in tasks:
        seed=task['seed'];fn=f'{seed}_curves.npz';assert sha(inp/fn)==files0[fn]
        assert sha(base/'moments'/task['file'])==task['sha256']
        with np.load(base/'moments'/task['file']) as z:regional=z['means'].copy()
        with np.load(inp/fn) as z:
            w=z['weights'];m=z['full_means'];masks=z['masks'];strategy=z['strategy_means'];s=strategy[pj];smask=masks[pj]
            unions=np.empty((35,4,257));umasks=np.zeros((35,len(w)),dtype=bool)
            for j,name in enumerate(mode_names):
                cj=names.index(name);c=strategy[cj];cmask=masks[cj];mask=smask|cmask;umasks[j]=mask
                if mask.any():
                    ww=w*mask;ww/=ww.sum();u=np.einsum('k,kbq->bq',ww,regional)
                else:u=c.copy()
                unions[j]=u
                group=('both_nonempty' if cmask.any() else 'primary_nonempty_control_empty') if smask.any() else ('primary_empty_control_nonempty' if cmask.any() else 'both_empty')
                for grid in p['grids']:
                    ii=np.arange(257) if grid==257 else np.arange(0,257,2);pairs=[]
                    for a,b in p['batch_pairs']:
                        es=float(np.mean((s[a,ii]-m[a,ii])*(s[b,ii]-m[b,ii])))
                        ec=float(np.mean((c[a,ii]-m[a,ii])*(c[b,ii]-m[b,ii])))
                        eu=float(np.mean((u[a,ii]-m[a,ii])*(u[b,ii]-m[b,ii])))
                        addition,discarding,difference=eu-ec,es-eu,es-ec
                        residual=abs(difference-addition-discarding);assert residual<1e-12;maxidentity=max(maxidentity,residual)
                        if name==primary:assert max(abs(addition),abs(discarding),abs(difference))<1e-12
                        pairs.append(dict(pair=[a,b],primary_excess=es,control_excess=ec,union_excess=eu,addition=addition,discarding=discarding,difference=difference))
                    mean={f:float(np.mean([r[f] for r in pairs])) for f in FIELDS}
                    oldgap=max(abs(mean['primary_excess']-risks[seed,primary,grid]['policy_excess']),abs(mean['control_excess']-risks[seed,name,grid]['policy_excess']))
                    assert oldgap<1e-12;maxold=max(maxold,oldgap)
                    rows.append(dict(seed=seed,control=name,grid=grid,group=group,pairs=pairs,**mean,
                        primary_modes=int(smask.sum()),control_modes=int(cmask.sum()),union_modes=int(mask.sum()),
                        primary_unique_modes=int((smask&~cmask).sum()),control_unique_modes=int((cmask&~smask).sum()),
                        primary_mass=float(w[smask].sum()),control_mass=float(w[cmask].sum()),union_mass=float(w[mask].sum()),
                        primary_unique_mass=float(w[smask&~cmask].sum()),control_unique_mass=float(w[cmask&~smask].sum())))
        filename=f'{seed}_union_means.npz';np.savez_compressed(out/filename,methods=np.array(mode_names),masks=umasks,means=unions)
        files[filename]=sha(out/filename)
    lookup={(r['seed'],r['control'],r['grid']):r for r in rows};ids=np.random.default_rng(262193).integers(0,64,(20000,64));methods=[];paired=[]
    for grid in p['grids']:
        for name in mode_names:
            rr=[lookup[s,name,grid] for s in p['seeds']]
            means={f:float(np.mean([r[f] for r in rr])) for f in FIELDS}
            groups=[]
            for group in GROUPS:
                members=[r for r in rr if r['group']==group]
                groups.append(dict(group=group,tasks=len(members),contribution_to_all64_mean={f:float(sum(r[f] for r in members)/64) for f in FIELDS}))
            for f in FIELDS:assert abs(sum(g['contribution_to_all64_mean'][f] for g in groups)-means[f])<1e-12
            methods.append(dict(control=name,grid=grid,**means,groups=groups,
                mean_primary_unique_modes=float(np.mean([r['primary_unique_modes'] for r in rr])),
                mean_control_unique_modes=float(np.mean([r['control_unique_modes'] for r in rr])),
                mean_union_mass=float(np.mean([r['union_mass'] for r in rr]))))
            for f in DIFFERENCES:
                delta=np.array([r[f] for r in rr]);paired.append(dict(control=name,grid=grid,metric=f,mean_difference=float(delta.mean()),
                    descriptive_ci95=np.quantile(delta[ids].mean(1),[.025,.975]).tolist(),lower_tasks=int((delta<-1e-12).sum()),equal_tasks=int((abs(delta)<=1e-12).sum()),higher_tasks=int((delta>1e-12).sum())))
    for filename,data in [('rows.json',rows),('methods.json',methods),('paired.json',paired),('files.json',files)]:dump(out/filename,data)
    for n,h in hashes.items():assert sha(src/n)==h,n
    ans=dict(passed=True,tasks=64,controls=35,rows=len(rows),paired_intervals=len(paired),max_identity_residual=maxidentity,
        max_original_policy_gap=maxold,seconds=time.perf_counter()-begin,phase_accesses_query_targets=False,
        outputs_sha256={n:sha(out/n) for n in ['protocol.json','rows.json','methods.json','paired.json','files.json']},
        next='Independent union and contribution audit; do not call union a matched-budget method')
    dump(out/'summary.json',ans);print(json.dumps(ans),flush=True)


if __name__=='__main__':main()
