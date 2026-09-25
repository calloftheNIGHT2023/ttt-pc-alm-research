"""Independent proof, geometry, proposal-volume and sample-count audit."""
import argparse,hashlib,json
from fractions import Fraction as F
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import rejection_credit_capture as capture
import parallelotope_sampler as sampler
import neighbor_mode_memory as geometry


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def ci(values):
    a=np.array(values);rng=np.random.default_rng(481751)
    return np.quantile(a[rng.integers(len(a),size=(20000,len(a)))].mean(1),[.025,.975]).tolist()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);args=ap.parse_args();root=args.project.resolve()
    inp=root/'results/rejection_credit/diagnostic';out=root/'results/rejection_credit/analysis'
    protocol=json.loads((inp/'protocol.json').read_text());audits=json.loads((inp/'audits.json').read_text());rows=json.loads((inp/'rows.json').read_text())
    for name,value in protocol['source_sha256'].items():assert sha(Path(__file__).with_name(name))==value,name
    assert len(rows)==320 and len(audits)==16
    checks=dict(boxes=0,exact_proofs=0,independent_empty_lps=0,sampler_replays=0,source_hashes=len(protocol['source_sha256']))
    task_rows=[];worst_count_z=0.;all_accepted=0;source_bytes=[]
    for audit in audits:
        seed=audit['seed'];prop=inp/f'proposal_{seed}.npz';proofpath=inp/f'proofs_{seed}.json'
        assert sha(prop)==audit['proposal_sha256'] and sha(proofpath)==audit['proofs_sha256']
        proofs=json.loads(proofpath.read_text())
        with np.load(prop) as z:
            x=z['x'];v=z['v'];keys=z['patterns'].tolist();volumes=z['volumes'];masks=dict(zip(z['methods'].tolist(),z['rejected']))
            transforms=z['transforms'];centers=z['centers'];radii=z['radii']
        regs=np.array([np.frombuffer(bytes.fromhex(k),np.uint8).reshape(4,len(x)) for k in keys]);boxes=[];matrices=[];rhs=[]
        for i,reg in enumerate(regs):
            p,c,a,r=geometry.pattern_matrix(x,v,reg);box=sampler.make_box(p,c,v)
            assert np.array_equal(box['transform'],transforms[i]) and np.array_equal(box['center'],centers[i]) and np.array_equal(box['radius'],radii[i])
            exact=F(1)
            for radius in radii[i]:exact*=2*F(float(radius))
            exact/=abs(sampler.determinant_exact(transforms[i]));assert abs(float(exact)/volumes[i]-1)<1e-14
            boxes.append(box);matrices.append(a);rhs.append(r);checks['boxes']+=1
        for kind in ['residual','bp','local']:
            certified=np.zeros(len(keys),bool)
            for proof in proofs[kind]['matching_proofs']+proofs[kind]['cross_proofs']:
                i=proof['index'];assert keys[i]==proof['pattern']
                exact=capture.retained.normal.exact_optimum(x,v,regs[i],np.array(proof['a']))
                assert exact==proof['exact'] and exact['positive'];certified[i]=True;checks['exact_proofs']+=1
            assert np.array_equal(certified,masks[kind])
        assert np.array_equal(masks['bp_residual'],masks['bp']|masks['residual'])
        union=masks['local']|masks['bp_residual']
        for i in np.flatnonzero(union):
            lp=linprog(np.zeros(4),A_ub=matrices[i],b_ub=rhs[i],bounds=[(-.12,.12)]*4,method='highs')
            assert lp.status==2,(seed,keys[i],lp.status);checks['independent_empty_lps']+=1
        for method in protocol['methods']:
            indices=np.flatnonzero(~masks[method]);q=float(volumes[indices].sum());assert q==audit['methods'][method]['remaining_proposal_volume']
            rr=[r for r in rows if r['seed']==seed and r['method']==method];n=sum(r['proposals'] for r in rr);count=sum(r['accepted'] for r in rr)
            prob=audit['target_volume']/q;zscore=(count-n*prob)/np.sqrt(n*prob*(1-prob))
            worst_count_z=max(worst_count_z,abs(float(zscore)));all_accepted+=count
            # Replay one complete repetition, including exact region counts.
            rng=np.random.default_rng(np.random.SeedSequence([protocol['rng_seed'],seed,0]))
            bank,owners,_=sampler.attempts([boxes[i] for i in indices],[matrices[i] for i in indices],[rhs[i] for i in indices],protocol['proposals'],rng)
            owners=indices[owners];counts=np.bincount(owners,minlength=len(regs));old=next(r for r in rr if r['repetition']==0)
            assert len(bank)==old['accepted'] and {keys[i]:int(c) for i,c in enumerate(counts) if c}==old['accepted_region_counts']
            checks['sampler_replays']+=1
        qbp=audit['methods']['bp_residual']['remaining_proposal_volume'];qlocal=audit['methods']['local']['remaining_proposal_volume']
        qunion=float(volumes[~union].sum());exclusive=masks['local']&~masks['bp_residual']
        # Exact identity for the recorded binary64 volumes, independent of
        # floating summation order and of their task-dependent physical scale.
        qexact=lambda mask:sum((F(float(t)) for t in volumes[mask]),F(0))
        assert qexact(~union)+qexact(exclusive)==qexact(~masks['bp_residual'])
        task_rows.append(dict(seed=seed,local_exclusive_modes=int(exclusive.sum()),local_exclusive_fraction_of_bp_proposals=float(volumes[exclusive].sum()/qbp),
            local_vs_bp_expected_proposal_ratio=qlocal/qbp,union_vs_bp_expected_proposal_ratio=qunion/qbp,
            local_acceptance=audit['target_volume']/qlocal,bp_acceptance=audit['target_volume']/qbp,
            proposed_union_acceptance=audit['target_volume']/qunion,
            local_screen_seconds=proofs['local']['seconds'],bp_screen_seconds=proofs['bp']['seconds'],residual_screen_seconds=proofs['residual']['seconds']))
        source_bytes.append(dict(seed=seed,proposal_arrays=sum(a.nbytes for a in [regs,volumes,transforms,centers,radii]),
                                 credit_numeric_subtotals={k:proofs[k]['retained_numeric_bytes'] for k in proofs}))
        print(json.dumps(dict(seed=seed,**checks)),flush=True)
    summaries=[]
    for method in protocol['methods']:
        rr=[r for r in rows if r['method']==method]
        summaries.append(dict(method=method,mean_acceptance=float(np.mean([r['empirical_acceptance'] for r in rr])),
            mean_reference_acceptance=float(np.mean([r['reference_expected_acceptance'] for r in rr])),
            mean_accepted_per_16384=float(np.mean([r['accepted'] for r in rr])),zero_sample_runs=sum(r['zero_samples'] for r in rr),
            mean_sampling_seconds=float(np.mean([r['sampling_seconds'] for r in rr])),
            mean_remaining_proposal_ratio=float(np.mean([a['methods'][method]['expected_proposals_relative_to_c20'] for a in audits]))))
    ratios=np.array([r['local_vs_bp_expected_proposal_ratio'] for r in task_rows]);gain=[r['local_acceptance']-r['bp_acceptance'] for r in task_rows]
    result=dict(complete=True,analysis_source_sha256=sha(Path(__file__)),input_sha256={n:sha(inp/n) for n in ['protocol.json','rows.json','audits.json','run_audit.json']},
        checks=checks,total_accepted=all_accepted,maximum_absolute_pooled_binomial_z=worst_count_z,
        local_exclusive_modes=sum(r['local_exclusive_modes'] for r in task_rows),local_exclusive_tasks=sum(r['local_exclusive_modes']>0 for r in task_rows),
        mean_local_exclusive_fraction_of_bp_proposals=float(np.mean([r['local_exclusive_fraction_of_bp_proposals'] for r in task_rows])),
        local_lower_proposal_tasks=int(np.sum(ratios<1-1e-12)),local_equal_proposal_tasks=int(np.sum(abs(ratios-1)<=1e-12)),local_higher_proposal_tasks=int(np.sum(ratios>1+1e-12)),
        local_acceptance_delta_mean=float(np.mean(gain)),descriptive_paired_task_ci95=ci(gain),
        mean_capture_seconds=float(np.mean([a['capture_seconds'] for a in audits])),mean_box_seconds=float(np.mean([a['box_seconds'] for a in audits])),
        mean_c20_seconds=float(np.mean([a['c20_seconds'] for a in audits])),summaries=summaries,
        scope='old development tasks, no multiplicity correction; proposed union is analytical only and includes global-BP control; no full-time or task-risk victory')
    out.mkdir(parents=True,exist_ok=True)
    for name,data in [('summary.json',result),('tasks.json',task_rows),('resource_subtotals.json',source_bytes)]:
        (out/name).write_text(json.dumps(data,indent=2),encoding='utf-8')
    fig,axes=plt.subplots(1,3,figsize=(14.4,4.8));idx=np.arange(16)
    axes[0].bar(idx,[100*r['local_exclusive_fraction_of_bp_proposals'] for r in task_rows],color='#167c80')
    axes[0].set_ylabel('BP-surviving proposal volume removed (%)');axes[0].set_title('Local certificates add nonzero information')
    axes[1].plot(idx,ratios,'o-',color='#9566ac',label='Local only / BP + residual')
    axes[1].axhline(1,color='#777',ls='--');axes[1].set_yscale('log');axes[1].set_ylabel('Expected proposals for equal accepted count')
    axes[1].set_title('Complementarity is not uniform dominance')
    for m,color in [('c20','#77838d'),('bp_residual','#326bb1'),('local','#167c80')]:
        rr=[r for r in rows if r['method']==m]
        axes[2].scatter([r['reference_expected_acceptance'] for r in rr],[r['empirical_acceptance'] for r in rr],s=13,alpha=.55,label=m,color=color)
    axes[2].plot([1e-6,1],[1e-6,1],color='#999',ls='--');axes[2].set_xscale('log');axes[2].set_yscale('symlog',linthresh=1/16384)
    axes[2].set_xlim(1e-5,.3);axes[2].set_ylim(-1e-5,.3);axes[2].set_xlabel('Reference acceptance Z / Q');axes[2].set_ylabel('Actual accepted / 16,384');axes[2].legend(fontsize=8)
    axes[2].set_title('320 finite-budget sampling runs')
    for ax in axes[:2]:ax.set_xlabel('Development task index');ax.set_xticks(range(0,16,3))
    for ax in axes:ax.grid(alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('Certified empty-volume removal: independent credit contribution, not yet end-to-end gain')
    fig.tight_layout();fig.savefig(out/'rejection_credit.png',dpi=160);plt.close(fig)
    (out/'figure_audit.json').write_text(json.dumps(dict(source_sha256=sha(Path(__file__)),summary_sha256=sha(out/'summary.json'),figure_sha256=sha(out/'rejection_credit.png')),indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
