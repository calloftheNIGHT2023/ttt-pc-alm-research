"""Audited conditional-risk figures and complete 46-method tables, no new fitting."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_multiplier_fixed_point_screen import sha,dump

SELECTED = [
    ('minimum_dual_alm64','Proposed dual + ALM64'),
    ('minimum_activity_alm64','Same activity; no injected dual'),
    ('minimum_reset_alm64','Reset dual before continuation'),
    ('alm65','Ordinary warm ALM65'),
    ('cold__alm128','Cold ALM128'),
    ('adam960','Warm Adam960'),
    ('cold__gn40_r33','Cold GN40, 33 starts'),
    ('cold__adam240_r33','Cold Adam240, 33 starts'),
    ('cold__prior16384_ridge','Prior-feature ridge, 16384'),
    ('cold__meta_ridge128','Meta-ridge128'),
    ('cold__meta_shallow64_20','Meta-shallow, 20 steps'),
    ('probe_all_alm64','All probes + ALM64')]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True)
    root=ap.parse_args().project.resolve();src=Path(__file__).parent
    base=root/'results/confirmation_conditional_risk';inp=base/'decomposition';out=base/'figures'
    out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    audit=base/'decomposition_audit';aa=json.loads((audit/'summary.json').read_text());assert aa['passed']
    ap0=json.loads((audit/'protocol.json').read_text());assert sha(inp/'summary.json')==ap0['decomposition_summary_sha256']
    p=json.loads((inp/'protocol.json').read_text());ss=json.loads((inp/'summary.json').read_text())
    for n,h in ss['outputs_sha256'].items():assert sha(inp/n)==h
    hashes=dict(ap0['source_sha256']);hashes[Path(__file__).name]=sha(Path(__file__))
    for n,h in hashes.items():assert sha(src/n)==h,n
    dump(out/'protocol.json',dict(source_sha256=hashes,design_sha256=p['design_sha256'],
        decomposition_audit_sha256=sha(audit/'summary.json'),selected_methods=SELECTED,
        phase_accesses_query_targets=False,scope='Post-result presentation, all46 methods in complete tables; selection is not a new test'))
    methods=json.loads((inp/'methods.json').read_text());paired=json.loads((inp/'paired.json').read_text())
    mm={(r['method'],r['grid']):r for r in methods};ci={(r['control'],r['grid'],r['metric']):r for r in paired}
    rows=json.loads((inp/'rows.json').read_text());lookup={(r['seed'],r['method'],r['grid']):r for r in rows}
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,(left,right)=plt.subplots(1,2,figsize=(15,7.7),gridspec_kw={'width_ratios':[1.12,1]},layout='constrained')
    yy=np.arange(len(SELECTED));labels=[label for _,label in SELECTED]
    pol=np.array([mm[n,257]['policy_excess'] for n,_ in SELECTED]);act=np.array([mm[n,257]['actual_excess'] for n,_ in SELECTED])
    left.barh(yy,pol,color=['#d17827']+['#b8d4d4']*(len(yy)-1),height=.68,label='Ideal discovery policy (or frozen head)')
    left.plot(act,yy,'o',color='#183f54',markersize=5,label='Frozen actual prediction',linestyle='none')
    left.set_yticks(yy,labels);left.invert_yaxis();left.set_xlabel('Mean conditional excess MSE above Bayes')
    left.set_title('A. Missing explanations vs. readout noise',loc='left',fontweight='bold')
    left.grid(axis='x',alpha=.2);left.legend(loc='lower right',fontsize=8)
    for offset,metric,color,label in [(-.12,'actual_excess','#183f54','Actual'),(.12,'policy_excess','#d17827','Ideal policy')]:
        for i,(name,_) in enumerate(SELECTED[1:],1):
            r=ci[name,257,metric];lo,hi=r['descriptive_ci95'];mean=r['mean_difference']
            right.errorbar(mean,i+offset,xerr=np.array([[mean-lo],[hi-mean]]),fmt='o',markersize=4,
                color=color,capsize=2,label=label if i==1 else None)
    right.axvline(0,color='#6d777e',linewidth=1);right.set_yticks(yy,labels);right.set_ylim(left.get_ylim())
    right.tick_params(axis='y',labelleft=False);right.set_xlabel('Proposed minus control (negative favors proposed)')
    right.set_title('B. Paired task differences: descriptive 95% CIs',loc='left',fontweight='bold')
    right.grid(axis='x',alpha=.2);right.legend(loc='lower left',fontsize=8)
    fig.suptitle('64 frozen tasks | posterior conditioned only on 4 observations | no query answers',fontsize=14,fontweight='bold')
    fig.supxlabel('Post-result mechanism analysis, not a new blind confirmation. Numerical volumes + 4 independent regional MC batches.',fontsize=9)
    fig.savefig(out/'268_conditional_risk.png',dpi=175);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5.6),layout='constrained')
    candidates=[('adam960','Adam960'),('minimum_activity_alm64','Same activity'),('cold__gn40_r33','GN33'),('cold__adam240_r33','Adam33')]
    for name,label in candidates:
        a=np.array([lookup[s,name,257]['policy_excess'] for s in p['seeds']])
        b=np.array([lookup[s,p['primary'],257]['policy_excess'] for s in p['seeds']])
        axes[0].scatter(a,b,label=label,s=20,alpha=.65)
    lim=max(axes[0].get_xlim()[1],axes[0].get_ylim()[1]);axes[0].plot([0,lim],[0,lim],'--',color='#8c969c',linewidth=1)
    axes[0].set(xlabel='Control ideal-policy excess risk',ylabel='Proposed ideal-policy excess risk',title='A. All 64 tasks retained for each control')
    axes[0].legend(fontsize=8);axes[0].grid(alpha=.2)
    internal=SELECTED[:8];xx=np.arange(len(internal));width=.27
    for shift,field,color,label in [(-width,'read_error','#247e82','Fixed readout squared error'),(0,'cross_term','#c87129','Signed cross term'),(width,'expected_mc','#7893a8','Expected resampling variance')]:
        axes[1].bar(xx+shift,[mm[n,257][field] for n,_ in internal],width,color=color,label=label)
    axes[1].axhline(0,color='#88939a',linewidth=.8)
    axes[1].set_xticks(xx,['Dual','Activity','Reset','ALM65','ALM128','Adam960','GN33','Adam33'],rotation=35,ha='right')
    axes[1].set_ylabel('Mean conditional excess-risk contribution')
    axes[1].set_title('B. Readout terms are shown separately, not clipped')
    axes[1].legend(fontsize=8);axes[1].grid(axis='y',alpha=.2)
    fig.suptitle('Discovery policy and Monte Carlo readout are distinct error sources',fontweight='bold')
    fig.savefig(out/'268_discovery_and_readout.png',dpi=175);plt.close(fig)
    lines=['# All frozen methods: conditional risk decomposition','',
        'Post-result diagnostic. All numbers use the fixed numerical-volume posterior. Negative estimates retained; no hidden task parameters or query answers are used.','']
    for grid in [257,129]:
        lines.extend([f'## {grid}-point arithmetic mean','',
            '| Method | Actual excess | Ideal-policy excess | Fixed read error | Signed cross | Expected MC | Conditional total | Empty pools | Complete seconds |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|'])
        for n in p['methods']:
            r=mm[n,grid]
            vals=[f"{r[f]:.10g}" for f in ['actual_excess','policy_excess','read_error','cross_term','expected_mc','conditional_total']]
            lines.append('| '+n+' | '+' | '.join(vals)+f" | {r['empty_pool_tasks']} | {r['mean_complete_call_seconds']:.7g} |")
        lines.extend(['','### Primary minus every control; unadjusted descriptive 95% task-bootstrap intervals','',
            '| Control | Metric | Difference | CI lower | CI upper | Lower/equal/higher tasks |','|---|---|---:|---:|---:|---|'])
        for r in paired:
            if r['grid']!=grid:continue
            lo,hi=r['descriptive_ci95'];lines.append(f"| {r['control']} | {r['metric']} | {r['mean_difference']:.10g} | {lo:.10g} | {hi:.10g} | {r['lower_tasks']}/{r['equal_tasks']}/{r['higher_tasks']} |")
        lines.append('')
    (out/'all_methods.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    sens=json.loads((inp/'sensitivity.json').read_text());gaps=json.loads((inp/'reference_gaps.json').read_text())
    detail=[]
    for n,_ in SELECTED:
        r=mm[n,257];rr=[s for s in sens if s['method']==n]
        detail.append(dict(**r,grid_actual_mean_change=float(np.mean([s['grid_257_minus_129']['actual_excess'] for s in rr])),
            grid_actual_max_abs_change=float(max(abs(s['grid_257_minus_129']['actual_excess']) for s in rr)),
            primary_minus_control={metric:ci[n,257,metric] for metric in p['paired_metrics']} if n!=p['primary'] else {}))
    stats=dict(selected=detail,mean_reference_pair_l2=float(np.mean([r['pair_l2'] for r in gaps if r['grid']==257])),
        maximum_reference_pair_l2=float(np.max([r['pair_l2'] for r in gaps if r['grid']==257])),
        maximum_grid_actual_change_all_methods=float(max(abs(s['grid_257_minus_129']['actual_excess']) for s in sens)),
        maximum_pair_actual_change_all_methods=float(max(abs(s['pair01_minus_pair23']['actual_excess']) for s in sens)))
    dump(out/'report_statistics.json',stats)
    ans=dict(passed=True,source_count=len(hashes),methods=46,tables=92,intervals=270,protocol_sha256=sha(out/'protocol.json'),
        outputs_sha256={n:sha(out/n) for n in ['268_conditional_risk.png','268_discovery_and_readout.png','all_methods.md','report_statistics.json']})
    dump(out/'summary.json',ans);print(json.dumps(ans),flush=True)


if __name__=='__main__':main()
