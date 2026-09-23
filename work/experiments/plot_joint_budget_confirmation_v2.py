"""Fixed-comparator scientific figures and all-method tables, after audit only."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_multiplier_fixed_point_screen import sha,dump

PRIMARY='minimum_dual_alm64'
DISPLAY=[(PRIMARY,'Candidate: min-dual + ALM64'),('minimum_activity_alm64','Same activity, no injected dual'),('minimum_reset_alm64','Reset dual before continuation'),('alm65','Ordinary warm ALM65'),('cold__alm16','Cold ALM16'),('cold__alm128','Cold ALM128'),('cold__alm256','Cold ALM256'),('adam960','Warm Adam960'),('gn160','Warm GN160'),('cold__gn40_r33','Cold GN40 / 33 starts'),('cold__adam240_r33','Cold Adam240 / 33 starts'),('cold__prior16384_ridge','Fixed-prior ridge / 16384'),('cold__meta_ridge128','Meta-ridge128'),('cold__meta_shallow64_20','Meta shallow head / 20 steps')]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();src=Path(__file__).parent
    base=root/'results/matched_budget_confirmation';inp=base/'joint_evaluation';audit=base/'joint_evaluation_audit';out=base/'figures_v2';out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    assert json.loads((audit/'summary.json').read_text())['passed'];assert json.loads((base/'mode_identity/summary.json').read_text())['passed'];hashes=dict(json.loads((audit/'protocol.json').read_text())['source_sha256']);hashes.update(json.loads((base/'mode_identity/protocol.json').read_text())['source_sha256']);hashes[Path(__file__).name]=sha(Path(__file__))
    for n,h in hashes.items():assert sha(src/n)==h,n
    dump(out/'protocol.json',dict(source_sha256=hashes,audit_sha256=sha(audit/'summary.json'),display_methods=DISPLAY,scope='all46 methods in table/cost plot; fixed key comparators for readable paired plot'))
    methods=json.loads((inp/'methods.json').read_text());paired=json.loads((inp/'paired.json').read_text());identity=json.loads((base/'mode_identity/by_control.json').read_text());mm={v:{r['method']:r for r in rows} for v,rows in methods.items()};pp={v:{r['control']:r for r in rows} for v,rows in paired.items()};ii={r['control']:r for r in identity}
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
    fig,(ax,bx)=plt.subplots(1,2,figsize=(16,9),gridspec_kw={'width_ratios':[1,1.1]});yy=np.arange(len(DISPLAY));labels=[t for _,t in DISPLAY]
    for version,offset,color,label in [('original',-.17,'#bdc6d2','Original failure-policy'),('conditioned',.17,'#2975a8','Common geometry correction')]:
        ax.barh(yy+offset,[mm[version][n]['mean_mse'] for n,_ in DISPLAY],height=.30,color=color,label=label)
    ax.set_yticks(yy,labels);ax.invert_yaxis();ax.set_xlabel('64-task mean query MSE (lower is better)');ax.set_title('A. Same tasks, same information, both implementations',loc='left',fontsize=11);fig.legend(*ax.get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,.005),ncol=2,frameon=False,fontsize=9);ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
    names=DISPLAY[1:];yy=np.arange(len(names))
    for version,offset,color in [('original',-.13,'#9ba6b4'),('conditioned',.13,'#c65142')]:
        means=np.array([pp[version][n]['mean_difference'] for n,_ in names]);intervals=np.array([pp[version][n]['descriptive_bootstrap95'] for n,_ in names]);bx.errorbar(means,yy+offset,xerr=np.stack([means-intervals[:,0],intervals[:,1]-means]),fmt='o',color=color,capsize=2,markersize=4,lw=1)
    bx.set_yticks(yy,[t for _,t in names]);bx.invert_yaxis();bx.axvline(0,color='#303743',lw=1);bx.set_xlabel('Candidate MSE minus control MSE\nnegative favors candidate; task-level 95% descriptive interval');bx.set_title('B. Paired differences; no multiplicity correction',loc='left',fontsize=11);bx.grid(axis='x',alpha=.2)
    fig.suptitle('64 held-out synthetic tasks | 4 context observations | 46 frozen configurations',fontsize=14,y=.995)
    fig.tight_layout(rect=[0,.045,1,.98],w_pad=2);fig.savefig(out/'265_confirmation_quality.png',dpi=180,bbox_inches='tight');plt.close(fig)
    fig,(ax,bx)=plt.subplots(1,2,figsize=(15,8),gridspec_kw={'width_ratios':[1.05,1]})
    allrows=methods['conditioned'];ax.scatter([r['mean_seconds'] for r in allrows],[r['mean_mse'] for r in allrows],c='#9aa5b3',s=27,alpha=.75,label='All 46 configurations')
    highlight=[PRIMARY,'cold__alm16','cold__alm256','adam960','cold__gn40_r33','cold__prior16384_ridge','cold__meta_ridge128']
    for index,name in enumerate(highlight):
        r=mm['conditioned'][name];color=['#c65142','#2975a8','#7b5999','#bf903e','#20806f','#4f77b5','#ac628c'][index];ax.scatter([r['mean_seconds']],[r['mean_mse']],c=color,s=115 if name==PRIMARY else 45,marker='*' if name==PRIMARY else 'o',label='candidate' if name==PRIMARY else name.replace('cold__',''))
    ax.set_xscale('log');ax.set_yscale('log');ax.set_xlabel('Fresh complete call, mean seconds/task (log)');ax.set_ylabel('Mean query MSE (log)');ax.set_title('C. All methods; correction and failed attempts charged',loc='left',fontsize=11);ax.grid(alpha=.18);ax.legend(fontsize=8,loc='upper right',frameon=False)
    names=[(n,t) for n,t in DISPLAY[1:] if n in ii];yy=np.arange(len(names));fractions=[ii[n]['identical_predictions'] for n,_ in names]
    bx.barh(yy,fractions,color='#517f75');bx.set_yticks(yy,[t for _,t in names]);bx.invert_yaxis();bx.set_xlim(0,68);bx.set_xlabel('Tasks with bytewise identical final readout (out of 64)');bx.set_title('D. Readout identity, not support-fit or parameter identity',loc='left',fontsize=11)
    for y,n in zip(yy,fractions):bx.text(n+.6,y,str(n),va='center',fontsize=9)
    bx.grid(axis='x',alpha=.2);bx.set_axisbelow(True);fig.tight_layout(w_pad=2.5);fig.savefig(out/'265_confirmation_cost_and_identity.png',dpi=180,bbox_inches='tight');plt.close(fig)
    text=['# All 46 configurations: same 64 tasks','','Original and corrected versions are both retained. Intervals are unadjusted task-level paired bootstrap intervals. Times are separately measured full calls; not a before/after speedup experiment.','', '| Method | Original MSE | Corrected MSE | Corrected seconds | Primary minus control [95%] | Better / same / worse |','|---|---:|---:|---:|---|---|']
    for r in sorted(allrows,key=lambda r:r['mean_mse']):
        n=r['method'];pair=pp['conditioned'].get(n);interval='primary' if pair is None else f"{pair['mean_difference']:.9f} [{pair['descriptive_bootstrap95'][0]:.9f}, {pair['descriptive_bootstrap95'][1]:.9f}]";signs='—' if pair is None else f"{pair['lower']} / {pair['same']} / {pair['higher']}"
        text.append(f"| {n} | {mm['original'][n]['mean_mse']:.9f} | {r['mean_mse']:.9f} | {r['mean_seconds']:.6f} | {interval} | {signs} |")
    # Generated artifact, not a source edit.
    (out/'all_methods.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    outputs={n:sha(out/n) for n in ['265_confirmation_quality.png','265_confirmation_cost_and_identity.png','all_methods.md']}
    ans=dict(passed=True,protocol_sha256=sha(out/'protocol.json'),outputs_sha256=outputs);dump(out/'summary.json',ans);print(json.dumps(ans),flush=True)

if __name__=='__main__':main()


