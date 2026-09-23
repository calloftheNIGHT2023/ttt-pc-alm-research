"""Audited development result figure and complete77-method tables."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_multiplier_fixed_point_screen import sha,dump


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();src=Path(__file__).parent
    base=root/'results/probe_credit_budget';inp=base/'evaluation';audit=base/'evaluation_audit';out=base/'figures';out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    aa=json.loads((audit/'summary.json').read_text());assert aa['passed'];ap0=json.loads((audit/'protocol.json').read_text());assert sha(inp/'summary.json')==ap0['evaluation_summary_sha256']
    hashes=dict(ap0['source_sha256']);hashes[Path(__file__).name]=sha(Path(__file__))
    for n,h in hashes.items():assert sha(src/n)==h,n
    p=json.loads((inp/'protocol.json').read_text());ss=json.loads((inp/'summary.json').read_text())
    for n,h in ss['outputs_sha256'].items():assert sha(inp/n)==h
    selected=[('credit_control_probe33','Fixed probe + ALM32/64'),('probe_archive_only','Probe archive only'),('probe_then_nodual33','Probe + no-dual32/64'),('probe_nodual64_33','Probe + no-dual64/128 [1.10]'),('probe_then_pc33','Probe + PC32/64'),('probe_pc64_33','Probe + PC64/128'),('probe_pc128_33','Probe + PC128/256 [1.10]'),('probe_then_adam60_33','Probe + Adam60'),('probe_then_adam120_33','Probe + Adam120'),('probe_then_adam240_33','Probe + Adam240'),('probe_then_adam480_33','Probe + Adam480 [1.10]'),('credit_control_alm33','Plain ALM32/64'),('plain_alm64_33','Plain ALM64/128'),('plain_alm128_33','Plain ALM128/256 [1.10]'),('cold__adam240_r33','Cold Adam240, 33 starts'),('cold__prior4096_ridge','Prior ridge4096'),('cold__prior16384_ridge','Prior ridge16384'),('cold__meta_shallow64_20','Meta-shallow20'),('probe_all_alm64','Full probe + ALM64 [over budget]')]
    dump(out/'protocol.json',dict(source_sha256=hashes,evaluation_audit_sha256=sha(audit/'summary.json'),selected=selected,scope='Development only; complete measured main1.00 and sensitivity1.10 costs, all77 methods retained'))
    methods=json.loads((inp/'methods.json').read_text());paired=json.loads((inp/'paired.json').read_text());mm={(r['method'],r['grid']):r for r in methods}
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(14,9.5),layout='constrained');yy=np.arange(len(selected));labels=[l for _,l in selected];colors=['#c7742e' if name==p['primary'] else '#86abad' if mm[name,257]['resource_budget_class']=='main_1.00' else '#c9dadd' for name,_ in selected]
    for ax,metric,title in zip(axes,['mse','actual_excess'],['A. Observed query MSE','B. Conditional excess risk above Bayes']):
        ax.barh(yy,[mm[n,257][metric] for n,_ in selected],color=colors,height=.67)
        ax.set_yticks(yy,labels if ax is axes[0] else ['']*len(labels));ax.invert_yaxis();ax.grid(axis='x',alpha=.2);ax.set_title(title,loc='left',fontweight='bold');ax.set_xlabel('Lower is better')
    fig.suptitle('Can the local-credit gain withstand time-matched and1.10-budget controls?',fontweight='bold',fontsize=14)
    fig.supxlabel('64 reused development tasks; budgets calibrated on8 contexts with3 repeats, not a per-task or deployment guarantee.',fontsize=9)
    fig.savefig(out/'286_probe_credit_budget_development.png',dpi=175);plt.close(fig)
    lines=['# All77 methods and all608 comparisons','',p['scope'],'']
    for grid in [257,129]:
        lines.extend([f'## {grid}-point grid','',
            '| Method | Query MSE | Point MSE | Actual excess | Ideal policy excess | Expected MC | Empty pools | Resource class | Calibrated seconds |','|---|---:|---:|---:|---:|---:|---:|---|---:|'])
        for name in p['methods']:
            r=mm[name,grid];lines.append('| '+name+' | '+' | '.join(f'{r[f]:.10g}' for f in ['mse','point_mse','actual_excess','policy_excess','expected_mc'])+f" | {r['empty_pool_tasks']} | {r['resource_budget_class']} | {r['resource_calibration_mean_seconds'] if r['resource_calibration_mean_seconds'] is not None else '-'} |")
        lines.extend(['','| Control | Metric | Primary minus control | CI lower | CI upper | Lower/equal/higher tasks |','|---|---|---:|---:|---:|---|'])
        for r in paired:
            if r['grid']==grid:
                lo,hi=r['descriptive_ci95'];lines.append(f"| {r['control']} | {r['metric']} | {r['mean_difference']:.10g} | {lo:.10g} | {hi:.10g} | {r['lower_tasks']}/{r['equal_tasks']}/{r['higher_tasks']} |")
        lines.append('')
    (out/'all_methods.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    ans=dict(passed=True,method_grid_tables=154,paired_intervals=608,protocol_sha256=sha(out/'protocol.json'),outputs_sha256={n:sha(out/n) for n in ['286_probe_credit_budget_development.png','all_methods.md']})
    dump(out/'summary.json',ans);print(json.dumps(ans),flush=True)


if __name__=='__main__':main()
