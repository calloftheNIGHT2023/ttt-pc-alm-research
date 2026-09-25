"""303 complete descriptive report with all configurations and task influence."""
import argparse
from pathlib import Path
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from diagnose_counterfactual_conditional_risk_v1 import read,save,sha

FAMILIES=['alm_keep','alm_reset','nodual','pc','adam','alm_random_sign','alm_residual','adam_perturb_001','adam_perturb_004']
GROUPS=['boundary_entry','forward_stasis','boundary_and_forward_stasis','primal_stasis']


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--project',type=Path,required=True)
    root=ap.parse_args().project.resolve();inp=root/'results/observable_trap_continuation/development_v1'
    out=root/'results/observable_trap_continuation/report_v2';out.mkdir(parents=True,exist_ok=False)
    summary=read(inp/'summary.json');assert summary['passed']
    for name,digest in summary['outputs_sha256'].items():assert sha(inp/name)==digest,name
    aggregates=read(inp/'aggregates.json');scores=read(inp/'scores.json')
    lookup={(r['method'],r['grid']):r for r in aggregates}
    indexed={(r['method'],r['grid'],r['seed']):r for r in scores}
    seeds=list(range(5910000,5910064));influence=[]
    # Every group/horizon/control and both grids, not only the favorable mean.
    for group in GROUPS:
        for horizon in [8,32,64]:
            for grid in [257,129]:
                candidate=f'{group}__alm_keep_{horizon}'
                for family in FAMILIES[1:]:
                    control=f'{group}__{family}_{horizon}'
                    d=np.array([indexed[candidate,grid,s]['policy_excess']-indexed[control,grid,s]['policy_excess'] for s in seeds])
                    mean=math.fsum(d)/64
                    leave=(math.fsum(d)-d)/63
                    pairmeans=[math.fsum(indexed[candidate,grid,s]['pairs'][j]['policy_excess']-indexed[control,grid,s]['pairs'][j]['policy_excess'] for s in seeds)/64 for j in range(2)]
                    influence.append(dict(group=group,horizon=horizon,grid=grid,control=family,
                        mean=mean,pair_means=pairmeans,lower=int(np.sum(d < -1e-12)),same=int(np.sum(abs(d)<=1e-12)),
                        higher=int(np.sum(d > 1e-12)),leave_one_out_min=float(leave.min()),leave_one_out_max=float(leave.max()),
                        most_negative_contribution_seed=seeds[int(np.argmin(d))],
                        most_positive_contribution_seed=seeds[int(np.argmax(d))],
                        task_deltas=[dict(seed=s,delta=float(v),leave_one_out_mean=float(l)) for s,v,l in zip(seeds,d,leave)]))
    assert len(influence)==192
    save(out/'task_influence.json',influence)
    primary=next(r for r in influence if r['group']=='forward_stasis' and r['horizon']==64 and r['grid']==257 and r['control']=='adam_perturb_001')
    fig,axes=plt.subplots(1,2,figsize=(13.5,6.8),layout='constrained')
    colors=['#2563eb','#94a3b8','#64748b','#475569','#8b5cf6','#059669','#d97706','#a855f7','#ec4899']
    short=['Keep dual','Reset dual','No dual','PC','Adam','Random dual','Residual dual','Adam + .01','Adam + .04']
    xs=np.arange(9)
    for ax,group,title in zip(axes,GROUPS[:2],['New box-face entries','Support-prediction stasis']):
        for horizon,offset,alpha in [(8,-.25,.4),(32,0,.7),(64,.25,1.)]:
            vals=[lookup[f'{group}__{f}_{horizon}',257]['delta']*1e6 for f in FAMILIES]
            ax.bar(xs+offset,vals,.24,color=colors,alpha=alpha,label=f'{horizon} steps')
        ax.axhline(0,color='black',linewidth=.7)
        ax.set_xticks(xs,short,rotation=45,ha='right',fontsize=8)
        ax.set_ylabel('Ideal pool-risk change vs original (x 0.000001)')
        ax.set_title(title+'\nOLD64; lower is better, not matched full-call cost',fontsize=10)
        ax.legend(fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.34),ncol=3)
    fig.suptitle('303: retained dual has a narrow mean advantage; broad boundary exploration favors perturbed Adam',fontsize=11)
    fig.savefig(out/'all_families.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10.5,4.8),layout='constrained')
    nonzero=[r for r in primary['task_deltas'] if abs(r['delta'])>1e-12]
    ax.bar([str(r['seed']) for r in nonzero],[r['delta']*1e3 for r in nonzero],color=['#2563eb' if r['delta']<0 else '#ef4444' for r in nonzero])
    ax.axhline(0,color='black',linewidth=.7)
    ax.set_ylabel('Keep-dual risk minus perturbed-Adam .01 (x 0.001)')
    ax.set_title('Task influence: all 5 nonzero differences shown; 59/64 ties\nSame support-stasis locations, 64 steps; descriptive old-task analysis')
    fig.savefig(out/'task_influence.png',dpi=170);plt.close(fig)
    text=['# 303｜可观察障碍位置上的108配置：局部线索与强反例','',
      '全部108配置完成：64旧任务×108方法×2网格=13824行。四个选择组与9更新族、8/32/64步完全按执行前协议保留；候选不读取精确驻点证书或查询答案。先封存全部支持轨迹，再读离线区域矩。',
      '', '![全部更新族](all_families.png)', '',
      '## 结果应怎样解释', '',
      '在预测停滞131位置（19任务）上，保留原乘子64步的理想条件超额风险为 .00127495388576，原池 .00130672362865；清零 .00130567133246、随机符号 .00130490821235、残差方向 .00130627039496、普通Adam .00129947045597、扰动.01 Adam .00129602216564、扰动.04 Adam .00129668768713。边界且预测停滞117位置产生相同的所有理想池风险，状态成本更少；这不是新任务确认。',
      '', '在广泛的新边界面2984位置上，保留乘子64步为 .00124951766212，扰动.01 Adam64为 .00096838778282，明显更低。完整原始状态停滞31位置的全部27配置均没有增加正体积分支或改变理想池风险。不能只展示有利选择组而声称整体优势。',
      '', '## 对正向均值做影响诊断', '', '![逐任务差值](task_influence.png)', '',
      f'预测停滞/64步下，保留乘子相对扰动.01 Adam的64任务均值差为{primary["mean"]:.12g}，3个任务更低、2个更高、59个相同（1e−12仅用于描述性平局计数）。两MC批次对差均值分别为{primary["pair_means"][0]:.12g}和{primary["pair_means"][1]:.12g}。',
      '', f'其中5910000单任务差为−.0019022778584，贡献主导了总体均值。逐一删除任一任务后的均值范围[{primary["leave_one_out_min"]:.12g}, {primary["leave_one_out_max"]:.12g}]跨过零。因此现有正向均值不足以声称稳定的任务分布优势；原64任务和所有结果均保留，不删除不利或有利任务重新包装结论。',
      '', 'task_influence.json保存全部4选择组×3步数×2网格×8对照=192配对的所有逐任务差与逐一删除诊断。这些是开发后的描述性分析，不是预注册显著性检验或新任务置信区间。优势主导任务5910000不是301发现的四个精确盒局部极小点任务，所以当前均值差也不能直接归因于那些精确证书。',
      '', '## 数学机制与下一步', '',
      '同一(b,h)状态、同一局部更新目标和块求解器，仅改变初始乘子，确实可能改变跨分支访问；本轮保留信用的风险均值低于清零/随机方向是待检验线索。但是一般ALM压力逃逸定理不能证明这里每个状态满足其前提，更不能推出风险改善。需要在完整在线实现中保留对照，查看保留乘子带来的新分支是否抵得过检测、续接及几何成本。',
      '', '下一步只作真实接口与资源可行性预检：实现支持预测停滞的在线检测及批量续接，验证与303封存轨迹一致、原模型路径不变，再实测完整调用。不会因为这次旧任务均值较好就直接扩大样本追逐微差。必要的强回归/浅头/强BP新冻结比较仍未完成。',
      '', '## 复核与资源边界', '',
      '3115个唯一位置，四组数量2984/131/117/31；45/45/62任务在后三组为空，按协议使用原池。576组初始点/盒检查、64组原ALM下一步、64组reset/nodual b、64组reset/nodual h、6912组前缀包含初始点、1728组嵌套选择池检查均通过。残差归一化最大误差2.22e−16，风险math.fsum检查最大误差3.47e−18。',
      '', '扰动Adam初始点包含在池内；两幅度用同一预定符号向量，未重抽挑选。所有原始b轨迹、支持误差、局部首末活动/乘子与原始起点保存。状态步、求导行、初始化和计时均记录；统一去重批次的诊断计时不是子集独立调用速度，命名状态小计不含工作区，非完整资源匹配。',
      '', '## 全量表：108方法×2网格', '',
      '两对MC批次内积估计全部保留，数值体积并非精确实数体积；风险允许非单调，不对负估计作截断。','',
      '| 方法 | 网格 | 理想超额风险 | 相对原池差 | 对01差 | 对23差 | 覆盖 | 状态步/任务 | 新正区域/任务数 |',
      '|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for r in aggregates:
        text.append(f'| {r["method"]} | {r["grid"]} | {r["policy_excess"]:.12g} | {r["delta"]:.12g} | {r["pair_delta_means"][0]:.12g} | {r["pair_delta_means"][1]:.12g} | {r["mass"]:.10g} | {r["shadow_steps"]:.6f} | {r["new_positive_modes"]} / {r["tasks_with_new_positive_modes"]} |')
    text+=['',f'303摘要SHA256：`{sha(inp/"summary.json")}`。','所有新资料仍是开发/机制/工程预检证据；未完成目标，不声称官方TTT或LLM/VLM结果。']
    with (out/'report.md').open('x',encoding='utf-8') as stream:stream.write('\n'.join(text)+'\n')
    save(out/'summary.json',dict(passed=True,source_sha256=sha(Path(__file__)),input_summary_sha256=sha(inp/'summary.json'),
        aggregate_rows=216,influence_pairs=192,visual_review_pending=True,
        outputs_sha256={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()}))
    print(dict(report=str(out/'report.md'),paired_mean=primary['mean'],leave_one_out_range=[primary['leave_one_out_min'],primary['leave_one_out_max']]),flush=True)


if __name__=='__main__':main()
