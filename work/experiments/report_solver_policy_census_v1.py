"""313-315 illustrated, evidence-bounded Chinese mechanism report."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from evaluate_complete_credit_mode_geometry_v1 import read,save,sha


def run(root,out):
    base=root/'results/solver_policy_recurrence'
    inputs=['development_v1','analysis_v1','audit_v1','two_cycle_v1','effective_witness_v1']
    summaries={n:read(base/n/'summary.json') for n in inputs}
    for n,s in summaries.items():
        assert s['passed']
        for name,digest in s.get('outputs_sha256',{}).items():assert sha(base/n/name)==digest
    a=read(base/'analysis_v1/aggregate.json');cycles=read(base/'two_cycle_v1/aggregate.json')
    names=['alm_keep','alm_reset','nodual'];labels=['ALM: keep u','ALM: reset initial u','No dual update']
    fig,axes=plt.subplots(1,2,figsize=(11.4,4.4),layout='constrained')
    xx=np.arange(3);w=.34
    for dx,kind,label,color in [(-w/2,'full','Full solver signature','#2E6F9E'),(w/2,'forward','Forward pattern only','#DB943A')]:
        vals=[100*a[n][kind]['thresholds']['4']['steps_in_segments']/8384 for n in names]
        bars=axes[0].bar(xx+dx,vals,w,label=label,color=color)
        axes[0].bar_label(bars,fmt='%.1f%%',padding=3,fontsize=9)
    axes[0].set_xticks(xx,labels,fontsize=9);axes[0].set_ylim(0,113)
    axes[0].set_ylabel('Share of replayed steps (%)')
    axes[0].set_title('Steps in constant runs of at least four',loc='left',fontsize=11,fontweight='bold')
    axes[0].legend(loc='upper left',fontsize=8,frameon=False)
    vals=[a[n]['success_events']['events_with_completed_segment_at_least_4'] for n in names]
    totals=[a[n]['success_events']['count'] for n in names]
    bars=axes[1].bar(xx,vals,color=['#2E6F9E','#5093BC','#778893'],width=.55)
    axes[1].set_xticks(xx,labels,fontsize=9);axes[1].set_ylim(0,13.2)
    axes[1].set_ylabel('Number of first-arrival events')
    axes[1].set_title('Completed full-signature run >= 4 before arrival',loc='left',fontsize=10.5,fontweight='bold')
    for bar,total,val in zip(bars,totals,vals):axes[1].text(bar.get_x()+bar.get_width()/2,val+.25,f'{val}/{total}',ha='center')
    for ax in axes:ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.13);ax.set_axisbelow(True)
    fig.savefig(out/'policy_stability.png',dpi=180);plt.close(fig)
    chinese={'alm_keep':'ALM保留乘子','alm_reset':'ALM重置起点乘子','nodual':'无乘子更新'}
    table='\n'.join(f"| {chinese[n]} | {a[n]['full']['segments']} | {a[n]['full']['singleton_fraction']:.2%} | {a[n]['full']['thresholds']['4']['trajectories_with_segment']}/131 | {a[n]['full']['thresholds']['4']['steps_in_segments']}/8384 | {a[n]['success_events']['events_with_completed_segment_at_least_4']}/{a[n]['success_events']['count']} |" for n in names)
    cycle_table='\n'.join(f"| {chinese[n]} | {cycles[n]['full']['trajectories_with_cycle']}/131 | {cycles[n]['full']['segments']} | {cycles[n]['full']['covered_steps']}/8384 | {cycles[n]['full']['maximum_length']} | {cycles[n]['full']['success_events_strictly_before']['with_cycle']}/{cycles[n]['full']['success_events_strictly_before']['total']} |" for n in names)
    report=f'''# 313—315：从“标签是否重复”走向“真正的更新公式是否重复”

本轮已完成393条完整轨迹、25,152步回放与逐起点独立复核。结果不支持直接把成功前演化当成长单策略或长二拍周期来跳步；但一个新的精确例子证明，保守策略标签可能过度分裂同一个实际更新公式。因此下一步是做有理代数规范化，而不是任意扩周期搜索或直接宣称加速。

## 1．固定策略下能证明什么

状态s=(b,h,u)。固定完整活动/偏置获胜策略、夹紧、切点排序和残差分支时，局部更新在理想实数算术下有形式Fπ(s)=Mπs+cπ。只有每个中间状态都满足相应守卫，才能用增广矩阵幂表达多步演化。前向分支不变不是完整策略不变；矩阵幂也不能代替守卫和舍入检查。

四层、四支持点的状态为36维，稠密36×36矩阵单独就需10,368字节的float64存储；不能把它算成312的160字节额外历史。这里尚未构建或计时任何在线矩阵推进方法。

## 2．完整旧轨迹的单策略统计

64旧任务、131原位置、3个预定族、各64步全部保留，没有按成功或查询误差筛选。45个无选定位置任务仍保留在任务清单中。744项完整签名包含非获胜候选、平局等信息，是保守分区，不是最小仿射映射编号。

| 轨迹族 | 完整策略段数 | 长度1占段数比例 | 有长度≥4段的轨迹 | 长度≥4段覆盖步数 | 成功首达前已完成≥4段的事件 |
|---|---:|---:|---:|---:|---:|
{table}

![完整策略与前向模式的稳定性差别](policy_stability.png)

ALM保留乘子共有8253次相邻调用转移，其中3267次前向模式未变而完整策略改变。另有50次完整策略未变、前向模式改变：即使更新公式相同，变化的状态也能跨真实前向边界，两者不能互相替代。

11个保留乘子成功首达事件中，10个在一个新的完整策略段首步出现，1个在第二步出现；此前最长已完成策略段分别只有1、2或3步。长稳定段存在，但不能据它们推断成功前的长演化可直接快进。无乘子轨迹更稳定，也不能据此推断发现质量更好。

## 3．固定二拍假设也完整检验了

只检测π_A≠π_B且至少A,B,A,B的真正二拍重复，常量段不算；重叠段覆盖取并集。不是任意周期网格。若真实策略交替，条件性两步公式是F_B(F_A(s))=(M_BM_A)s+M_Bc_A+c_B，但奇数拍和偶数拍的守卫都要成立。

| 轨迹族 | 有真正二拍段的轨迹 | 段数 | 覆盖步数 | 最长段 | 成功首达前已有二拍段的事件 |
|---|---:|---:|---:|---:|---:|
{cycle_table}

保留乘子的二拍签名只覆盖13/8384步，最长5步。这不足以支撑现在实现以长二拍周期为主的加速器；也不能因此证明任何等价映射合并都无效。

## 4．正面数学进展：不同标签可以是同一个仿射映射

在一个观察可由合法教师产生的一层例子里，x=0.1、v=0.4、ε=0.001、τ=0.01。当前b∈[-0.12,-0.08]、h∈[0,1]、u∈[-0.02,0.02]。虽然旧读取g(x+b)在b=-0.1两侧改变分支，活动总被夹到观察带下沿，完整更新在整个状态域上却统一为：

    h' = 0.399
    b' = (0.398 + 2u + 0.01b) / 4.01
    u' = u + 0.0995 - b'

这是覆盖整个域的代数推导，另外在1032个精确有理状态（含8个角点）上用独立完整局部求解器核验。两个实际浮点输入的完整策略标签确实不同，仍符合上述同一公式；十进制有理模型与binary64常量模型严格区分，数值差≤1.39e-17。

该例说明“旧读取的分支变化”可能被后续夹紧完全消掉，正如早期输入方向可能被LayerNorm消掉，分析必须沿完整计算路径进行。它不是新的任务性能结果，也不证明旧四层轨迹必有长等价段。

下一315实现将传播实际获胜公式的精确有理系数，按36维状态的完整系数逐项相等来合并；不删除难看的字段、不按浮点容差合并。完整规范化引擎及真实轨迹规范化尚未实现。

## 5．复核、成本与研究边界

插桩与原Local逐位核对75,456个状态数组，归档参数状态25,152项、首步活动57数组、末步活动/乘子114数组全部一致。另以逐起点方式独立回放全部25,152步，核对301,824组策略、75,456个输出数组及25,152条独立前向编码。段长126个阈值数值、42个汇总字段通过。

二拍检测先穷举长度1至10的全部2046个二值序列，再用独立朴素枚举核对真实786条完整周期清单和96条事件前缀清单。没有使用查询答案，没有改变任何原轨迹，更没有把已有成功前状态作为新候选免费输入。

完整插桩/封存诊断约{summaries['development_v1']['seconds']:.3f}秒，逐起点独立复核约{summaries['audit_v1']['seconds']:.3f}秒。均不是在线运行时间，部分只读诊断曾并行，不能作速度比较。未做同资源新任务查询收益验证，主目标保持active。

证据：[313回放](../development_v1/summary.json)、[313独立审计](../audit_v1/summary.json)、[314二拍普查](../two_cycle_v1/summary.json)、[315精确见证](../effective_witness_v1/summary.json)。完整推导位于 outputs/ttt-pc-alm-research/313_solver_policy_recurrence_design.md、314_two_policy_cycle_protocol.md、315_effective_affine_map_design.md。
'''
    with (out/'report.md').open('x',encoding='utf-8') as stream:stream.write(report)
    save(out/'summary.json',dict(passed=True,source_sha256=sha(Path(__file__)),
        input_summaries_sha256={n:sha(base/n/'summary.json') for n in inputs},
        outputs_sha256={n:sha(out/n) for n in ['policy_stability.png','report.md']},
        visual_qa_pending=True,independent_task_gain_established=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False);run(Path(__file__).resolve().parents[2],args.out)
