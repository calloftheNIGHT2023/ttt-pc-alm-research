"""312 auditable Chinese result note and scientific state-scale figure."""
import argparse
from pathlib import Path
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from evaluate_complete_credit_mode_geometry_v1 import read,save,sha


def run(root,out):
    base=root/'results/paired_primal_dual_predictor'
    folders=[base/'development_v1',base/'geometry_v1',base/'audit_v1']
    summaries=[read(f/'summary.json') for f in folders]
    for f,s in zip(folders,summaries):
        assert s['passed']
        for n,digest in s['outputs_sha256'].items():assert sha(f/n)==digest
    support,evaluation,audit=summaries
    protocol=read(folders[0]/'protocol.json')
    assert sum(t['selected_states']>0 for t in protocol['task_manifest'])==19
    assert support['counts']['alpha_less_than_one']==0
    assert evaluation['counts']['actual_parameter_points_feasible']==0
    diags=read(folders[2]/'state_diagnostics.json')
    fig,ax=plt.subplots(figsize=(8.4,4.6),layout='constrained')
    colors=['#2860A4','#C97B12','#10867A']
    for name,label,color in zip(['db_max','dh_max','du_max'],['Parameters b','Activities h','Multipliers u'],colors):
        values=np.sort(np.maximum([d[name] for d in diags],1e-18))
        ax.step(values,np.arange(1,len(values)+1)/len(values),where='post',label=label,color=color,linewidth=2.2)
    ax.set_xscale('log');ax.set_xlim(7e-19,1.2);ax.set_ylim(0,1.035)
    ax.set_xlabel('Maximum absolute one-step change (zero displayed at 1e-18)')
    ax.set_ylabel('Fraction of 131 selected states')
    ax.set_title('Forward stasis does not mean all internal states stop',loc='left',fontweight='bold')
    ax.grid(alpha=.18);ax.legend(loc='lower right',frameon=False)
    ax.text(.02,.96,'57/131 parameter changes are exactly zero\n89/131 are at most 1e-12',
            transform=ax.transAxes,va='top',fontsize=10,bbox=dict(facecolor='white',edgecolor='none',alpha=.9))
    fig.savefig(out/'state_change_ecdf.png',dpi=180);plt.close(fig)
    labels={'normal_one':'原更新一步','normal_two':'原更新两步','paired_bu':'参数—乘子配对（主候选）',
        'bias_only':'只预测参数','dual_only':'只预测乘子，同步长','dual_full':'只预测乘子，完整一步',
        'full_bhu':'预测参数、活动、乘子','paired_b_zero_u':'预测参数，乘子清零',
        'paired_b_last_du':'预测参数，只留乘子增量','paired_b_random_du':'乘子增量随机符号',
        'random_db_paired_u':'参数增量随机符号','paired_b_current_residual_u':'用当前残差替代乘子增量'}
    table='\n'.join(f"| {labels[n]} | {a['tasks_with_new_positive']} | {a['task_positive_pairs']} | {a['new_pairs_vs_all_308_309']} |" for n,a in evaluation['aggregate'].items())
    c=audit['counts'];norm=audit['state_norms']
    report=f'''# 312：因果参数—乘子配对预测，正式结果

**全量运行与独立核验已完成；本轮的一步配对预测没有发现原池之外的新有效参数区域。** 所有12种预定配置均为0，不能归因为某个对照获得了额外标签或更多参数，也不能把数值正确升级为任务收益。

## 1．检验了什么

保持64个旧开发任务的131个前向停滞位置。利用真实当前与前一步状态，令 Δb=b_t−b_(t−1)、Δu=u_t−u_(t−1)，先预测 b_hat=b_t+αΔb、u_hat=u_t+αΔu，再做一次完整原局部更新。α是盒约束允许的[0,1]最大值，不搜网格。本次131处α全为1，没有被边界截断，也没有α=0退化例。

每种方法保留实际形成的所有参数点，包括预测点，原两步控制也保留第一步点。12方法不合并成免费大候选池。全部提议先封存，再作共同几何检查；没有使用查询答案、教师参数或后验矩进行候选生成。

## 2．完整结果

下表“新”均扣除原有正体积池；有效区域必须有严格内部点的有理证书，不仅是不同的分支编码。

| 配置 | 有新区域的任务/64 | 新任务—区域对 | 超出全部308/309旧方法的对数 |
|---|---:|---:|---:|
{table}

“原池外为0”不等于方法没有访问任何有效旧分支，也不证明所有闭式头优于内部适应。它只说明本轮候选没有提供设定中的新增覆盖收益。64任务全部保留，其中有选定位置的19任务、无选定位置的45任务均计入分母。

所有3275个实际参数点均未直接满足严格观察带；“某分支存在可行参数”和“已经找到了该参数”严格分开。统一评价了1252个不同任务—模式对（含原正体积池），1133个有正体积、119个不可行，无未定；其中仅3项需要新增分类，其余复用并重新核验证书。这里1133不能写成候选新发现数量。

## 3．一步趋势为何不够：已观察到的事实

![三类状态变化的经验分布](state_change_ecdf.png)

131处有{c['zero_db_states']}处参数增量严格为0，{c['db_at_most_1e_minus_12_states']}处最大参数增量≤1e-12；参数增量中位数为{norm['db_max']['median']:.6g}。与此同时活动增量中位数为{norm['dh_max']['median']:.6g}，乘子增量中位数为{norm['du_max']['median']:.6g}。前向停滞并没有让全部内部变量停住。

配对后最终参数与只预测乘子逐位相同的有{c['paired_final_b_equal_dual_only']}处，但不是所有131处；与原一步相同仅{c['paired_final_b_equal_normal_one']}处。最大参数差可到{norm['paired_vs_normal_b_gap']['maximum']:.6g}，仍没有新增有效区域。因此不能只用“参数没变”解释全部结果；即便改变很大，也可能落在不满足观察约束的分支。

这是描述性归因，不是证明所有失败都由小Δb造成。311观察到的成功前参数—乘子协同，不能反推在早期停滞点沿最近一步直线便能复制成功。

## 4．数值与资源复核

1572配置共1703次局部调用，独立Fraction参考最大差{support['max_exact_reference_gap']:.6g}，0超阈值差异、0模式差异。393个过去→当前数组、786个原一步端点数组逐位复现。

另行独立重构4716个预测数组、批量重跑4716个首步输出数组及393个第二步数组，重算3275次前向与3275个真实点的精确约束，复核2685项证书、768个方法集合、60个聚合数值。禁止全局BP/优化器的候选运行检查通过。

理想实数下Δu=.5r_t；本次浮点最大偏差{support['max_du_vs_half_residual_gap']:.6g}。不能把“历史增量”和“当前残差”包装成两项独立信息来源。

支持侧全诊断耗时{support['seconds']:.3f}秒，几何评价{evaluation['seconds']:.3f}秒，独立审计{audit['seconds']:.3f}秒。这些包含封存/参考/I/O或离线几何，不是部署时速度。主候选额外历史数组b/u为160字节，但不含工作区、求解器状态和峰值，不能宣称内存优势。

## 5．下一步准入条件

不扩α网格，不继续换随机方向。下一步先检验真正多步状态演化是否存在可利用的稳定局部求解策略区间：固定完整分支、赢家、夹紧及切点顺序时，整个(b,h,u)更新是仿射递推；但前向分支不变并不足以保证该策略不变。

只有在原完整轨迹上验证这种结构与开销，才考虑事件驱动的多步处理。不能用一次线性外推代替几十步实际状态演化，更不能免费使用未来成功状态。此方向尚未实现，本报告不声称会有效。

本轮仍是旧任务机制开发，未完成独立新任务、匹配资源的查询收益关口，也没有官方TTT或LLM/VLM的新验证。整体目标保持active。

证据：[支持封存](../development_v1/summary.json)、[共同几何](../geometry_v1/summary.json)、[独立核验](../audit_v1/summary.json)；方法协议位于 outputs/ttt-pc-alm-research/312_paired_primal_dual_predictor_protocol.md。
'''
    # A generated report is an artifact; sources and prior results remain immutable.
    with (out/'report.md').open('x',encoding='utf-8') as stream:stream.write(report)
    save(out/'summary.json',dict(passed=True,source_sha256=sha(Path(__file__)),
        input_summaries_sha256={str(f.relative_to(root)):sha(f/'summary.json') for f in folders},
        outputs_sha256={n:sha(out/n) for n in ['state_change_ecdf.png','report.md']},
        visual_qa_pending=True,independent_task_gain_established=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False);run(Path(__file__).resolve().parents[2],args.out)
