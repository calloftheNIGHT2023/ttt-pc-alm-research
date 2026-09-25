"""Verified 319/320 outcome and 321 structural gap, not an online win claim."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from evaluate_complete_credit_mode_geometry_v1 import read,save,sha


def run(root,out):
    main=root/'results/factorized_dual_branch_search'
    results=read(main/'geometry_v1/summary.json');audit=read(main/'audit_v1/summary.json')
    diagnostics=root/'results/branch_image_chain/diagnostic_v1/summary.json';diag=read(diagnostics)
    proposals=read(main/'development_v1/summary.json')
    assert results['passed'] and audit['passed'] and diag['passed']
    labels=['ALM dual','Dual + residual','Residual','BP credit','Random sign','Zero credit']
    names=list(results['aggregate']);x=np.arange(len(names))
    removed=np.array([diag['aggregate'][n]['excluded'] for n in names])
    survivors=np.array([diag['aggregate'][n]['survivors'] for n in names])
    fig,(ax,diagram)=plt.subplots(1,2,figsize=(14,6.5),gridspec_kw={'width_ratios':[1.35,1]})
    ax.barh(x,removed,color='#dc8653',label='Provably incompatible by image check')
    ax.barh(x,survivors,left=removed,color='#779db5',label='Survives check; may still be infeasible')
    for i,total in enumerate(removed+survivors):ax.text(total+25,i,str(total),va='center',fontsize=10)
    ax.set_yticks(x,labels);ax.invert_yaxis();ax.set_xlim(0,max(removed+survivors)*1.18)
    ax.set_xlabel('Distinct task-mode proposals (channels overlap)')
    ax.set_title('A. Structural gaps in the sealed 320 proposals',loc='left',pad=16,fontsize=12)
    ax.legend(loc='upper left',bbox_to_anchor=(0,-.14),frameon=False,fontsize=9)
    ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',alpha=.16);ax.set_axisbelow(True)
    diagram.axis('off');diagram.set_title('B. A constraint retained by the new selector',loc='left',pad=16,fontsize=12)
    boxes=[(.5,.84,'Previous layer: flat activation branch','#e5ecf2'),
           (.5,.63,'Actual activation must be h = 0','#e5ecf2'),
           (.5,.42,'Shared bias bound: |b| <= 0.12\nNext preactivation: z in [-0.12, 0.12]','#e5ecf2'),
           (.5,.18,'Cannot select the branch z in [0.5, 1]','#f6dfd1')]
    for xx,yy,label,color in boxes:
        diagram.text(xx,yy,label,ha='center',va='center',fontsize=11,
                     bbox=dict(boxstyle='round,pad=.8',facecolor=color,edgecolor='none'))
    for y1,y2 in [(.79,.69),(.57,.49),(.34,.25)]:
        diagram.annotate('',xy=(.5,y2),xytext=(.5,y1),arrowprops=dict(arrowstyle='->',color='#526b7a',lw=1.7))
    fig.suptitle('Feasibility structure first; credit-source advantage still requires testing',fontsize=15,y=.99)
    fig.subplots_adjust(top=.86,bottom=.2,wspace=.36,left=.11,right=.97)
    fig.savefig(out/'mechanism.png',dpi=160);plt.close(fig)
    rows=[]
    for name,label in zip(names,labels):
        r=results['aggregate'][name];p=proposals['aggregate'][name];d=diag['aggregate'][name]
        rows.append(f"| {label} | {p['current_certified_infeasible']} | {r['proposal_pairs']} | {d['excluded']} | {r['positive_pairs']} | {r['new_vs_prior']} |")
    text='''# 从退出快进到直接分支选择：319–321阶段报告

## 结论

319已完成退出后的五组续接与独立复核：匹配等效迭代长度后，前向模式及有效区域集合相同，最大完整状态差约3e-15。它支持沿校正状态轨迹跳过停滞段的等价性，但不证明净加速或新任务收益。

320实现了新的精确K最佳分支选择器：用逐层优化对偶下界，在给定Hamming改动数量下寻找完整网络的最佳分支组合。它不是继续拉长319轨迹，也不是对局部活动独立翻标签。数学与小规模全枚举检查成立；但旧任务中没有新增有效区域，不能据此声称优于回归或同参数BP。

321进一步定位了结构缺口，并已实现保留相邻层分支像的链式选择器。此报告只给出其已完成的结构诊断和原语验证；新的131起点主实验、几何和最终核验以独立目录为准，不把未完成结果写成成功。

![候选结构缺口与修正原理](mechanism.png)

## 320：同起点、同选择器、六种信用

全部64旧任务保留，其中19个任务提供131个触发状态，45个无触发任务保留为空池。每状态六通道、每壳K=8、至多三个壳，生成18842次提议；按任务-模式去重后共同几何共5107项。参数、观测、先验与起点相同。没有读取查询答案。

| 信用来源 | 当前区域正证书/131 | 去重提议 | 分支像检查排除 | 正体积模式 | 相对旧控制新增 |
|---|---:|---:|---:|---:|---:|
'''+ '\n'.join(rows)+'''

不同通道可能重复同一模式，不能将正体积列直接相加为独立发现数。共同分类为5103不可行、4正体积；4项均已在旧控制池中。模式有正体积也不意味着当前参数已拟合支持，更不等于查询泛化改善。

## 独立核验与计算边界

320主提议用时101.724秒，统一几何48.166秒，独立审计68.133秒，均为本机诊断时间而非公平部署基准。独立审计核对393个来源数组、786个信用数组、2244个不同精确层表、11562个全局最小壳值、18842个建议下界、245个当前区域不可行结论、16703个几何证书及所有通道集合。K最佳次序用6个可全枚举小问题检查，未声称穷举实际4^16空间。

真实运行仍需取得原ALM状态；历史归档与信用不能免费使用。Fraction表、DP、几何、读出、Python/原生峰值都必须计费。本轮未进行新冻结查询风险、官方TTT完整对照或LLM/VLM验证，也未建立独立任务优势。

## 321：修正明确的数学缺口

320将内部活动统一放在[0,1]，但平坦激活分支的输出实际只能为0。这使部分建议组合在相邻层间不可能实现。对全部5107模式，基本分支像检查精确排除3463项（67.81%）：2793项有空共享偏置区间，1239项在末层与观测冲突，两类有重叠；4个正体积模式全部保留。剩余1644项仍有1640不可行，所以该检查不是完整可行性求解器。

固定分支和信用，新域是旧松弛域的子集，故 D_chain >= D_row；真实可行模式仍满足 D_chain <= 0。严格支配例、相等例、24个可行参数不误排、6个小问题全枚举已通过。这是证书下界的保证，不是查询误差的保证。

新选择器把“上一层哪些观察输出固定为零”作为DP状态，共2^n种掩码；各层整行仍有4^n种。结构增强给全部六通道共享。只有增强后的同算法比较出现乘子独立增量，才有理由进入完整成本与冻结新任务收益验证。指数复杂度是少样本小模型适用限制，不能外推长上下文部署。

## 文件

- 319：results/post_escape_continuation，交接319_execution_handoff_v1.md。
- 320：results/factorized_dual_branch_search，协议320_factorized_dual_branch_search_protocol_v1.md。
- 321：results/branch_image_chain，设计321_branch_image_chain_design_v1.md及执行细节321_execution_details_v1.md。

全部为本地结果，本报告不代表已推送GitHub。研究总目标仍未完成。
'''
    (out/'report.md').write_text(text,encoding='utf-8')
    save(out/'summary.json',dict(passed=True,source_sha256=sha(Path(__file__)),
         input_sha256={str(p.relative_to(root)):sha(p) for p in [main/'geometry_v1/summary.json',main/'audit_v1/summary.json',diagnostics]},
         table_fields_checked=36,figure_numeric_values={'excluded':removed.tolist(),'survivors':survivors.tolist()},
         outputs_sha256={n:sha(out/n) for n in ['report.md','mechanism.png']},visual_qa_performed=False))
    print(str(out/'report.md'),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False);run(Path(__file__).resolve().parents[2],args.out)
