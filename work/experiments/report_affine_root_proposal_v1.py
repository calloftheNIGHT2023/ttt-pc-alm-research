"""316 post-hoc invariant-coordinate diagnosis and audited-count report."""
import argparse
from collections import Counter,defaultdict
from fractions import Fraction as F
from pathlib import Path
import time
import traceback
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from evaluate_complete_credit_mode_geometry_v1 import read,save,sha


def run(root,out):
    source=root/'results/affine_root_proposal/development_v1';audit=root/'results/affine_root_proposal/audit_v1'
    audited=read(audit/'summary.json');assert audited['passed']
    aggregate=read(audit/'aggregate.json');assert sha(audit/'aggregate.json')==audited['aggregate_sha256']
    manifest=read(source/'before_aggregation_manifest.json')
    for name,digest in manifest['files_sha256'].items():assert sha(source/name)==digest
    families=['alm_keep','alm_reset','nodual'];extra={f:dict(counts=Counter(),fixed_tasks=set(),float_fixed_gaps=[],
                                                        distinct_bias=set(),invariant_witnesses=[]) for f in families}
    for r in read(source/'rows.json'):
        record=read(source/r['file']);sol=record['solution'];group=extra[r['family']]
        if r['status']=='actual_fixed_point':
            group['fixed_tasks'].add(r['seed']);group['float_fixed_gaps'].append(record['float_candidate_fixed_gap'])
            group['distinct_bias'].add((r['seed'],*sol['full_root'][:4]))
        if r['status']!='root_outside_domain':continue
        point=list(map(F,sol['full_root']));v=list(map(F,record['v']));n=len(v);d=len(point)//(1+2*n)
        indices=sol['active'];columns={k:j for j,k in enumerate(indices)}
        basis=[list(map(F,row)) for row in sol['nullspace']]
        witness=None
        for j in range(d+d*n):
            lo,hi=-F(.12),F(.12)
            if j>=d:
                layer,i=divmod(j-d,n);lo,hi=F(0),F(1)
                if layer==d-1:lo,hi=max(lo,v[i]-F(.001)),min(hi,v[i]+F(.001))
            if not lo<=point[j]<=hi and all(vector[columns[j]]==0 for vector in basis):
                witness=dict(file=r['file'],coordinate=j,value=point[j],lower=lo,upper=hi,
                             complete_nullspace_dimension=len(basis),coordinate_constant_on_entire_root_set=True)
                break
        if witness is not None:
            group['counts']['all_roots_outside_domain_certified']+=1;group['invariant_witnesses'].append(witness)
        else:group['counts']['outside_chosen_root_only_unresolved']+=1
    for f,g in extra.items():
        g['fixed_tasks']=sorted(g['fixed_tasks']);g['unique_fixed_bias_per_task']=len(g.pop('distinct_bias'))
        g['max_float_gap_at_true_fixed_points']=max(g.pop('float_fixed_gaps'),default=None)
        g['counts']=dict(g['counts'])
    save(out/'invariant_coordinate_diagnosis.json',dict(post_hoc=True,method='Coordinate invariant on complete exact nullspace',families=extra))
    colors=['#476B9D','#D98B44','#BA5867','#489C86']
    categories=['no_fixed_point_in_selected_formula','root_outside_domain','root_wrong_actual_update','actual_fixed_point']
    labels=['No algebraic fixed point','Chosen root outside domain','Domain-valid, wrong update','Verified fixed point']
    fig,axes=plt.subplots(1,2,figsize=(11.8,4.7),gridspec_kw={'width_ratios':[1.5,1]},layout='constrained')
    left=np.zeros(3)
    for key,label,color in zip(categories,labels,colors):
        values=np.array([aggregate[f]['counts'].get(key,0) for f in families])
        axes[0].barh(np.arange(3),values,left=left,label=label,color=color)
        for j,value in enumerate(values):
            if value>=8:axes[0].text(left[j]+value/2,j,str(value),ha='center',va='center',color='white',fontsize=11)
        left+=values
    assert np.all(left==131)
    axes[0].set_yticks(np.arange(3),['ALM: retained dual','ALM: reset dual','No dual']);axes[0].invert_yaxis()
    axes[0].set_xlabel('Current starting states (131 per family)');axes[0].set_title('One fixed-formula root proposal')
    axes[0].set_xlim(0,140);axes[0].legend(loc='upper center',bbox_to_anchor=(.5,-.16),ncol=2,fontsize=8,frameon=False)
    fixed=[aggregate[f]['counts'].get('actual_fixed_point',0) for f in families]
    feasible=[aggregate[f]['counts'].get('fixed_support_feasible',0) for f in families]
    xx=np.arange(3);axes[1].bar(xx-.18,fixed,.35,label='Verified fixed point',color=colors[3])
    axes[1].bar(xx+.18,feasible,.35,label='Also fits observation band',color='#806AA6')
    axes[1].set_xticks(xx,['Retain','Reset','No dual']);axes[1].set_ylim(0,48);axes[1].set_title('A fixed point need not fit support')
    axes[1].set_ylabel('Number of states');axes[1].legend(loc='upper left',fontsize=8,frameon=False)
    for j in range(3):
        axes[1].text(j-.18,fixed[j]+.8,str(fixed[j]),ha='center')
        axes[1].text(j+.18,feasible[j]+.8,str(feasible[j]),ha='center')
    fig.suptitle('316 | Exact root-or-drift test on existing development states',fontsize=14)
    fig.savefig(out/'root_or_drift.png',dpi=180);plt.close(fig)
    lines=['# 316 阶段结果：固定分支求根与乘子漂移证书','',
           '结论：本轮没有得到新的支持可行 ALM 根，也没有未见查询收益结论。获得了精确可复核的机制区分：ALM 当前公式大多没有固定点；无乘子组可直接解到一批有残差的固定点。下一步必须设计实际的分支推进动作，而非把无根证书当成动作本身。','',
           '![三族固定点与支持可行性](root_or_drift.png)','',
           '## 1. 范围与结果','',
           '旧开发集共 64 个任务，其中 19 个任务有 131 个触发位置，45 个任务没有触发；每个位置比较三族，共 393 个状态。每次重新执行当前局部步并构建公式，不用未来状态选择根；查询答案没有访问。','',
           '| 方法 | 无代数根 | 根越界 | 域内但非真实固定点 | 真实固定点 | 其中满足观察带 |',
           '|---|---:|---:|---:|---:|---:|']
    cn={'alm_keep':'保留乘子','alm_reset':'重置乘子','nodual':'无乘子'}
    for f in families:
        c=aggregate[f]['counts'];lines.append('| '+cn[f]+' | '+' | '.join(str(c.get(k,0)) for k in categories+['fixed_support_feasible'])+' |')
    lines+=['','注意：无乘子限制 u=0，仅求解 b/h；没有偷偷引入非零乘子。42 个真实固定点不是 42 个独立任务，来自 '+str(len(extra['nodual']['fixed_tasks']))+' 个任务、'+str(extra['nodual']['unique_fixed_bias_per_task'])+' 个按任务去重的偏置向量。37 个根降低了支持均方误差，但全部未满足观察带；不把这种支持改善当作泛化收益。','',
            '## 2. 数学含义','',
            '固定公式 F(s)=Ms+c；令 A=I-M。求根为 As=c。一致时，自由变量保留当前值；不一致时输出 w，逐项精确验证 w^T A=0、w^T c=1。因此只要实际更新沿用此公式，w^T s 每步增加 1。','',
            '这不是整个任务不可行的证书，不可直接剪除前向模式。即使一直增长，乘子也可能无界，因此本轮尚未证明有限步必定离开公式区域。ALM 的真实固定点应有零残差，而无乘子固定点没有这个保证。','',
            '## 3. 越界是否只是自由变量选得不好（结果后诊断）','',
            '对完整精确零空间逐坐标检查：如果某个越界坐标在所有零空间方向上都不变，则所有代数根都在该坐标越界。这是根集合结论；若未找到这种坐标，仍保留未决，不声称根集合不可行。','',
            '| 方法 | 所有根必定越界的证书 | 仅当前根越界、未决 |','|---|---:|---:|']
    for f in families:
        c=extra[f]['counts'];lines.append(f"| {cn[f]} | {c.get('all_roots_outside_domain_certified',0)} | {c.get('outside_chosen_root_only_unresolved',0)} |")
    lines+=['','## 4. 独立核验','',
            '- 294 个合成线性系统与既有独立 RREF 一致；含极小非零主元、非唯一根和无界漂移反例。48 个合成网络状态及一个已知精确 ALM 固定点通过。',
            '- 全部 393 个真实系统独立秩/一致性检查；246 个左零空间证书，738 个非当前点漂移恒等式检查。',
            '- 147 个根及 154 个零空间向量核验；54 个域内根通过独立三点能量插值，核对 1080 个标量块、2666 个分段二次区间。该方法不复用主求解器的候选公式。',
            '- 精确数学固定与浮点执行分开。42 个真实固定点的最大浮点重算偏差为 '+str(extra['nodual']['max_float_gap_at_true_fixed_points'])+'。',
            '', '## 5. 成本与贡献边界','',
            '以下为当前开发实现每状态的阶段计时均值，不包括旧触发位置的获取成本，也不是匹配预算的性能实验。精确消元涉及整个状态，不能称为免费或纯局部计算。','',
            '| 方法 | 当前局部步（秒） | 全部核心计算（秒） | 最大有理数位数 |','|---|---:|---:|---:|']
    for f in families:
        a=aggregate[f];lines.append(f"| {cn[f]} | {a['seconds']['current_local_sweep']/131:.6f} | {a['mean_total_core_seconds']:.6f} | {a['max_rational_bits']} |")
    lines+=['','固定点/Newton 加速已有 [Newton-ADMM (Ali et al., ICML 2017)](https://proceedings.mlr.press/v70/ali17a.html) 等先例，不能把线性求根作为新贡献，且其凸优化收敛条件不能直接套到这里。独立收益仍须新任务、匹配资源、强回归/浅层头/同参数强优化器比较。','',
            '## 6. 下一步','',
            '优先研究可证明的分支推进：能否把无根公式的漂移方向与实际求解器的有效条件结合，得到合法的离开/进入位置。必须同时处理参数、活动和乘子，验证真实完整更新；不能只改变乘子幅度重跑已完成的 308 网格，也不能把本轮证书直接当成网络区域不可行证书。','',
            '本报告不是论文胜负结论；goal 保持 active。所有原始候选、失败条件和费用保留。','']
    (out/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    save(out/'summary.json',dict(passed=True,query_targets_accessed=False,independent_task_gain_established=False,
                                source_sha256=sha(Path(__file__)),input_audit_sha256=sha(audit/'summary.json'),
                                files_sha256={n:sha(out/n) for n in ['report.md','root_or_drift.png','invariant_coordinate_diagnosis.json']}))
    print({f:{'counts':extra[f]['counts'],'fixed_tasks':extra[f]['fixed_tasks'],
               'true_fixed_float_gap':extra[f]['max_float_gap_at_true_fixed_points']} for f in families},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    try:run(Path(__file__).resolve().parents[2],args.out)
    except Exception:
        save(args.out/'failure.json',dict(traceback=traceback.format_exc()));raise
