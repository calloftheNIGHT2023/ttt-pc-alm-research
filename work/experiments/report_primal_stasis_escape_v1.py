"""318 result report from the sealed support, replay and geometry summaries."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from evaluate_complete_credit_mode_geometry_v1 import read,save,sha


def run(root,out):
    base=root/'results/primal_stasis_escape';audit=read(base/'audit_v1/summary.json');assert audit['passed']
    assert read(base/'geometry_audit_v2/summary.json')['passed']
    groups=read(base/'audit_v1/aggregate.json');geo=read(base/'geometry_v1/summary.json');assert geo['passed']
    families=['alm_keep','alm_reset','nodual'];names=['Retained dual','Reset dual','No dual'];cn=['保留乘子','重置乘子','无乘子']
    applicable=[groups[f]['counts'].get('applicable',0) for f in families]
    exits=[groups[f]['counts'].get('first_primal_exit_found',0) for f in families]
    fig,axes=plt.subplots(1,2,figsize=(11.8,4.6),layout='constrained')
    xx=np.arange(3);axes[0].bar(xx-.18,applicable,.35,label='Valid corrected entry',color='#668DB6')
    axes[0].bar(xx+.18,exits,.35,label='First exit within 64 phases',color='#359B86')
    axes[0].set_xticks(xx,names);axes[0].set_ylim(0,62);axes[0].set_ylabel('Starting states (131 per family)')
    axes[0].set_title('Certified first-exit proposals');axes[0].legend(frameon=False,fontsize=8)
    for j in range(3):
        axes[0].text(j-.18,applicable[j]+.8,str(applicable[j]),ha='center')
        axes[0].text(j+.18,exits[j]+.8,str(exits[j]),ha='center')
    for family,label,color,offset in [('alm_keep','Retained dual','#668DB6',-.18),('alm_reset','Reset dual','#DA9250',.18)]:
        hist=groups[family]['exit_phase_histogram'];ts=sorted(map(int,hist))
        axes[1].bar(np.array(ts)+offset,[hist[str(t)] for t in ts],width=.35,label=label,color=color)
    axes[1].set_xlabel('First changed update number minus one (input phase T)')
    axes[1].set_ylabel('Number of states');axes[1].set_title('Exact exit time, independently replayed')
    axes[1].legend(frameon=False,fontsize=8);axes[1].set_xticks([1,5,10,15,20,26])
    fig.suptitle('318 | From a dual-drift witness to a verified first-exit operator',fontsize=14)
    fig.savefig(out/'certified_first_exit.png',dpi=180);plt.close(fig)
    lines=['# 317–318：从漂移证书到可执行的精确首次退出算子','',
           '本轮实现了一个具体动作：联合校正内部状态，沿乘子漂移线直接定位参数或活动第一次改变的整数时刻，再执行真实完整更新。全部首次退出已独立逐步验证。它还没有产生任务收益：退出瞬间的前向区域仍不可行，因此下一步应检验退出后的完整适应过程。','',
           '![首次退出算子与精确步数](certified_first_exit.png)','',
           '## 1. 算子是什么','',
           '先从当前完整更新公式 F(s)=Ms+c 求 q/d，使 F(q+t*d)=q+(t+1)*d，且 d 的参数/活动分量为零。令 A=I-M，条件是 A*q+d=c、A*d=0。317 共393提议中384个代数可解；实际选择条件另外检查，不能把代数线直接当实际轨迹。','',
           '对于合法入口 q 和正确漂移 d_u=eta*r(q)，每个标量候选 z 相对当前值的能量差为 Delta(z)+t*beta(z)。其关于 t 的二次项相减消失。beta 在每个网络标量区间上仿射，因此端点就能判断是否存在负斜率。负斜率给出有限退出上界；没有负斜率则证明原始变量可永久停滞。','',
           '利用“仍选择当前块值”的相位集合是从0开始的区间，对整数 t 二分，找到首个退出输入相位 T。q 到 q+T*d 的前 T 次更新保留参数/活动，第 T+1 次完整更新才改变它们。中间可有无关策略标签变化，不要求完整策略签名一致。','',
           '**重要：q 是校正后的新提议，不是原状态。** 因此不能把这一动作说成从原始轨迹免费跳过计算；校正、线性求解、能量检查与状态成本都必须计入。有限退出思想继承246/307，本轮的新实现是与317联合状态校正连接，而不是重新命名旧证书。','',
           '## 2. 全部393状态的结果','',
           '旧64任务共有131触发位置；每位置三族，45个无触发任务仍保留。这里只使用支持观察，不使用查询答案。','',
           '| 方法 | 总状态 | 合法且实际保持原始变量的入口 | 64相位内首次退出 | 有限退出但超过64 | 证明永不退出 |','|---|---:|---:|---:|---:|---:|']
    for f,label in zip(families,cn):
        c=groups[f]['counts'];lines.append('| '+label+' | '+' | '.join(str(c.get(k,0)) for k in ['cases','applicable','first_primal_exit_found','finite_exit_beyond_cap','primal_stasis_for_all_nonnegative_phases'])+' |')
    lines+=['','保留乘子的48个退出来自3个旧任务；重置乘子的38个退出全部来自1个旧任务，不能当作86个独立任务。无乘子的42个入口 d=0，仍停在有残差的固定点。','',
            '## 3. 验证与任务含义','',
            '- 合成测试：4056个精确能量差恒等式；保留了不相容观察导致无限乘子漂移的反例；合法教师任务的有限退出见证在输入相位22退出，并逐相位独立验证。',
            '- 317：393个独立线性系统核验、1536个非当前点漂移线恒等式、7120标量能量块/17534区间核验。',
            '- 318：9347个独立端点能量证书、4754次完整逐整数更新，确认86个退出确实是第一次，未跳过更早变化。',
            '- 数学与浮点退出的前向模式没有差异；最大完整状态差分别3.33e-16和2.43e-16。',
            '', '退出原始变量不等于找到支持可行参数或新的有效前向区域。状态校正 q 与退出后参数的精确/浮点对照共444点，归入6个任务-模式对，全部有精确不可行证书；没有正体积有效区域，也没有满足观察带的参数点。独立重建了444次前向及点约束，复核18个区域证书。尚未访问查询答案或计算新任务风险。','',
            '## 4. 成本','',
            '| 方法 | 当前映射与漂移线求解核心总秒数 | 首次退出增量总秒数 | 浮点退出核验总秒数 |','|---|---:|---:|---:|']
    for f,label in zip(families,cn):
        s=groups[f]['seconds'];lines.append(f"| {label} | {s['prior_line_generation_core']:.6f} | {s['incremental_exact_escape']:.6f} | {s.get('floating_exit_validation',0):.6f} |")
    lines+=['','以上每行覆盖该族131状态，是精确算术开发实现的阶段计时，不含旧触发起点获取成本，也不是部署/匹配资源速度证明。虽然数学上可跳过若干保持原始变量的步，当前还不能宣称净加速。','',
            '## 5. 下一实验：退出后的完整适应','',
            '对同一批起点比较：原始状态继续64步；只校正到q再继续64步；直接定位首次退出后继续64步；按同一虚拟迭代总数匹配的q轨迹与跳跃轨迹。先封存最终参数和成本，再做共同区域检查。这样可分别识别状态校正、额外推进和数值路径差异的作用。若出现实际有效区域，再进入完整资源匹配和新冻结任务验证，不把这次局部退出直接称为论文级收益。','',
            '目标仍为：相同信息、先验和合理资源下，相对于强回归、浅层头及同参数强优化器的可复现独立查询收益。本报告只完成其中一个可验证的内部动作。','']
    (out/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    save(out/'summary.json',dict(passed=True,source_sha256=sha(Path(__file__)),input_audit_sha256=sha(base/'audit_v1/summary.json'),
                                input_geometry_audit_sha256=sha(base/'geometry_audit_v2/summary.json'),
                                files_sha256={n:sha(out/n) for n in ['report.md','certified_first_exit.png']}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False);run(Path(__file__).resolve().parents[2],args.out)
