"""356 auditable common-pool report. No new tuning or query evaluation."""
from pathlib import Path
import numpy as np
from report_search_radius_development_v1 import read, sha, save, write, complete, table

CHANNELS=['dual','dual_plus_residual','residual','bp','random_sign','zero']
PREFIXES=[1,4,16,64,128]
BASE='results/region_conditioned_credit'
ENTRY='outputs/ttt-pc-alm-research/356_region_conditioned_credit_results_v1.md'


def main():
    root=Path(__file__).resolve().parents[2]; source=root/BASE/'development_v1'
    primary=complete(source); audit=complete(root/BASE/'audit_v1')
    assert audit['source_summary_sha256']==sha(source/'summary.json')
    protocol=read(source/'protocol.json')
    for path,digest in protocol['source_sha256'].items(): assert sha(root/path)==digest
    repair=protocol['serialization_repair']; assert sha(root/repair['adapter_file'])==repair['adapter_sha256']
    out=root/BASE/'report_v1'; out.mkdir(parents=True,exist_ok=False)
    after={c:[0]*5 for c in CHANNELS}; exclusive={c:0 for c in CHANNELS}; witnesses=[]
    for task in read(source/'tasks.json'):
        directory=source/str(task['seed']); inp=read(directory/'input.json')
        steps={}
        for c in CHANNELS:
            with np.load(directory/(c+'.npz'),allow_pickle=False) as z: steps[c]=z['first_step']
        with np.load(directory/'controls.npz',allow_pickle=False) as z: c20=z['c20']
        masks={c:s>0 for c,s in steps.items()}
        for c in CHANNELS:
            unique=masks[c]&~c20&~np.logical_or.reduce([masks[o] for o in CHANNELS if o!=c])
            exclusive[c]+=int(unique.sum())
            after[c]=[old+int(((steps[c]>0)&(steps[c]<=n)&~c20).sum()) for old,n in zip(after[c],PREFIXES)]
            for index in np.flatnonzero(unique):
                witnesses.append(dict(seed=task['seed'],channel=c,mode=inp['modes'][index],index=int(index)))
    totals=audit['totals']; rows=[]
    for c in CHANNELS:
        t=totals[c]
        rows.append([c,*[t['prefix_'+str(p)] for p in PREFIXES],t['beyond_c20'],t['response_pairs'],f"{t['seconds']:.6f}"])
    controls=[[c,totals[c]['rejected'],f"{totals[c]['seconds']:.6f}"] for c in ['c5','c20','pdhg60']]
    contrasts=[[c,*[audit['contrasts'][c][o] for o in CHANNELS],exclusive[c]] for c in CHANNELS]
    data=dict(prefixes=PREFIXES,channels=CHANNELS,
        coverage={c:[totals[c]['prefix_'+str(p)] for p in PREFIXES] for c in CHANNELS},
        beyond_c20_prefixes=after,component_seconds={c:totals[c]['seconds'] for c in totals},
        exclusive_after_c20_and_other_credits=exclusive)
    save(out/'figure_data.json',data);save(out/'table_data.json',dict(main=rows,controls=controls,contrasts=contrasts))
    save(out/'posthoc_witnesses.json',dict(descriptive_only=True,selection='C20 and all other five initializations fail at 128',witnesses=witnesses))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,5.9))
    colors=['#147d75','#659994','#9a723c','#456baf','#956bb0','#ce6b48']
    labels=['ALM dual','Dual + residual','Residual','BP','Random sign','Zero']
    for c,color,label in zip(CHANNELS,colors,labels):
        axes[0].plot(PREFIXES,data['coverage'][c],marker='o',color=color,label=label,linewidth=1.5)
        axes[1].plot(PREFIXES,after[c],marker='o',color=color,label=label,linewidth=1.5)
    axes[0].axhline(totals['c20']['rejected'],color='#333333',linestyle='--',label='C20 (9,548)')
    axes[0].set_title('Common pool: exact certified rejections',loc='left',pad=13)
    axes[1].set_title('Additional certificates beyond C20',loc='left',pad=13)
    axes[0].set_ylabel('Regions rejected out of 9,636');axes[1].set_ylabel('Regions missed by C20')
    for ax in axes:
        ax.set_xscale('log',base=2);ax.set_xticks(PREFIXES,[str(p) for p in PREFIXES]);ax.set_xlabel('Response checks (first check = initialization)')
        ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.18)
    axes[0].legend(frameon=False,fontsize=9,loc='lower right');axes[1].set_ylim(-1,28)
    fig.subplots_adjust(left=.085,right=.97,top=.86,bottom=.22,wspace=.3)
    fig.text(.085,.09,'32 exposed development tasks; all credits get the same regions and update. No query answers used.',fontsize=10)
    fig.text(.085,.035,'Component evidence only: initialization cost excluded. More certificates do not establish lower query risk.',fontsize=10)
    fig.savefig(out/'region_credit_coverage.png',dpi=170);plt.close(fig)
    report=['# 356｜区域条件化信用：理论保证、实际证书与初始化归因','',
        '本轮完成了可运行的自由信用更新与独立证书审计：主 ALM 乘子初始化在9,636个共同候选区域中严格排除9,372个，其中21个不被C20区间收缩排除。**这21个证实局部联合约束信用可以补充区间收缩；但零初始化也额外排除20个，主方法没有独立平均优势。**','',
        '这不是另起一个回避原目标的任务，而是在原分支搜索计算路径内检验：已有乘子是否能把不可行区域更快筛掉。固定方向扩为区域条件化方向后确实获得证书，但本次增益不能归因于ALM初始化。主比较保持dual，不按事后结果改为最优变体。','',
        '## 从网络约束到可检验的更新','',
        '```text\n观测 x/v + 候选分支 R\n  → 共享偏置/分支像集构成局部凸域 Y_R\n  → 当前信用 a 的解析最坏响应 g=r(y)\n  → D_R(a)>0? ──是──→ 整数有理数复核 → 安全排除该区域\n         │否\n         └→ 投影到半空间 {a: aᵀg≥1} → 下次响应（最多128次）\n```','',
        'D_R(a)=min_(y∈Y_R) aᵀr(y)。非空凸紧域上 max_(||a||₂≤1)D_R(a)=min_(y∈Y_R)||r(y)||₂，因此不可行域存在可缩放分离信用。这是标准凸对偶，不是独有的新定理。','',
        '固定δ=1，a_(t+1)=a_t+[(1−a_tᵀg_t)_+/||g_t||₂²]g_t。对任何满足D_R(a*)≥1的分离信用，理想精确算术下有：','',
        r'$$\|a_{t+1}-a^*\|_2^2\le\|a_t-a^*\|_2^2-\frac{(1-D_R(a_t))^2}{\|g_t\|_2^2}.$$','',
        '成功前D≤0且||g||²≤dn，给出依赖初始距离的有限步上界。这个上界也说明ALM需要证明什么：它应使到分离集合的距离或实际有限步轨迹更好。信用正确、无全局BP本身都不能推出这一点。浮点响应仅作提议；只有严格整数/Fraction正值能实际排除。','',
        '## 冻结共同池结果','',
        '六方法使用同一348触发状态、同一350提议并集、相同归一化和δ=1。均为已暴露开发任务；没有输入查询答案、几何解或后验参考。老提议文件含历史分类，但提取器只取模式及来源路径，不将分类传给候选。','',
        table(['初始化','第1次','第4次','第16次','第64次','第128次','C20外新增','响应对数','32任务合计秒'],rows),'',
        '![共同区域证书覆盖与C20之外的增量](region_credit_coverage.png)','',
        table(['现有对照','排除区域','32任务合计秒'],controls),'',
        '上述时间包含本组件解析响应和精确验证，但不含加载信用前的母轨迹、提议生成及最终预测。一次测量，不是重复统计或等墙钟预算比较。每种自由信用方法命名数组峰值小计850,896字节，不是进程峰值。旧PDHG60使用更宽活动松域，因此覆盖差不能全部归因更新方式。','',
        '## 收益究竟来自哪里','',
        '固定主dual初次检查仅排除31个，128次后9,372个，说明区域条件化更新真实改变了信用方向的作用；但零初始化从0增至9,374个，且响应对数更少。C20自身排除9,548个，主遗漏其中197个，同时补充21个，因此主与C20互补，但不能直接替代C20。','',
        '下表行A列B表示“A能排除而B不能”的区域数，不是误差改善。最后一列还要求C20与其它五信用全部不能排除，是事后描述性见证统计。','',
        table(['A / B',*CHANNELS,'C20及其它五信用之外独占'],contrasts),'',
        'dual比zero独有19个，但zero比dual独有21个；dual对C20与其它五信用的并集没有独占。不能把19个局部互补例子表述为“必须使用ALM”。本轮支持将联合约束证书作为可用组件，尚不支持ALM初始化是不可替代点。','',
        '## 独立核验与保留的失败记录','',
        f"原语320例解析响应/独立LP/Fraction一致，最大误差4.44e-15；40次受守卫求解、20次已知可行模式保留、100条投影距离检查通过。正式审计{audit['counts']['independent_fraction_proofs']:,}条独立Fraction证书、{audit['counts']['independent_pdhg_proofs']:,}条PDHG证书和{audit['counts']['geometry_certificates']:,}条原几何证书通过。所有排除都落在已被独立严格证明不可行的区域；4个正体积及2个零体积/未判空区域均保留。",'',
        '首次运行在第1任务保存PDHG NumPy数组时失败，development_attempt1保留原文件和异常；v2适配器仅把数组序列化为列表，未修改冻结数值源码，全部32任务重新运行。协议记录适配器和失败记录哈希。','',
        '## 与在线任务结果的衔接','',
        '本轮不产生新的查询MSE，也没有替换353已封存预测。当前最近在线主MSE为0.0521723419，仍高于同增强零信用0.0521210081和同起点Adam240的0.0512499136。尚无官方TTT-MLP/LLM/VLM验证。','',
        '后续如使用本组件，应先以C20处理明显冲突，再检验剩余区域的动态分离信用能否被复用来约束整个搜索空间，而不是继续对所有易排除区域重复求解。该方向需要核对旧跨模式信用与PDHG工作，并用同增强非乘子/强优化器检验完整收益；此处只列待检验假设，不冒充新实现或结果。','',
        '[冻结数学协议](../../../outputs/ttt-pc-alm-research/355_region_conditioned_credit_protocol_v1.md) · [完整组件审计](../audit_v1/summary.json) · [32任务逐项比较](../audit_v1/rows.json) · [最新在线图文](../../unvisited_online/report_v1/report.md)','']
    write(out/'report.md','\n'.join(report))
    write(root/ENTRY,'# 356｜区域条件化信用结果\n\n[完整图文、数学条件与全部对照](../../results/region_conditioned_credit/report_v1/report.md)\n\n主dual严格排除9,372个共同区域，补充C20遗漏21个；zero为9,374个、补充20个。联合约束组件有效，不等于ALM初始化独立收益。本轮没有新查询MSE。\n')
    save(out/'manifest.json',dict(passed=True,source_sha256=sha(Path(__file__)),entry_file=ENTRY,entry_sha256=sha(root/ENTRY),
        input_summary_sha256={BASE+'/audit_v1/summary.json':sha(root/BASE/'audit_v1/summary.json')},
        outputs_sha256={n:sha(out/n) for n in ['report.md','figure_data.json','table_data.json','posthoc_witnesses.json','region_credit_coverage.png']}))
    print(dict(passed=True,report=str(out/'report.md')),flush=True)


if __name__=='__main__':main()
