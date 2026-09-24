"""298 full report; retain all46 configurations and both grids."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from diagnose_counterfactual_conditional_risk_v1 import read,save,sha


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project',type=Path,required=True)
    root=ap.parse_args().project.resolve()
    inp=root/'results/dual_amplitude_paths/development_v1'
    out=root/'results/dual_amplitude_paths/report_v1'
    out.mkdir(parents=True,exist_ok=False)
    summary=read(inp/'summary.json')
    assert summary['passed']
    for name,digest in summary['outputs_sha256'].items(): assert sha(inp/name)==digest,name
    rows=read(inp/'aggregates.json')
    lookup={(r['method'],r['grid']):r for r in rows}
    first_nonzero=0
    for r in read(inp/'proposals.json'):
        with np.load(inp/r['file'],allow_pickle=False) as z:
            first_nonzero+=int(np.any(z['dual_direction'][:,0]))
    alphas=read(inp/'protocol.json')['alphas']
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    for family,label in [('dual','Accumulated dual'),('residual','Norm-matched residual'),('random_sign','Random signs of dual')]:
        axes[0].plot(alphas,[lookup[f'{family}_a{i}',257]['delta']*1e6 for i in range(7)],'.-',label=label)
    axes[0].axhline(0,color='black',linewidth=.7)
    axes[0].set_xlabel('Input-credit amplitude alpha')
    axes[0].set_ylabel('Ideal pool-risk change vs original ALM (x 0.000001)')
    axes[0].set_title('All fixed amplitudes: no monotone risk improvement')
    axes[0].legend(fontsize=8)
    names=['dual_union','residual_union','random_sign_union','alm_reset_1','adam_4','adam_8']
    labels=['Dual union','Residual union','Random-sign union','Single reset','Adam4','Adam8']
    values=[lookup[n,257]['policy_excess']*1e3 for n in names]
    axes[1].barh(labels,values,color=['#2563eb','#f59e0b','#10b981','#64748b','#a855f7','#7c3aed'])
    axes[1].invert_yaxis()
    axes[1].set_xlabel('Ideal conditional excess risk (x 0.001)')
    axes[1].set_xlim(0,.0014*1e3)
    axes[1].set_title('Strong controls retained; not matched full-call timing')
    fig.suptitle('OLD64 support-only proposals | cached posterior diagnosis | 257-point grid',fontsize=12)
    fig.savefig(out/'amplitude_paths.png',dpi=170)
    plt.close(fig)
    ms=summary['math']
    text=['# 298｜乘子幅度路径：数学核验与全量开发结果','',
      '64个旧开发任务，24个新幅度/并集配置，加上20个短续接对照、原ALM和mode-change池，共46配置。所有支持侧提议封存后才读取离线后验参照；没有读取真实教师或查询答案。',
      '', '![乘子幅度族与强对照](amplitude_paths.png)', '',
      '## 结论及下一步', '',
      '单分支区间内的仿射局部更新结构得到核验，但本轮原乘子方向没有显示独立的搜索优势。三个七幅度并集的理想条件超额风险：dual .00123007003、等范数残差 .00123380532、随机符号 .00120842002；Adam4/8为 .00107828275/.00104292517。它们不是等完整时间比较，不能据此宣称速度胜负，也不能只挑dual的某个幅度宣称成功。',
      '', '幅度缩放与最终查询性能之间没有单调性：覆盖质量提高仍可能增大函数均值偏差。接下来检查触发是否遗漏原参数/模式停滞而约束仍不满足的阶段，而不是增加粒子数或继续盲目加长续接。',
      '', '## 数学核验的真实覆盖', '',
      f'每任务固定取首个触发状态，共64状态，其中{first_nonzero}个具有非零乘子。α中心0/.5/1的192个±1e-7窗口内部选择签名均未改变；公式斜率的最大仿射残差为{ms["max_affine_gap"]:.4g}。零乘子窗口是平凡恒定情形，不能将全部192项都称作非平凡信用传播验证。',
      '', f'标量独立枚举未复用生产bias_solve累积系数，b/h最大差{ms["max_independent_scalar_gap"]:.4g}。宽区间[-1,0,1]有{ms["wide_nonaffine_cases"]}个跨内部选择的非仿射示例，最大中点差{ms["max_wide_midpoint_gap"]:.8g}；这是局部公式不能跨分支外推的具体检查。',
      '', f'2427触发状态中{ms["zero_multiplier_states"]}个u=0，r=0状态{ms["zero_residual_states"]}个。归一化误差{ms["max_normalization_gap"]:.4g}。所有256组α=0/1端点逐位恒等式通过；旧297的20方法风险重放最大差为0。',
      '', '数学适用范围：当前固定分段线性激活的偏置模型、相同局部获胜分支和截断状态；有限分段仿射允许切换和跳跃。不代表全局光滑、收敛、查询改善或一般GELU MLP。',
      '', '## 完整结果：46方法×2网格', '',
      'a0…a6依次表示α=-1,-.5,0,.5,1,1.5,2；union为七个幅度的全部提议并集。两对批次差不是任务采样置信区间。状态步不等于完整墙钟资源。',
      '', '| 方法 | 网格 | 理想超额风险 | 相对原池差 | 对(0,1)差 | 对(2,3)差 | 覆盖 | 状态步/任务 | 新正区域/任务数 |',
      '|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for r in rows:
        text.append(f'| {r["method"]} | {r["grid"]} | {r["policy_excess"]:.12g} | {r["delta"]:.12g} | {r["pair_delta_means"][0]:.12g} | {r["pair_delta_means"][1]:.12g} | {r["mass"]:.10g} | {r["shadow_steps"]:.5f} | {r["new_positive_modes"]} / {r["tasks_with_new_positive_modes"]} |')
    text+=['',f'摘要SHA256：`{sha(inp/"summary.json")}`。逐任务数据、支持侧提议、全部内部状态自检、先封存后评分证据均在development_v1。总目标仍未完成；没有修改294新任务结果或旧8192强BP结论。']
    with (out/'report.md').open('x',encoding='utf-8') as stream:
        stream.write('\n'.join(text)+'\n')
    save(out/'summary.json',dict(passed=True,result_summary_sha256=sha(inp/'summary.json'),
        source_sha256=sha(Path(__file__)),math_first_states_nonzero_multiplier=first_nonzero,
        visual_review_pending=True,outputs_sha256={n:sha(out/n) for n in ['report.md','amplitude_paths.png']}))
    print(dict(report=str(out/'report.md'),math_first_states_nonzero_multiplier=first_nonzero),flush=True)


if __name__=='__main__':
    main()
