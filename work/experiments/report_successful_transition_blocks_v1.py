"""310 report of complete census and all factorial alternatives, no query claims."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from evaluate_complete_credit_mode_geometry_v1 import read,save,sha


def run(root,out):
    base=root/'results/successful_transition_census'
    folders=[base/n for n in ['development_v2','block_replay_v1','block_audit_v1']]
    census,replay,audit=[read(p/'summary.json') for p in folders]
    for folder,s in zip(folders,[census,replay,audit]):
        assert s['passed']
        for n,digest in s['outputs_sha256'].items():assert sha(folder/n)==digest
    events=read(folders[2]/'events.json')
    families=['alm_keep','alm_reset','nodual','pc']
    masks=[format(i,'03b') for i in range(8)]
    counts=[]
    for target_only in [True,False]:
        a=np.zeros((8,4),dtype=int)
        for c,f in enumerate(families):
            ee=[e for e in events if e['event']['family']==f]
            for e in ee:
                for r in e['alternatives']:
                    a[masks.index(r['mask']),c]+=r['target_reached'] if target_only else r['new_positive']
        counts.append(a)
    totals=np.array([sum(e['event']['family']==f for e in events) for f in families])
    fig,axes=plt.subplots(1,2,figsize=(13,6.6),sharey=True)
    labels=[' / '.join('current' if c=='1' else 'initial' for c in m) for m in masks]
    for ax,values,title in zip(axes,counts,['Return to the same certified target','Reach ANY positive mode outside original pool']):
        im=ax.imshow(values/totals[None],vmin=0,vmax=1,cmap='Blues',aspect='auto')
        ax.set_xticks(range(4),['ALM keep','ALM reset','No dual','PC'],fontsize=10)
        ax.set_title(title,fontsize=11,pad=13)
        ax.set_yticks(range(8),labels,fontsize=9)
        for i in range(8):
            for j in range(4):
                ax.text(j,i,f'{values[i,j]}/{totals[j]}',ha='center',va='center',
                        color='white' if values[i,j]/totals[j]>.6 else 'black',fontsize=11)
    axes[0].set_ylabel('Parameter b / activity h / multiplier u',fontsize=10)
    fig.suptitle('310: factorial state interventions at every first successful local transition',fontsize=14,y=.975)
    fig.subplots_adjust(left=.24,right=.91,bottom=.17,top=.87,wspace=.16)
    ca=fig.add_axes([.935,.17,.015,.70]);fig.colorbar(im,cax=ca,label='Fraction of selected events')
    fig.text(.24,.055,'26 events from old development trajectories; repeated origins are NOT independent tasks.\nFuture states are diagnostic interventions, not free online inputs. No query-risk gain established.',fontsize=9)
    fig.savefig(out/'state_interactions.png',dpi=160);plt.close(fig)
    unique=[e for e in events if e['event']['family']=='alm_keep' and e['successful_new_positive_masks']==['111']]
    unique_tasks={e['event']['seed'] for e in unique}
    unique_pairs={(e['event']['seed'],e['event']['mode']) for e in unique}
    lines=['# 310：真实成功转移依赖的是完整状态协同，而非单独放大乘子', '',
        f'本轮取得了一个明确的因果定位：保留乘子的11个首次成功事件中，{len(unique)}个在全部8种初始/当前状态组合里，只有完整当前(b,h,u)能提出原池外的正体积模式。这{len(unique)}个事件合并后是{len(unique_pairs)}个任务—模式，来自{len(unique_tasks)}个旧任务，不能当成9个独立任务。', '',
        '这解释了为何同一起点的幅度搜索和层开关不足以复现多步结果：成功前的参数位置、内部活动及乘子经历了共同演化。它还不是新的可部署算法，更没有证明相对强BP或回归的任务优势。', '',
        '## 首达普查：全部七族，59,605个访问状态', '',
        '|方法|任务|新(task,mode)对|首次到达最早—最晚步|步0|步1|首次参数已拟合观察带|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for n,r in census['aggregate'].items():
        steps=[int(s) for s in r['first_step_histogram']]
        lines.append(f"|{n}|{r['tasks']}|{r['task_mode_pairs']}|{min(steps)}–{max(steps)}|{r['step_zero_pairs']}|{r['step_one_pairs']}|{r['first_parameters_in_support_band_pairs']}|")
    lines+=['', '64任务、131相同起点、917条完整轨迹全部保留，917项模式池与308完全一致。有效模式表示其中存在满足支持约束的参数；首次进入该模式的当前参数本身并未拟合到观察带，不能偷换成已经学习成功。', '',
        '## 完整状态反事实', '',
        '对每个局部方法的全部首次成功事件，保存成功更新前当前状态S_t=(b_t,h_t,u_t)，与原起点S_0的对应块做2³种组合，再执行同一次原更新。所有26事件均保留，不按查询筛选。图中分母是事件数，不是任务数。', '',
        '![三块状态替换的完整因果矩阵](state_interactions.png)', '',
        '只判定“是否回到原目标模式”会漏掉其他有效解释。因此右图又统一检查全部208种实际输出：67个有严格内部证书、141个有精确不可行证书；包含原池模式，67不是新增模式数。7个干预输出虽未回到原目标，却提出另一个原池外有效模式，全部保留。', '',
        '|家族|事件数|仅完整当前三块能找回任一新有效模式|旧参数也可能产生另一新有效模式的事件|',
        '|---|---:|---:|---:|']
    for n,r in audit['aggregate'].items():
        lines.append(f"|{n}|{r['events']}|{r['all_current_only_events']}|{r['old_b_any_new_positive_events']}|")
    lines+=['', '特别注意：某些事件的多个最小充分块组合互不包含；增加一个当前块也可能破坏原有成功。这是离散、非单调的8点反事实结果，不是“新状态总优于旧状态”的定理。', '',
        '## 可核验的结论与不能推出的结论', '',
        '令T是同一个完整局部更新，P_new是原池外有正体积支持可行性的模式集合。对上述9个事件，实测且逐项认证：mode(T(S_111))∈P_new，而其余7种S_abc都不在P_new。该断言只覆盖明确定义的8个状态组合；不排除其他活动、乘子、优化器、闭式算法或更简单方法。', '',
        '因此后续候选应研究可观察的状态协调或局部更新次序：怎样在正常在线执行中更早形成这样的联合状态，并与相同操作的无乘子、PC和强优化器比较。不能把未来的S_t免费交给方法，也不能仅凭这9个事后事件挑任务或声称“必须使用PC-ALM”。', '',
        '## 复核与范围', '',
        f"全部33,536个局部参数状态、76个首步活动数组、152个末步活动/乘子数组逐位重放通过。208个干预的52个端点对照通过；非PC干预用独立Fraction完整步核验，最大差{replay['max_exact_reference_gap']:.12g}，模式差异0。PC没有冒充通过该精确ALM参考。",
        '另以单起点重放复核624个b/h/u数组，并检查677项精确几何证书。候选重放期间全局BP和全局优化器入口被禁用；几何LP只在独立后处理评价中执行。',
        '310首版普查误读308清单格式，失败源与记录保留；v2验证正确的before_geometry_manifest及所有文件哈希后运行。查询答案与后验均值均未读取。诊断时间不是在线匹配资源结果，全部研究目标保持active。', '']
    (out/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    save(out/'summary.json',dict(passed=True,
        input_summaries_sha256={p.name:sha(p/'summary.json') for p in folders},
        all_current_only_keep_events=len(unique),distinct_tasks=len(unique_tasks),distinct_task_modes=len(unique_pairs),
        counts_target=counts[0].tolist(),counts_any_new=counts[1].tolist(),event_totals=totals.tolist(),
        source_sha256=sha(Path(__file__)),outputs_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file()},
        visual_review_required=True,independent_task_gain_established=False))
    print(dict(passed=True,report=str(out/'report.md'),all_current_only_keep_events=len(unique)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False);run(Path(__file__).resolve().parents[2],args.out)
