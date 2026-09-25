"""OLD64 descriptive mechanism figure. No query quality or speedup claim."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from diagnose_gradient_flat_split_states_v1 import read, save, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    root = parser.parse_args().project.resolve()
    a = root/'results/state_matched_dual_intervention/development_v2'
    b = root/'results/counterfactual_branch_proposals/development_v1'
    aa, bb = read(a/'summary.json'), read(b/'summary.json')
    assert aa['passed'] and bb['passed'] and aa['tasks']==bb['tasks']==64
    for folder, ss in [(a, aa), (b, bb)]:
        for name, digest in ss['outputs_sha256'].items():
            assert sha(folder/name)==digest
    out = root/'results/state_matched_dual_intervention/figures_v1'
    out.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10,
                         'axes.spines.top':False, 'axes.spines.right':False})
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), layout='constrained')
    groups = ['All newly found\npositive branches', 'Not found by zero dual\nin the discovery round',
              'Not found in the entire\nstate-matched shadow union']
    sets = aa['mechanism_sets']
    all_keys = ['actual_new','absent_entire_round_zero','absent_zero_entire_shadow_union']
    bp_keys = ['actual_only_vs_bp','only_vs_bp_absent_entire_round_zero','only_vs_bp_absent_entire_shadow_union']
    x = np.arange(3)
    for offset, keys, label, color in [(-.2, all_keys, 'All ALM additions', '#387f86'),
                                      (.2, bp_keys, 'Subset also absent from original BP', '#b47739')]:
        values = [sets[k]['task_modes'] for k in keys]
        bars = axes[0].bar(x+offset, values, width=.36, color=color, label=label)
        axes[0].bar_label(bars, padding=3)
    axes[0].set(xticks=x, xticklabels=groups, ylim=(0,215), ylabel='Task-specific branch count',
                title='A. State-matched dual erasure\n69,632 original transitions reproduced bitwise')
    axes[0].legend(loc='upper right', fontsize=9)
    axes[0].grid(axis='y', alpha=.2)
    rules = ['all_steps','changed_forward_mode','first_global_forward_mode']
    names = ['Original ALM\nno added proposals', 'Shadow at every step\n1,088 proposals/task',
             'When forward mode changes\n271.28 proposals/task', 'First global forward mode\n37.92 proposals/task']
    values = [bb['old_alm_coverage']]+[bb['rules'][r]['mean_union_coverage'] for r in rules]
    bars = axes[1].barh(np.arange(4), np.array(values)*100, color=['#9daab0','#387f86','#b47739','#7c9690'])
    axes[1].bar_label(bars, labels=[f'{v*100:.4f}%' for v in values], padding=4)
    axes[1].set(yticks=np.arange(4), yticklabels=names, xlim=(0,104),
                xlabel='Mean numerical support-posterior coverage (%)',
                title='B. Chargeable proposal unions\nSame OLD64 tasks; no query scoring')
    axes[1].invert_yaxis()
    axes[1].grid(axis='x', alpha=.2)
    fig.suptitle('290–291 | OLD64 exploratory mechanism and candidate design', fontweight='bold')
    fig.supxlabel('Shadow steps depend on ALM history. Coverage is not query accuracy; proposal counts are not matched runtime. Numerical posterior reference is offline only.', fontsize=9)
    fig.savefig(out/'290_291_mechanism.png', dpi=170)
    plt.close(fig)
    save(out/'summary.json', dict(passed=True, source_sha256=sha(Path(__file__)),
         intervention_summary_sha256=sha(a/'summary.json'), proposal_summary_sha256=sha(b/'summary.json'),
         image_sha256=sha(out/'290_291_mechanism.png'), visual_qa_required=True,
         query_quality_evaluated=False, core_research_goal_complete=False))


if __name__ == '__main__':
    main()
