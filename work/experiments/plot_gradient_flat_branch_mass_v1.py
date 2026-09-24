"""Static scientific view of the complete old64 mass census, not query risk."""
from pathlib import Path
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from diagnose_gradient_flat_split_states_v1 import read, save, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    root = parser.parse_args().project.resolve()
    inp = root/'results/gradient_flat_split_states/branch_mass_v1'
    summary = read(inp/'summary.json')
    assert summary['passed'] and summary['tasks'] == 64
    for name, digest in summary['outputs_sha256'].items():
        assert sha(inp/name) == digest
    out = root/'results/gradient_flat_split_states/figures_v1'
    out.mkdir(parents=True, exist_ok=False)
    roles = ['archive', 'nodual', 'bp', 'alm']
    labels = ['Shared probe archive', 'No dual: 32 + anchor32', 'Adam: 240 rounds', 'ALM: 32 + anchor32']
    colors = ['#9aa1a9', '#638391', '#5470af', '#bd6b32']
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.8), layout='constrained')
    values = [100*summary['method_coverage'][r]['mean'] for r in roles]
    bars = ax[0].barh(labels, values, color=colors, height=.57)
    ax[0].bar_label(bars, labels=[f'{v:.2f}%' for v in values], padding=5)
    ax[0].set_xlim(0, 104)
    ax[0].set_xlabel('Mean covered posterior mass (%)')
    ax[0].set_title('A. All 64 old development contexts')
    ax[0].invert_yaxis()
    ax[0].grid(axis='x', alpha=.18)
    for role, label, color in zip(roles[1:], labels[1:], colors[1:]):
        y = 100*np.array(summary['mean_round_coverage'][role])
        # Common prefix only: all 33 origins advance each round in each method.
        ax[1].plot(np.arange(33), y[:33], label=label.split(':')[0], color=color, lw=2.3)
    ax[1].axhline(values[0], color=colors[0], linestyle='--', lw=1, label='Shared archive')
    ax[1].set(xlabel='Continuation rounds (33 origins per round)', ylabel='Mean covered posterior mass (%)',
              title='B. First 32 continuation rounds', ylim=(73, 90), xlim=(0, 32))
    ax[1].grid(alpha=.18)
    ax[1].legend(loc='lower right')
    fig.suptitle('Mechanism diagnostic: posterior coverage, NOT unseen-query accuracy', fontsize=15, fontweight='bold')
    fig.supxlabel('Old64 only; numerical volumes with audited support enumeration. Iteration counts are NOT equal wall-clock budgets.', fontsize=10)
    path = out/'289_branch_mass.png'
    fig.savefig(path, dpi=180)
    plt.close(fig)
    save(out/'summary.json', dict(passed=True, visual_review_required=True, tasks=64,
         source_sha256=sha(Path(__file__)), mass_summary_sha256=sha(inp/'summary.json'),
         outputs_sha256={path.name: sha(path)}, scope='Plot creation only; no new query evidence'))
    print(path)


if __name__ == '__main__':
    main()
