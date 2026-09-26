"""Phase-retrieval figures for report 444 (results/lifted_credit/pr*_*/rows.json)."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2] / 'results' / 'lifted_credit'
FIG = ROOT / 'figures'; FIG.mkdir(exist_ok=True)
PAL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID,
                     'figure.dpi': 150, 'legend.frameon': False})
SERIES = [('PC-ALM, random init, 1 start', ['rand1_PCALM']),
          ('same-rho PC without multipliers, 1 start', ['rand1_PCnomult_rhosel']),
          ('spectral(orth) + TAF', ['spectral_orth+TAF']),
          ('spectral(orth) + RAF', ['spectral_orth+RAF']),
          ('spectral(orth) + AltMin / GS', ['spectral_orth+ALTMIN']),
          ('random init + Wirtinger flow, 16 starts', ['rand16_WF']),
          ('GAMP (Gaussian operators only), 16 starts', ['rand16_GAMP']),
          ('closed-form quadratic kernel head', ['closedform_quadratic_krr'])]


def load(*runs):
    rows = []
    for r in runs:
        p = ROOT / r / 'rows.json'
        if p.exists():
            rows += json.loads(p.read_text())['rows']
    return rows


def panel(ax, rows, d, title, xlabel):
    rows = [r for r in rows if r['d'] == d]
    for i, (lab, keys) in enumerate(SERIES):
        pts = {}
        for r in rows:
            if r['method'] in keys:
                pts[r['alpha']] = r['success']          # later runs (v2) overwrite v1 at the same alpha
        if not pts:
            continue
        xs = sorted(pts)
        ax.plot(xs, [pts[x] for x in xs], color=PAL[i], lw=2.6 if i == 0 else 1.6, marker='o', ms=4, label=lab)
    ax.set_ylim(-0.03, 1.03); ax.set_title(title, loc='left', fontsize=10, color=INK); ax.set_xlabel(xlabel)
    if 'masks' in xlabel:
        ax.set_xticks(sorted({r['alpha'] for r in rows})); ax.set_xticklabels([f"{int(a)}" for a in sorted({r['alpha'] for r in rows})])


def main():
    g256 = load('pr_gauss_d256', 'pr2_gauss_d256'); g1024 = load('pr_gauss_d1024', 'pr2_gauss_d1024')
    c = load('pr_cdp', 'pr2_cdp'); t64 = load('pr2_cdp_tiny64')
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.9), sharey=True)
    panel(axes[0], g256, 256, 'Gaussian, CIFAR-10 16x16 (d=256)', 'measurements per pixel  n/d')
    panel(axes[1], g1024, 1024, 'Gaussian, CIFAR-10 32x32 (d=1024)', 'measurements per pixel  n/d')
    panel(axes[2], c, 1024, 'Coded diffraction, CIFAR-10 32x32', 'masks L  (n = L d)')
    panel(axes[3], t64, 4096, 'Coded diffraction, Tiny-ImageNet 64x64', 'masks L  (n = L d)')
    axes[0].set_ylabel('images recovered (rel. error < 0.05)')
    hs, ls = [], []
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in ls:
                hs.append(h); ls.append(l)
    fig.legend(hs, ls, loc='lower center', ncol=4, fontsize=8.3, bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(FIG / 'fig6_phase_retrieval.png', bbox_inches='tight'); plt.close(fig)
    small = load('pr_gauss_krr_small')
    if small:
        fig, ax = plt.subplots(figsize=(6, 3.6))
        panel(ax, small, 16, 'Gaussian, 4x4 images (d=16): closed-form threshold', 'n/d')
        ax.axvline(136 / 16, color=INK2, ls=':', lw=1); ax.text(136 / 16 + .2, .05, 'n = d(d+1)/2', color=INK2, fontsize=8)
        ax.set_ylabel('recovered / predicted'); ax.legend(fontsize=7.5, loc='center right')
        fig.tight_layout(); fig.savefig(FIG / 'fig7_closed_form_threshold.png', bbox_inches='tight'); plt.close(fig)
    print('figures ->', FIG)


if __name__ == '__main__':
    main()
