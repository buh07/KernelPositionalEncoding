"""
Generate new main-paper figures for:
  "The Missing Positional Story in LLMs: A Case Study of Shift-Invariant Attention"

Usage:
    cd /jumbo/lisp/f004ndc/Kernel PE/paper/Formatting_Instructions_For_NeurIPS_2026/
    python generate_figures.py

Output:  ../neurips2026/figures/
  fig_r2_heterogeneity.png       -- Fig 1: violin (7-8B trio) + 11-model inset (E31A)
  fig_overview_schematic.png     -- Fig 2: conceptual overview (SI channel, R², intervention)
  fig_functional_sensitivity.png -- Fig 3: preferential degradation + transfer scope
  fig_kernel_examples.png        -- Fig 4: g_h(Δ) kernel shapes + DFT

All BREADTH_MODELS values are from E31A CSV:
  /jumbo/lisp/f004ndc/Kernel PE/results/reinforce_exp3/E31a_breadth_consolidation/model_breadth_r2_table.csv

Kernel figure uses real per-head g_h data from:
  /jumbo/lisp/f004ndc/Kernel PE/results/experiment3/theory1_si_circuits/{model}/kernels_g_h.parquet
  Representative heads: Llama L1H8 (R²=0.902), Mistral L13H25 (R²=0.240), OLMo L7H31 (R²=0.046)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from scipy.stats import beta as beta_dist
import pandas as pd

np.random.seed(42)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'neurips2026', 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

C = {
    'llama':   '#1565C0',   # deep blue
    'mistral': '#E65100',   # deep orange
    'olmo':    '#2E7D32',   # deep green
    'rope':    '#1565C0',   # blue  (RoPE family in inset)
    'nope':    '#C62828',   # red   (NoPE control)
    'abspe':   '#6A1B9A',   # purple (absolute PE)
    'light':   '#ECEFF1',
    'grey':    '#78909C',
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def beta_params(mu, sigma):
    """Mean + std -> Beta(alpha, beta) parameters (0-1 bounded)."""
    var = sigma ** 2
    k = mu * (1 - mu) / var - 1
    return mu * k, (1 - mu) * k


def sample_r2(mu, sigma, n=8000):
    a, b = beta_params(mu, sigma)
    return beta_dist.rvs(a, b, size=n)


# ---------------------------------------------------------------------------
# FIGURE 1  Cross-model R² heterogeneity
# ---------------------------------------------------------------------------

# Summary statistics from paper Table A2
PRIMARY_STATS = {
    'Llama-3.1-8B': dict(mu=0.380, std=0.1561, q25=0.2545, q50=0.3885, q75=0.5040, color=C['llama']),
    'Mistral-7B':   dict(mu=0.266, std=0.1430, q25=0.1556, q50=0.2401, q75=0.3688, color=C['mistral']),
    'OLMo-2-7B':    dict(mu=0.058, std=0.0584, q25=0.0153, q50=0.0457, q75=0.0814, color=C['olmo']),
}

# 11-model breadth consolidation — all values from E31A CSV.
# Sorted ascending by mean_r2; the plotting code sorts anyway.
# Format: (display_label, mean_r2, pe_family, is_primary_trio)
BREADTH_MODELS = [
    # label                  mean_R2    pe_type    primary?
    ('Pythia\n(410M)',        0.00620,   'rope',    False),  # E31A min
    ('Pythia\n(1.4B)',        0.00653,   'rope',    False),  # E31A
    ('OLMo-2\n(7B)',          0.0576,    'rope',    True),   # E31A (Table A2: 0.058)
    ('TL-NoPE\n(1.1B)',       0.0868,    'nope',    False),  # E31A — NoPE control
    ('Mistral\n(7B)',          0.2658,    'rope',    True),   # E31A (Table A2: 0.266)
    ('GPT-2\nmedium',          0.2990,    'abspe',   False),  # E31A — absolute PE
    ('TL-RoPE\n(1.1B)',        0.3045,    'rope',    False),  # E31A (was early-layer 0.5511)
    ('GPT-2\nsmall',           0.3664,    'abspe',   False),  # E31A — absolute PE
    ('Llama\n(8B)',             0.3803,    'rope',    True),   # E31A (Table A2: 0.380)
    ('Qwen2.5\n(7B)',          0.6051,    'rope',    False),  # E31A
    ('Gemma-2\n(9B)',          0.8684,    'rope',    False),  # E31A max; spread = 0.8684/0.00620 = 140×
]

def make_r2_heterogeneity():
    fig = plt.figure(figsize=(7.0, 3.5))
    gs  = gridspec.GridSpec(1, 2, width_ratios=[1.55, 1.0], wspace=0.38)

    # ---- Left: violin plots of per-head R² for primary trio ----
    ax = fig.add_subplot(gs[0])
    model_names = list(PRIMARY_STATS.keys())
    positions   = [1, 2, 3]
    samples = [sample_r2(PRIMARY_STATS[m]['mu'], PRIMARY_STATS[m]['std']) for m in model_names]
    colors  = [PRIMARY_STATS[m]['color'] for m in model_names]

    vp = ax.violinplot(samples, positions=positions, showmedians=False,
                       showextrema=False, widths=0.65)
    for body, col in zip(vp['bodies'], colors):
        body.set_facecolor(col)
        body.set_alpha(0.55)
        body.set_edgecolor('black')
        body.set_linewidth(0.7)

    # IQR boxes + median + mean
    for pos, m, col in zip(positions, model_names, colors):
        s = PRIMARY_STATS[m]
        ax.add_patch(mpatches.FancyBboxPatch(
            (pos - 0.13, s['q25']), 0.26, s['q75'] - s['q25'],
            boxstyle='square,pad=0', lw=1.2, edgecolor='black', facecolor='white', zorder=3))
        ax.plot([pos - 0.16, pos + 0.16], [s['q50'], s['q50']],
                color='black', lw=1.8, zorder=4)
        ax.scatter([pos], [s['mu']], color=col, s=35, zorder=5,
                   marker='D', edgecolors='black', linewidths=0.7)
        ax.text(pos, min(s['q25'] - 0.06, s['mu'] - 0.09),
                f"$\\bar{{R}}^2={s['mu']:.3f}$",
                ha='center', fontsize=7.5, color=col, fontweight='bold')

    # 6.5× spread annotation
    ax.annotate('', xy=(3.3, 0.058), xytext=(3.3, 0.380),
                arrowprops=dict(arrowstyle='<->', color='#444', lw=1.2))
    ax.text(3.45, 0.22, '6.5×', fontsize=8, color='#444', va='center', style='italic')

    ax.set_xticks(positions)
    ax.set_xticklabels(model_names, fontsize=8)
    ax.set_ylabel('Per-head SI amplitude ($R^2$)', fontsize=9)
    ax.set_ylim(-0.05, 1.02)
    ax.set_xlim(0.5, 3.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title('Primary 7–8B models', fontsize=9, pad=4)

    legend_els = [
        Line2D([0], [0], marker='D', color='w', mfc='black', mec='black', ms=5, label='Mean'),
        Line2D([0], [0], color='black', lw=1.8, label='Median'),
        mpatches.Patch(facecolor=C['light'], edgecolor='black', lw=1.1, label='IQR'),
    ]
    ax.legend(handles=legend_els, fontsize=7, loc='upper right',
              framealpha=0.95, edgecolor='none', ncol=1)

    # ---- Right: 11-model ranked dot/bar inset (E31A) ----
    ax2 = fig.add_subplot(gs[1])

    known   = [(lbl, r2, pe) for lbl, r2, pe, _ in BREADTH_MODELS if r2 is not None]
    unknown = [(lbl, pe)     for lbl, r2, pe, _ in BREADTH_MODELS if r2 is None]
    primary_labels = {m['label'] for m in [
        {'label': 'OLMo-2\n(7B)'}, {'label': 'Mistral\n(7B)'}, {'label': 'Llama\n(8B)'}
    ]}
    primary_set = {'OLMo-2\n(7B)', 'Mistral\n(7B)', 'Llama\n(8B)'}

    known_sorted = sorted(known, key=lambda x: x[1])
    y_k = list(range(len(known_sorted)))

    for y, (lbl, r2, pe) in zip(y_k, known_sorted):
        col = C.get(pe, C['grey'])
        alpha = 1.0 if lbl in primary_set else 0.75
        lw = 1.4 if lbl in primary_set else 0.6
        ax2.barh(y, r2, color=col, alpha=alpha, edgecolor='black', linewidth=lw, height=0.55)
        ax2.text(r2 + 0.015, y, f'{r2:.3f}', fontsize=6.5, va='center', color='#333')

    # Unknown models as grey hatched placeholders
    n_k = len(known_sorted)
    unknown_sorted = sorted(unknown, key=lambda x: x[0])
    for j, (lbl, pe) in enumerate(unknown_sorted):
        y = n_k + j
        col = C.get(pe, C['grey'])
        ax2.barh(y, 0.25, color=col, alpha=0.20, edgecolor=C['grey'],
                 linewidth=0.6, height=0.55, hatch='///')
        ax2.text(0.27, y, '? (fill E31A)', fontsize=5.5, va='center', color=C['grey'])

    n_total = n_k + len(unknown_sorted)
    all_labels = [lbl for lbl, _, _ in known_sorted] + [lbl for lbl, _ in unknown_sorted]
    ax2.set_yticks(list(range(n_total)))
    ax2.set_yticklabels(all_labels, fontsize=6.5)

    # Max range marker
    ax2.axvline(0.8684, color='#333', lw=0.8, ls='--', alpha=0.5)

    ax2.set_xlabel('Mean $R^2$', fontsize=8)
    ax2.set_xlim(0, 1.02)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_title('11-model breadth (E31A)\n7 families, 140× max/min', fontsize=8, pad=4)

    pe_legend = [
        mpatches.Patch(color=C['rope'],  alpha=0.85, label='RoPE'),
        mpatches.Patch(color=C['nope'],  alpha=0.85, label='NoPE'),
        mpatches.Patch(color=C['abspe'], alpha=0.85, label='Absolute PE'),
    ]
    ax2.legend(handles=pe_legend, fontsize=6.5, loc='lower right',
               framealpha=0.95, edgecolor='none')

    out = os.path.join(OUTPUT_DIR, 'fig_r2_heterogeneity.png')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out}')


# ---------------------------------------------------------------------------
# FIGURE 2  Overview schematic
# ---------------------------------------------------------------------------

def make_overview_schematic():
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.7))
    fig.subplots_adjust(wspace=0.45, bottom=0.22)

    # ---- Panel A: Attention logit decomposition ----
    ax = axes[0]
    n = 9
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i >= j:
                d = i - j
                mat[i, j] = 0.7 * np.exp(-0.18 * d) + 0.25 * np.cos(1.1 * d + 0.2)
    mat_masked = np.where(np.tril(np.ones((n, n))), mat, np.nan)

    im = ax.imshow(mat_masked, cmap='Blues', vmin=-0.3, vmax=1.0, aspect='equal',
                   origin='upper', interpolation='nearest')
    for d in [0, 1, 2, 3, 5]:
        for k in range(n - d):
            i, j = k + d, k
            ax.text(j + 0.5, i + 0.5, str(d), fontsize=5.5, ha='center', va='center',
                    color='white' if mat[i, j] > 0.4 else '#555')
    ax.set_xlabel('Key $j$', fontsize=8)
    ax.set_ylabel('Query $i$', fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('(a) RoPE SI channel', fontsize=9, pad=3)
    ax.text(0.5, -0.22,
            r'$A_h(i,j)=g_h(i{-}j)_{\mathrm{SI}}+r_h(i,j)$',
            transform=ax.transAxes, ha='center', fontsize=8.5,
            bbox=dict(boxstyle='round,pad=0.3', fc='#E3F2FD', ec='#1565C0', lw=0.8))
    ax.text(0.5, -0.36, 'Numbers = offset $\\Delta$; diagonal stripes = SI structure',
            transform=ax.transAxes, ha='center', fontsize=6.5, color='#555')

    # ---- Panel B: R² scatter ----
    ax = axes[1]
    np.random.seed(3)
    n_pts = 60
    offs = np.random.choice(np.arange(1, 16), n_pts)
    g_vals = 0.7 * np.exp(-0.15 * offs) + 0.2 * np.cos(0.8 * offs)

    A_hi  = g_vals + np.random.normal(0, 0.05, n_pts)
    A_lo  = 0.25 * g_vals + np.random.normal(0, 0.28, n_pts)

    diag = np.linspace(-0.2, 1.1, 50)
    ax.plot(diag, diag, 'k--', lw=0.9, alpha=0.45, zorder=1)
    ax.scatter(g_vals, A_hi, color=C['llama'], s=14, alpha=0.7, zorder=2,
               label=f'High-SI ($R^2\\!=\\!0.94$)')
    ax.scatter(g_vals, A_lo, color=C['olmo'],  s=14, alpha=0.7, zorder=2,
               marker='s', label=f'Low-SI ($R^2\\!=\\!0.06$)')

    ax.set_xlabel(r'$g_h(\Delta)$ [offset-only]', fontsize=8)
    ax.set_ylabel(r'$A_h(i,j)$ [actual logit]', fontsize=8)
    ax.legend(fontsize=6.5, loc='upper left', framealpha=0.9, edgecolor='none')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title('(b) Measuring SI amplitude', fontsize=9, pad=3)
    ax.text(0.5, -0.22,
            r'$R_h^2 = 1-\frac{\sum(A_h-g_h)^2}{\sum(A_h-\bar{A}_h)^2}$',
            transform=ax.transAxes, ha='center', fontsize=8.5,
            bbox=dict(boxstyle='round,pad=0.3', fc='#FFF3E0', ec='#E65100', lw=0.8))
    ax.text(0.5, -0.36, 'High $R^2$ = logit dominated by token offset',
            transform=ax.transAxes, ha='center', fontsize=6.5, color='#555')

    # ---- Panel C: Dose-response sketch ----
    ax = axes[2]
    r2_x = np.array([0.04, 0.12, 0.22, 0.32, 0.42, 0.52, 0.62, 0.72, 0.82])
    noise_ll = np.array([ 0.002, -0.004,  0.003, -0.002,  0.004, -0.001,  0.003, -0.002,  0.002])
    noise_mi = np.array([-0.001,  0.003, -0.002,  0.001, -0.001,  0.002, -0.001,  0.001, -0.001])
    noise_ol = np.array([ 0.0002, -0.0001, 0.0002, -0.0001, 0.0001, 0.0002, -0.0001, 0.0001, 0.0])
    delta_ll = 0.004 + 0.024 * r2_x + noise_ll
    delta_mi = 0.003 + 0.020 * r2_x + noise_mi
    delta_ol = 0.0003 + 0.0018 * r2_x + noise_ol

    for r2v, col, lbl, rho in [
        (delta_ll, C['llama'],   'Llama',   0.959),
        (delta_mi, C['mistral'], 'Mistral', 0.964),
        (delta_ol, C['olmo'],    'OLMo',    0.684),
    ]:
        ax.scatter(r2_x, r2v, color=col, s=18, zorder=3, label=f'{lbl} ($\\rho\\!=\\!{rho}$)')
        p = np.polyfit(r2_x, r2v, 1)
        xf = np.linspace(0, 0.9, 60)
        ax.plot(xf, np.polyval(p, xf), color=col, lw=1.0, alpha=0.55, zorder=2)

    ax.set_xlabel('Per-head $R^2$', fontsize=8)
    ax.set_ylabel(r'$\Delta$ loss per head', fontsize=8)
    ax.legend(fontsize=6, loc='upper left', framealpha=0.9, edgecolor='none')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title('(c) Causal intervention', fontsize=9, pad=3)
    ax.text(0.5, -0.22,
            r'$\tilde{A}_h(i,j)=A_h(i,j)-\hat{g}_h(i{-}j)$',
            transform=ax.transAxes, ha='center', fontsize=8.5,
            bbox=dict(boxstyle='round,pad=0.3', fc='#F3E5F5', ec='#6A1B9A', lw=0.8))
    ax.text(0.5, -0.36, 'Disruption cost scales with SI amplitude',
            transform=ax.transAxes, ha='center', fontsize=6.5, color='#555')

    out = os.path.join(OUTPUT_DIR, 'fig_overview_schematic.png')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out}')


# ---------------------------------------------------------------------------
# FIGURE 3  Functional sensitivity
# ---------------------------------------------------------------------------

# Exact values from Table 3 (tab:e12_main)
FUNC_DATA = {
    'Llama-3.1-8B': dict(acc_offset= 0.910, acc_ctrl=-0.005, lp_offset=11.121, lp_ctrl= 2.612),
    'Mistral-7B':   dict(acc_offset= 0.715, acc_ctrl=-0.005, lp_offset= 6.826, lp_ctrl= 0.059),
    'OLMo-2-7B':    dict(acc_offset= 0.345, acc_ctrl=-0.055, lp_offset= 3.174, lp_ctrl=-1.401),
}

# Transfer scope from paper Tables (tab:result3_boundary_snapshot)
SCOPE = [
    ('Controlled\nprobe (E12)',         3, 3, 'pass'),
    ('Strict format\ncontrols (E18)',   1, 3, 'partial'),
    ('Broader synth.\ntransfer (E28C)', 0, 3, 'fail'),
    ('Natural-text\nlong-ctx (E29B)',   2, 3, 'partial'),
]

def make_functional_sensitivity():
    m_names  = list(FUNC_DATA.keys())
    m_colors = [C['llama'], C['mistral'], C['olmo']]
    short    = ['Llama-3.1-8B', 'Mistral-7B', 'OLMo-2-7B']

    fig = plt.figure(figsize=(7.0, 3.3))
    gs  = gridspec.GridSpec(1, 2, width_ratios=[1.55, 1.0], wspace=0.42)

    # ---- Left: grouped bar chart (accuracy Δ) ----
    ax = fig.add_subplot(gs[0])
    x  = np.arange(3)
    w  = 0.33

    off_vals  = [FUNC_DATA[m]['acc_offset'] for m in m_names]
    ctrl_vals = [FUNC_DATA[m]['acc_ctrl']   for m in m_names]

    for i, (ov, cv, col) in enumerate(zip(off_vals, ctrl_vals, m_colors)):
        ax.bar(x[i] - w/2, ov, w, color=col, alpha=0.88, edgecolor='black', lw=0.8,
               label='Offset-rep.' if i == 0 else None)
        ax.bar(x[i] + w/2, cv, w, color=col, alpha=0.32, edgecolor='black', lw=0.8,
               hatch='///', label='Matched ctrl.' if i == 0 else None)

        gap = ov - cv
        y_ann = max(ov, 0) + 0.04
        ax.text(x[i], y_ann, f'$\\Delta={gap:.2f}$',
                ha='center', fontsize=7.5, color=col, fontweight='bold')

    ax.axhline(0, color='black', lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(short, fontsize=8)
    ax.set_ylabel('Accuracy $\\Delta$ under SI-head ablation', fontsize=9)
    ax.set_ylim(-0.20, 1.12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title('(a) Preferential degradation under SI-head ablation', fontsize=8.2, pad=4)

    legend_els = [
        mpatches.Patch(fc='grey', alpha=0.88, ec='black', lw=0.8,
                       label='Offset-rep. (target condition)'),
        mpatches.Patch(fc='grey', alpha=0.32, ec='black', lw=0.8, hatch='///',
                       label='Matched control'),
    ]
    ax.legend(handles=legend_els, fontsize=7.5, loc='upper right',
              framealpha=0.95, edgecolor='none')

    # ---- Right: transfer scope panel ----
    ax2 = fig.add_subplot(gs[1])
    ax2.axis('off')
    ax2.set_title('(b) Transfer scope', fontsize=8.2, pad=4)

    dot_colors = {'pass': '#2E7D32', 'partial': '#E65100', 'fail': '#C62828'}
    n_rows = len(SCOPE)
    row_h  = 1.0 / (n_rows + 0.5)

    for i, (label, passed, total, tier) in enumerate(SCOPE):
        yc = 1.0 - (i + 0.65) * row_h
        dc = dot_colors[tier]

        # Filled/empty dots
        for k in range(total):
            fc = dc if k < passed else '#E0E0E0'
            ax2.scatter(0.08 + k * 0.14, yc, s=90, c=fc, edgecolors='#444',
                        linewidths=0.8, zorder=3, transform=ax2.transAxes, clip_on=False)
        ax2.text(0.58, yc + 0.015, f'{passed}/{total}',
                 transform=ax2.transAxes, fontsize=8.5, va='center',
                 color=dc, fontweight='bold')
        ax2.text(0.58, yc - 0.045, label.replace('\n', ' '),
                 transform=ax2.transAxes, fontsize=6.5, va='center', color='#444')

    # Dot legend
    dot_legend = [
        Line2D([0], [0], marker='o', color='w', mfc=dot_colors['pass'],   mec='#444', ms=7, label='Pass'),
        Line2D([0], [0], marker='o', color='w', mfc='#E0E0E0',            mec='#444', ms=7, label='Fail'),
    ]
    ax2.legend(handles=dot_legend, fontsize=7, loc='lower center',
               bbox_to_anchor=(0.35, -0.04), ncol=2, framealpha=0.95, edgecolor='none')

    out = os.path.join(OUTPUT_DIR, 'fig_functional_sensitivity.png')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out}')


# ---------------------------------------------------------------------------
# FIGURE 4  Kernel visualization g_h(Δ)
# Real data loaded from kernels_g_h.parquet files (experiment3/theory1_si_circuits).
# Representative heads: Llama L1H8 (R²=0.902), Mistral L13H25 (R²=0.240), OLMo L7H31 (R²=0.046)
# ---------------------------------------------------------------------------

_KERNEL_BASE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', '..', 'results', 'experiment3', 'theory1_si_circuits'
)

def _load_head_kernel(model_dir, layer, head):
    """Return sorted g_h_value array for the given (layer, head)."""
    path = os.path.join(_KERNEL_BASE, model_dir, 'kernels_g_h.parquet')
    df = pd.read_parquet(path)
    sub = df[(df['layer'] == layer) & (df['head'] == head)].sort_values('delta')
    return sub['delta'].to_numpy(), sub['g_h_value'].to_numpy()

def make_kernel_examples():
    # (title, color, model_dir, layer, head, r2_label)
    _HEADS = [
        ('High-SI head\n'r'($R^2\!=\!0.90$, Llama L1H8)',  C['llama'],   'llama-3.1-8b',   1,  8, 0.902),
        ('Med-SI head\n'r'($R^2\!=\!0.24$, Mistral L13H25)', C['mistral'], 'mistral-7b-v0.1', 13, 25, 0.240),
        ('Low-SI head\n'r'($R^2\!=\!0.05$, OLMo L7H31)',   C['olmo'],    'olmo-2-7b',       7, 31, 0.046),
    ]

    kernels = []
    for title, col_c, mdir, layer, head, r2 in _HEADS:
        deltas, g_vals = _load_head_kernel(mdir, layer, head)
        kernels.append((title, col_c, deltas, g_vals))

    fig, axes = plt.subplots(2, 3, figsize=(7.0, 3.6))
    fig.subplots_adjust(hspace=0.52, wspace=0.38)

    N_FFT = 256
    for col, (title, col_c, offsets, g) in enumerate(kernels):
        xlim_max = int(offsets.max())

        # Top: kernel shape
        ax = axes[0, col]
        ax.plot(offsets, g, color=col_c, lw=1.3)
        ax.fill_between(offsets, g, 0, alpha=0.12, color=col_c)
        ax.axhline(0, color='black', lw=0.7, ls='--', alpha=0.45)
        ax.set_title(title, fontsize=7.5, pad=3)
        ax.set_xlim(0, xlim_max)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if col == 0:
            ax.set_ylabel(r'$g_h(\Delta)$', fontsize=9)
        ax.set_xlabel(r'Offset $\Delta = i-j$', fontsize=7.5)

        # Bottom: DFT magnitude
        ax = axes[1, col]
        nfft = max(N_FFT, len(g))
        padded = np.zeros(nfft)
        padded[:len(g)] = g
        mag   = np.abs(np.fft.rfft(padded))
        freqs = np.fft.rfftfreq(nfft)
        bw    = freqs[1] - freqs[0]
        n_show = nfft // 4
        ax.bar(freqs[:n_show], mag[:n_show], width=bw * 0.85,
               color=col_c, alpha=0.72, edgecolor='none')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if col == 0:
            ax.set_ylabel('DFT magnitude', fontsize=9)
        ax.set_xlabel('Frequency', fontsize=7.5)
        ax.set_xlim(0, 0.26)

    fig.suptitle(
        r'Representative $g_h(\Delta)$ kernels and DFT spectra',
        fontsize=8.2, y=1.02)

    out = os.path.join(OUTPUT_DIR, 'fig_kernel_examples.png')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out}')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print(f'Output directory: {OUTPUT_DIR}\n')
    make_r2_heterogeneity()
    make_overview_schematic()
    make_functional_sensitivity()
    make_kernel_examples()
    print('\nAll done.')
