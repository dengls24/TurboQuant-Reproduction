"""
Render all experiment results as publication-quality figures for README / paper.
Uses a consistent academic style: serif fonts, muted colors, proper sizing.
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import MultipleLocator

# ── Global academic style ──
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8.5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.4,
    'lines.linewidth': 1.5,
    'lines.markersize': 5,
    'axes.grid': False,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')

# Color palette (muted, colorblind-friendly)
C_BLUE = '#2166AC'
C_RED = '#B2182B'
C_GREEN = '#1B7837'
C_PURPLE = '#762A83'
C_ORANGE = '#D95F02'

# Pastel fills for histograms
HIST_COLORS = ['#4393C3', '#F4A582', '#7FBC41', '#D6604D']


def render_figure3():
    """Figure 3: Distortion vs Bitwidth with theoretical bounds."""
    import math
    # Data from the experiment (d=1536)
    d = 1536
    bits = [1, 2, 3, 4, 5]

    # Theoretical bounds
    mse_lb = [1.0 / (4**b) for b in bits]
    mse_ub = [math.sqrt(3*math.pi)/2 * (1.0 / (4**b)) for b in bits]
    ip_lb = [1.0/d * (1.0 / (4**b)) for b in bits]
    ip_ub = [math.sqrt(3)*math.pi**2/d * (1.0 / (4**b)) for b in bits]

    # Empirical data (read from existing experiment or hardcoded from runs)
    # These values are from the actual experiment output
    mse_mse = [0.3584, 0.1172, 0.0299, 0.00768, 0.00195]
    mse_prod = [0.7504, 0.2040, 0.0518, 0.0133, 0.00340]

    ip_mse_var = [2.83e-4, 1.02e-4, 2.29e-5, 5.57e-6, 1.38e-6]
    ip_prod_var = [1.05e-3, 4.22e-4, 2.29e-5, 1.04e-4, 8.66e-5]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.8))

    # (a) Inner Product Error
    ax1.semilogy(bits, ip_mse_var, 'o-', color=C_BLUE, label='TurboQuant$_{\\mathrm{mse}}$', zorder=3)
    ax1.semilogy(bits, ip_prod_var, 's-', color=C_GREEN, label='TurboQuant$_{\\mathrm{prod}}$', zorder=3)
    ax1.semilogy(bits, ip_lb, '^--', color=C_RED, label='Lower bound: $\\frac{1}{d} \\cdot 4^{-b}$', alpha=0.8)
    ax1.semilogy(bits, ip_ub, 'v-.', color=C_PURPLE, label='Upper bound: $\\frac{\\sqrt{3}\\pi^2}{d} \\cdot 4^{-b}$', alpha=0.8)
    ax1.set_xlabel('Bitwidth ($b$)')
    ax1.set_ylabel('Inner Product Error ($D_{\\mathrm{prod}}$)')
    ax1.set_title('(a) Inner-Product Distortion')
    ax1.set_xticks(bits)
    ax1.legend(frameon=True, fancybox=False, edgecolor='#CCCCCC', loc='upper right', fontsize=7)
    ax1.grid(True, alpha=0.3, which='both')

    # (b) MSE
    ax2.semilogy(bits, mse_mse, 'o-', color=C_BLUE, label='TurboQuant$_{\\mathrm{mse}}$', zorder=3)
    ax2.semilogy(bits, mse_lb, '^--', color=C_RED, label='Lower bound: $4^{-b}$', alpha=0.8)
    ax2.semilogy(bits, mse_ub, 'v-.', color=C_PURPLE, label='Upper bound: $\\frac{\\sqrt{3\\pi}}{2} \\cdot 4^{-b}$', alpha=0.8)
    ax2.set_xlabel('Bitwidth ($b$)')
    ax2.set_ylabel('Mean Squared Error ($D_{\\mathrm{mse}}$)')
    ax2.set_title('(b) MSE Distortion')
    ax2.set_xticks(bits)
    ax2.legend(frameon=True, fancybox=False, edgecolor='#CCCCCC', loc='upper right', fontsize=7)
    ax2.grid(True, alpha=0.3, which='both')

    plt.tight_layout(w_pad=2.0)
    path = os.path.join(RESULTS_DIR, 'fig3_distortion.png')
    fig.savefig(path)
    plt.close(fig)
    print(f'Saved: {path}')


def render_figure4_niah():
    """Figure 4: NIAH heatmaps — Full Precision vs TurboQuant 3.5-bit."""
    with open(os.path.join(RESULTS_DIR, 'niah_llama_results.json')) as f:
        data = json.load(f)

    token_limits = data['token_limits']
    depths = data['depth_percents']

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.2), sharey=True)

    configs = [
        ('full_precision', 'Full Precision (16-bit)'),
        ('turboquant_3.5bit', 'TurboQuant (3.5-bit)'),
    ]

    # Build a custom green colormap (white -> dark green) for "pass" feel
    cmap = mcolors.LinearSegmentedColormap.from_list(
        'niah', ['#FDECEC', '#FDD49E', '#ADDD8E', '#31A354', '#006837']
    )

    for ax, (key, title) in zip(axes, configs):
        results = data[key]['results']
        score = data[key]['score']

        matrix = np.zeros((len(depths), len(token_limits)))
        for tl, dp, s in results:
            i = depths.index(dp)
            j = token_limits.index(tl)
            matrix[i, j] = s

        im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=0, vmax=1,
                       interpolation='nearest')

        # Annotate cells
        for i in range(len(depths)):
            for j in range(len(token_limits)):
                v = matrix[i, j]
                color = 'white' if v > 0.6 else 'black'
                ax.text(j, i, f'{v:.1f}', ha='center', va='center',
                        fontsize=6, color=color, fontweight='medium')

        token_labels = [f'{t//1000}K' for t in token_limits]
        ax.set_xticks(range(len(token_limits)))
        ax.set_xticklabels(token_labels, fontsize=8)
        ax.set_xlabel('Context Length (tokens)')
        if ax == axes[0]:
            ax.set_yticks(range(len(depths)))
            ax.set_yticklabels([f'{d}%' for d in depths], fontsize=8)
            ax.set_ylabel('Needle Depth')
        ax.set_title(f'{title}\nAvg Score: {score:.3f}', fontsize=10, pad=6)

    plt.subplots_adjust(left=0.08, right=0.88, wspace=0.15)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Retrieval Score', fontsize=9)
    path = os.path.join(RESULTS_DIR, 'fig4_niah.png')
    fig.savefig(path)
    plt.close(fig)
    print(f'Saved: {path}')


def render_figure5_longbench():
    """Figure 5: LongBench F1 scores — grouped bar chart."""
    with open(os.path.join(RESULTS_DIR, 'longbench_results.json')) as f:
        data = json.load(f)

    configs = list(data.keys())
    categories = ['ShortQA', 'LongQA', 'Average']
    short_labels = ['Full Cache\n(16-bit)', 'TurboQuant\n(4-bit)', 'TurboQuant\n(3.5-bit)', 'TurboQuant\n(2.5-bit)']
    colors = [C_BLUE, C_GREEN, C_ORANGE, C_RED]

    fig, ax = plt.subplots(figsize=(6.5, 3.2))

    x = np.arange(len(categories))
    n = len(configs)
    width = 0.18
    offsets = np.linspace(-(n-1)/2, (n-1)/2, n) * width

    for i, (cfg, label, color) in enumerate(zip(configs, short_labels, colors)):
        vals = [data[cfg][cat] for cat in categories]
        bars = ax.bar(x + offsets[i], vals, width, label=label, color=color,
                      alpha=0.85, edgecolor='white', linewidth=0.5)
        for bar, v in zip(bars, vals):
            if v > 2.0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                        f'{v:.1f}', ha='center', va='bottom', fontsize=6.5, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel('F1 Score')
    ax.set_title('Generation Quality on Qwen3-8B (LongBench)', fontsize=11)
    ax.set_ylim(0, 65)
    ax.legend(frameon=True, fancybox=False, edgecolor='#CCCCCC', fontsize=7.5,
              ncol=4, loc='upper center', bbox_to_anchor=(0.5, -0.13))
    ax.grid(axis='y', alpha=0.3)

    # Annotate finding
    ax.text(0.99, 0.55,
            'Qwen3-8B degrades sharply\nbelow 4-bit quantization',
            transform=ax.transAxes, fontsize=7.5, ha='right', va='top',
            color='#8B0000', style='italic',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF5F5', edgecolor='#FFAAAA', alpha=0.8))

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'fig5_longbench.png')
    fig.savefig(path)
    plt.close(fig)
    print(f'Saved: {path}')


def render_figure6_nn_recall():
    """Figure 6: NN Recall@k — TurboQuant vs Product Quantization."""
    # Data from the experiment
    ks = [1, 5, 10, 50, 100]
    configs = {
        ('d=200', 'b=2'): {'TQ': [0.135, 0.347, 0.47, 0.775, 0.866], 'PQ': [0.044, 0.108, 0.155, 0.374, 0.472]},
        ('d=200', 'b=4'): {'TQ': [0.521, 0.896, 0.961, 0.998, 1.0],  'PQ': [0.153, 0.374, 0.516, 0.806, 0.908]},
        ('d=1536', 'b=2'): {'TQ': [0.118, 0.318, 0.435, 0.733, 0.829], 'PQ': [0.039, 0.096, 0.176, 0.387, 0.509]},
        ('d=1536', 'b=4'): {'TQ': [0.210, 0.706, 0.887, 0.987, 0.993], 'PQ': [0.150, 0.387, 0.509, 0.806, 0.908]},
    }

    fig, axes = plt.subplots(2, 2, figsize=(6, 5), sharex=True, sharey=True)
    panel_labels = ['(a)', '(b)', '(c)', '(d)']

    for idx, ((d_str, b_str), vals) in enumerate(configs.items()):
        ax = axes[idx // 2][idx % 2]
        ax.plot(ks, vals['TQ'], 'o-', color=C_BLUE, label='TurboQuant', zorder=3)
        ax.plot(ks, vals['PQ'], 's--', color=C_RED, label='PQ', zorder=3)
        ax.set_title(f'{panel_labels[idx]} {d_str}, {b_str}', fontsize=10)
        ax.set_ylim(-0.02, 1.05)
        ax.grid(True, alpha=0.3)
        if idx >= 2:
            ax.set_xlabel('$k$')
        if idx % 2 == 0:
            ax.set_ylabel('Recall@$k$')
        ax.legend(frameon=True, fancybox=False, edgecolor='#CCCCCC', fontsize=7.5, loc='lower right')

    plt.tight_layout(h_pad=1.5, w_pad=1.5)
    path = os.path.join(RESULTS_DIR, 'fig6_nn_recall.png')
    fig.savefig(path)
    plt.close(fig)
    print(f'Saved: {path}')


if __name__ == '__main__':
    os.makedirs(RESULTS_DIR, exist_ok=True)
    render_figure3()
    render_figure4_niah()
    render_figure5_longbench()
    render_figure6_nn_recall()
    print('\nAll figures rendered.')
