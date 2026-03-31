"""
Experiment 4.1: Empirical Validation of TurboQuant
Reproduces Figures 1 and 3 from the paper.

- Figure 1: Error distribution histograms of inner product distortion
  for TurboQuant_prod and TurboQuant_mse at bit-widths 1-4
- Figure 3: MSE and inner-product error vs bit-width with theoretical bounds
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from turboquant import (
    TurboQuantMSE, TurboQuantProd,
    compute_mse_distortion, compute_inner_product_distortion,
    theoretical_mse_upper_bound, theoretical_mse_lower_bound,
    theoretical_ip_upper_bound, theoretical_ip_lower_bound,
)

# ─── Config ───
D = 1536           # dimension (matches paper's OpenAI3 embeddings)
N_TRAIN = 10000    # number of training vectors (paper uses 100K, we use 10K for speed)
N_QUERY = 1000     # number of query vectors
BIT_WIDTHS = [1, 2, 3, 4, 5]
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Config: d={D}, n_train={N_TRAIN}, n_query={N_QUERY}, device={DEVICE}")


# ─── Generate synthetic data on the unit hypersphere ───
print("\nGenerating synthetic data on S^{d-1}...")
torch.manual_seed(SEED)
x_train = torch.randn(N_TRAIN, D, device=DEVICE)
x_train = x_train / torch.norm(x_train, dim=-1, keepdim=True)

x_query = torch.randn(N_QUERY, D, device=DEVICE)
x_query = x_query / torch.norm(x_query, dim=-1, keepdim=True)

print(f"  x_train: {x_train.shape}, x_query: {x_query.shape}")
print(f"  Norms check — train: {x_train.norm(dim=-1).mean():.4f}, query: {x_query.norm(dim=-1).mean():.4f}")


# ═══════════════════════════════════════════════════════════════════
# Figure 1: Inner product error histograms
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Figure 1: Inner Product Error Distribution Histograms")
print("=" * 60)

fig1, axes = plt.subplots(2, 4, figsize=(20, 8))
fig1.suptitle("Inner Product Distortion Distribution", fontsize=16, y=1.02)

colors_prod = ['#3498db', '#e67e22', '#2ecc71', '#e74c3c']
colors_mse = ['#3498db', '#e67e22', '#2ecc71', '#e74c3c']

for idx, b in enumerate([1, 2, 3, 4]):
    print(f"\n  Bit-width b={b}:")

    # --- TurboQuant_prod ---
    t0 = time.time()
    tq_prod = TurboQuantProd(D, b, device=DEVICE, seed=SEED)
    x_hat_prod = tq_prod.quantize_dequantize(x_train)
    t_prod = time.time() - t0

    # Compute per-pair inner product errors
    # Use a subset of queries for the histogram
    n_hist = min(200, N_QUERY)
    ip_orig = (x_query[:n_hist].unsqueeze(1) * x_train.unsqueeze(0)).sum(-1)  # (n_hist, N_TRAIN)
    ip_hat = (x_query[:n_hist].unsqueeze(1) * x_hat_prod.unsqueeze(0)).sum(-1)
    ip_errors_prod = (ip_orig - ip_hat).flatten().cpu().numpy()

    ax = axes[0, idx]
    ax.hist(ip_errors_prod, bins=100, color=colors_prod[idx], alpha=0.7, density=True)
    ax.set_title(f"Bitwidth = {b}", fontsize=12)
    ax.set_xlabel("Inner Product Distortion")
    if idx == 0:
        ax.set_ylabel("Frequency")
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))

    print(f"    TurboQuant_prod: mean_err={ip_errors_prod.mean():.6f}, std={ip_errors_prod.std():.6f}, time={t_prod:.2f}s")

    # --- TurboQuant_mse ---
    t0 = time.time()
    tq_mse = TurboQuantMSE(D, b, device=DEVICE, seed=SEED)
    x_hat_mse = tq_mse.quantize_dequantize(x_train)
    t_mse = time.time() - t0

    ip_hat_mse = (x_query[:n_hist].unsqueeze(1) * x_hat_mse.unsqueeze(0)).sum(-1)
    ip_errors_mse = (ip_orig - ip_hat_mse).flatten().cpu().numpy()

    ax = axes[1, idx]
    ax.hist(ip_errors_mse, bins=100, color=colors_mse[idx], alpha=0.7, density=True)
    ax.set_title(f"Bitwidth = {b}", fontsize=12)
    ax.set_xlabel("Inner Product Distortion")
    if idx == 0:
        ax.set_ylabel("Frequency")
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))

    print(f"    TurboQuant_mse:  mean_err={ip_errors_mse.mean():.6f}, std={ip_errors_mse.std():.6f}, time={t_mse:.2f}s")

axes[0, 0].annotate("(a) TurboQuant_prod", xy=(0.5, 1.15), xycoords='axes fraction',
                      fontsize=14, ha='center', fontweight='bold')
axes[1, 0].annotate("(b) TurboQuant_mse", xy=(0.5, 1.15), xycoords='axes fraction',
                      fontsize=14, ha='center', fontweight='bold')

plt.tight_layout()
fig1_path = os.path.join(OUTPUT_DIR, 'figure1_ip_error_histograms.png')
fig1.savefig(fig1_path, dpi=150, bbox_inches='tight')
print(f"\nFigure 1 saved to: {fig1_path}")


# ═══════════════════════════════════════════════════════════════════
# Figure 3: MSE and Inner-Product Error vs Bit-width
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Figure 3: MSE and Inner-Product Error vs Bit-width")
print("=" * 60)

mse_results_mse = []
mse_results_prod = []
ip_results_mse = []
ip_results_prod = []
ub_mse_list = []
lb_mse_list = []
ub_ip_list = []
lb_ip_list = []

for b in BIT_WIDTHS:
    print(f"\n  Bit-width b={b}:")

    # MSE quantizer
    tq_mse = TurboQuantMSE(D, b, device=DEVICE, seed=SEED)
    x_hat_mse = tq_mse.quantize_dequantize(x_train)
    mse_val = compute_mse_distortion(x_train, x_hat_mse)
    bias_mse, var_mse = compute_inner_product_distortion(x_train, x_hat_mse, x_query)
    mse_results_mse.append(mse_val)
    ip_results_mse.append(var_mse)

    # Prod quantizer
    tq_prod = TurboQuantProd(D, b, device=DEVICE, seed=SEED)
    x_hat_prod = tq_prod.quantize_dequantize(x_train)
    mse_prod = compute_mse_distortion(x_train, x_hat_prod)
    bias_prod, var_prod = compute_inner_product_distortion(x_train, x_hat_prod, x_query)
    mse_results_prod.append(mse_prod)
    ip_results_prod.append(var_prod)

    # Theoretical bounds
    ub_mse_list.append(theoretical_mse_upper_bound(b))
    lb_mse_list.append(theoretical_mse_lower_bound(b))
    ub_ip_list.append(theoretical_ip_upper_bound(b, D))
    lb_ip_list.append(theoretical_ip_lower_bound(b, D))

    print(f"    MSE:  mse_quant={mse_val:.6f}, prod_quant={mse_prod:.6f}, bounds=[{lb_mse_list[-1]:.6f}, {ub_mse_list[-1]:.6f}]")
    print(f"    IP:   mse_quant={var_mse:.6f} (bias={bias_mse:.6f}), prod_quant={var_prod:.6f} (bias={bias_prod:.6f})")

# Plot Figure 3
fig3, (ax_ip, ax_mse) = plt.subplots(1, 2, figsize=(14, 6))

# (a) Inner-product error
ax_ip.semilogy(BIT_WIDTHS, ip_results_mse, 'b-o', label='TurboQuant_mse', linewidth=2, markersize=8)
ax_ip.semilogy(BIT_WIDTHS, ip_results_prod, 'g-s', label='TurboQuant_prod', linewidth=2, markersize=8)
ax_ip.semilogy(BIT_WIDTHS, lb_ip_list, 'r--^', label=f'Lower Bound: 1/d·4^{{-b}}', linewidth=2, markersize=8)
ax_ip.semilogy(BIT_WIDTHS, ub_ip_list, 'm--v', label=f'Upper Bound: √3π²/d·4^{{-b}}', linewidth=2, markersize=8)
ax_ip.set_xlabel('Bitwidth (b)', fontsize=13)
ax_ip.set_ylabel('Inner Product Error (D_prod)', fontsize=13)
ax_ip.set_title('(a) inner-prod error', fontsize=14)
ax_ip.legend(fontsize=11)
ax_ip.set_xticks(BIT_WIDTHS)
ax_ip.grid(True, alpha=0.3)

# (b) MSE
ax_mse.semilogy(BIT_WIDTHS, mse_results_mse, 'b-o', label='TurboQuant_mse', linewidth=2, markersize=8)
ax_mse.semilogy(BIT_WIDTHS, lb_mse_list, 'r--^', label=f'Lower Bound: 4^{{-b}}', linewidth=2, markersize=8)
ax_mse.semilogy(BIT_WIDTHS, ub_mse_list, 'm--v', label=f'Upper Bound: √(3π)/2·4^{{-b}}', linewidth=2, markersize=8)
ax_mse.set_xlabel('Bitwidth (b)', fontsize=13)
ax_mse.set_ylabel('Mean Squared Error (D_mse)', fontsize=13)
ax_mse.set_title('(b) MSE', fontsize=14)
ax_mse.legend(fontsize=11)
ax_mse.set_xticks(BIT_WIDTHS)
ax_mse.grid(True, alpha=0.3)

plt.tight_layout()
fig3_path = os.path.join(OUTPUT_DIR, 'figure3_distortion_vs_bitwidth.png')
fig3.savefig(fig3_path, dpi=150, bbox_inches='tight')
print(f"\nFigure 3 saved to: {fig3_path}")


# ═══════════════════════════════════════════════════════════════════
# Summary Table
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Summary Table")
print("=" * 60)
print(f"{'b':>3} | {'MSE(mse)':>10} | {'MSE(prod)':>10} | {'MSE LB':>10} | {'MSE UB':>10} | {'IP(mse)':>10} | {'IP(prod)':>10} | {'IP LB':>10} | {'IP UB':>10}")
print("-" * 107)
for i, b in enumerate(BIT_WIDTHS):
    print(f"{b:3d} | {mse_results_mse[i]:10.6f} | {mse_results_prod[i]:10.6f} | {lb_mse_list[i]:10.6f} | {ub_mse_list[i]:10.6f} | "
          f"{ip_results_mse[i]:10.6f} | {ip_results_prod[i]:10.6f} | {lb_ip_list[i]:10.6f} | {ub_ip_list[i]:10.6f}")

print("\nDone!")
