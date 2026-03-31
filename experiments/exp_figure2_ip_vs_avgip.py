"""
Experiment: Figure 2 from the paper.
IP error distribution conditioned on average inner product, at bit-width b=2.

Shows:
  (a) TurboQuant_prod: variance remains constant regardless of avg IP
  (b) TurboQuant_mse: bias increases with average inner product
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
from turboquant import TurboQuantMSE, TurboQuantProd

# ─── Config ───
D = 1536
N_DB = 5000
N_QUERY = 500
B = 2  # bit-width fixed at 2 as in the paper
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Target average inner product bins
TARGET_AVG_IPS = [0.01, 0.06, 0.10, 0.17]
BIN_HALF_WIDTH = 0.02

print(f"Config: d={D}, n_db={N_DB}, n_query={N_QUERY}, b={B}, device={DEVICE}")

# ─── Generate data with varied inner products ───
# Use multiple cluster tightness levels to populate all IP ranges
print("\nGenerating clustered data for varied inner products...")
torch.manual_seed(SEED)

n_clusters = 10
cluster_centers = torch.randn(n_clusters, D, device=DEVICE)
cluster_centers = cluster_centers / torch.norm(cluster_centers, dim=-1, keepdim=True)

mix_alphas = [0.2, 0.3, 0.4, 0.5, 0.6]
n_per_group_db = N_DB // (n_clusters * len(mix_alphas))
n_per_group_q = N_QUERY // (n_clusters * len(mix_alphas))

x_db_list = []
for alpha in mix_alphas:
    for c in range(n_clusters):
        noise = torch.randn(n_per_group_db, D, device=DEVICE)
        noise = noise / torch.norm(noise, dim=-1, keepdim=True)
        x_c = alpha * cluster_centers[c:c+1] + (1 - alpha) * noise
        x_c = x_c / torch.norm(x_c, dim=-1, keepdim=True)
        x_db_list.append(x_c)
x_db = torch.cat(x_db_list, dim=0)

x_query_list = []
for alpha in mix_alphas:
    for c in range(n_clusters):
        noise = torch.randn(n_per_group_q, D, device=DEVICE)
        noise = noise / torch.norm(noise, dim=-1, keepdim=True)
        x_c = alpha * cluster_centers[c:c+1] + (1 - alpha) * noise
        x_c = x_c / torch.norm(x_c, dim=-1, keepdim=True)
        x_query_list.append(x_c)
x_query = torch.cat(x_query_list, dim=0)

# Compute all pairwise inner products
print("Computing pairwise inner products...")
ip_orig = x_query @ x_db.T  # (N_QUERY, N_DB)
print(f"  IP range: [{ip_orig.min().item():.4f}, {ip_orig.max().item():.4f}]")
print(f"  IP mean: {ip_orig.mean().item():.4f}, std: {ip_orig.std().item():.4f}")

# Show distribution at key percentiles
ip_flat = ip_orig.flatten().cpu().numpy()
for p in [50, 75, 90, 95, 99]:
    print(f"  {p}th percentile: {np.percentile(ip_flat, p):.4f}")

# Quantize database vectors
print("\nQuantizing database vectors...")
tq_prod = TurboQuantProd(D, B, device=DEVICE, seed=SEED)
x_hat_prod = tq_prod.quantize_dequantize(x_db)

tq_mse = TurboQuantMSE(D, B, device=DEVICE, seed=SEED)
x_hat_mse = tq_mse.quantize_dequantize(x_db)

# Compute quantized inner products
ip_hat_prod = x_query @ x_hat_prod.T
ip_hat_mse = x_query @ x_hat_mse.T

# Errors
err_prod = (ip_orig - ip_hat_prod).cpu().numpy()
err_mse = (ip_orig - ip_hat_mse).cpu().numpy()
ip_orig_np = ip_orig.cpu().numpy()

# ═══════════════════════════════════════════════════════════════════
# Figure 2
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Figure 2: IP Error Distribution vs Average Inner Product (b=2)")
print("=" * 60)

fig, axes = plt.subplots(2, 4, figsize=(20, 8))

colors = ['#3498db', '#e67e22', '#2ecc71', '#e74c3c']

for idx, target_ip in enumerate(TARGET_AVG_IPS):
    mask = np.abs(ip_orig_np - target_ip) < BIN_HALF_WIDTH
    n_pairs = mask.sum()

    if n_pairs == 0:
        # Fallback: use wider bin
        for bw in [0.03, 0.04, 0.05, 0.08, 0.1]:
            mask = np.abs(ip_orig_np - target_ip) < bw
            n_pairs = mask.sum()
            if n_pairs >= 1000:
                break

    actual_avg = ip_orig_np[mask].mean() if n_pairs > 0 else target_ip
    print(f"\n  Target Avg IP = {target_ip:.2f} (found {n_pairs} pairs, actual mean={actual_avg:.4f}):")

    # --- TurboQuant_prod ---
    errors_prod = err_prod[mask] if n_pairs > 0 else np.array([0.0])
    ax = axes[0, idx]
    ax.hist(errors_prod, bins=100, color=colors[idx], alpha=0.7, density=True)
    ax.set_title(f"Avg IP = {actual_avg:.2f}", fontsize=12)
    ax.set_xlabel("Inner Product Distortion")
    if idx == 0:
        ax.set_ylabel("Frequency")
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax.set_xlim([-0.06, 0.06])
    print(f"    Prod: mean_err={errors_prod.mean():.6f}, std={errors_prod.std():.6f}")

    # --- TurboQuant_mse ---
    errors_mse = err_mse[mask] if n_pairs > 0 else np.array([0.0])
    ax = axes[1, idx]
    ax.hist(errors_mse, bins=100, color=colors[idx], alpha=0.7, density=True)
    ax.set_title(f"Avg IP = {actual_avg:.2f}", fontsize=12)
    ax.set_xlabel("Inner Product Distortion")
    if idx == 0:
        ax.set_ylabel("Frequency")
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax.set_xlim([-0.06, 0.06])
    print(f"    MSE:  mean_err={errors_mse.mean():.6f}, std={errors_mse.std():.6f}")

axes[0, 0].annotate("(a) TurboQuant_prod", xy=(0.5, 1.15), xycoords='axes fraction',
                      fontsize=14, ha='center', fontweight='bold')
axes[1, 0].annotate("(b) TurboQuant_mse", xy=(0.5, 1.15), xycoords='axes fraction',
                      fontsize=14, ha='center', fontweight='bold')

plt.tight_layout()
fig_path = os.path.join(OUTPUT_DIR, 'figure2_ip_vs_avgip.png')
fig.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"\nFigure 2 saved to: {fig_path}")
print("Done!")
