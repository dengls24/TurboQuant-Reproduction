"""
Experiment 4.4: Nearest Neighbor Search with TurboQuant
Reproduces Table 2 (quantization time) and recall@k comparisons from the paper.

Compares:
  - TurboQuant (online, training-free)
  - Naive Product Quantization (requires k-means training)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from turboquant import TurboQuantMSE, TurboQuantProd

# ─── Config ───
DIMS = [200, 1536]     # dimensions to test
N_TRAIN = 10000        # database vectors
N_QUERY = 1000         # query vectors
BIT_WIDTHS = [2, 4]    # bit-widths for NN search
TOP_K_VALUES = [1, 5, 10, 50, 100]
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─── Simple Product Quantization baseline ───
class SimpleProductQuantization:
    """
    Basic Product Quantization (PQ) baseline.
    Splits vectors into sub-vectors and runs k-means on each.
    """
    def __init__(self, d: int, b: int, n_subvectors: int = None, device=None, seed=42):
        self.d = d
        self.b = b
        self.device = device or torch.device('cpu')
        # Number of sub-vectors: choose so each sub-vector has b bits
        # PQ with n_sub sub-vectors, each with 2^b_sub codebook entries
        # Total bits = n_sub * b_sub = d * b (approximately)
        # We'll use sub_dim = d / n_sub, b_sub = b bits per sub-vector
        self.n_sub = min(d, max(1, d // 4))  # sub-vector groups of size 4
        self.sub_dim = d // self.n_sub
        assert self.sub_dim * self.n_sub == d, f"d={d} must be divisible by n_sub={self.n_sub}"
        self.n_codes = 2 ** b  # codebook size per sub-vector
        self.codebooks = None
        torch.manual_seed(seed)

    def train(self, x: torch.Tensor, n_iter: int = 20):
        """Train PQ codebooks using k-means on each sub-vector."""
        n = x.shape[0]
        self.codebooks = []

        for s in range(self.n_sub):
            sub = x[:, s * self.sub_dim:(s + 1) * self.sub_dim]  # (n, sub_dim)

            # k-means
            indices = torch.randperm(n, device=self.device)[:self.n_codes]
            centroids = sub[indices].clone()  # (n_codes, sub_dim)

            for _ in range(n_iter):
                dists = torch.cdist(sub, centroids)  # (n, n_codes)
                assign = dists.argmin(dim=1)  # (n,)
                for k in range(self.n_codes):
                    mask = assign == k
                    if mask.sum() > 0:
                        centroids[k] = sub[mask].mean(dim=0)

            self.codebooks.append(centroids)

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize vectors. Returns indices (n, n_sub)."""
        n = x.shape[0]
        indices = torch.zeros(n, self.n_sub, dtype=torch.long, device=self.device)
        for s in range(self.n_sub):
            sub = x[:, s * self.sub_dim:(s + 1) * self.sub_dim]
            dists = torch.cdist(sub, self.codebooks[s])
            indices[:, s] = dists.argmin(dim=1)
        return indices

    def dequantize(self, indices: torch.Tensor) -> torch.Tensor:
        """Dequantize from indices."""
        n = indices.shape[0]
        x_hat = torch.zeros(n, self.d, device=self.device)
        for s in range(self.n_sub):
            x_hat[:, s * self.sub_dim:(s + 1) * self.sub_dim] = self.codebooks[s][indices[:, s]]
        return x_hat

    def quantize_dequantize(self, x: torch.Tensor) -> torch.Tensor:
        indices = self.quantize(x)
        return self.dequantize(indices)


def recall_at_k(x_query, x_db, x_db_hat, k):
    """
    Compute recall@k: fraction of queries where the true top-1
    by inner product is found in the top-k of the quantized approximation.
    """
    # True inner products
    ip_true = x_query @ x_db.T  # (n_query, n_db)
    # Approximate inner products
    ip_approx = x_query @ x_db_hat.T  # (n_query, n_db)

    # True top-1 for each query
    true_top1 = ip_true.argmax(dim=1)  # (n_query,)

    # Approximate top-k for each query
    _, approx_topk = ip_approx.topk(k, dim=1)  # (n_query, k)

    # Check if true top-1 is in approximate top-k
    hits = (approx_topk == true_top1.unsqueeze(1)).any(dim=1)
    return hits.float().mean().item()


# ═══════════════════════════════════════════════════════════════════
# Main Experiments
# ═══════════════════════════════════════════════════════════════════
print("=" * 70)
print("Nearest Neighbor Search Experiment (Section 4.4)")
print("=" * 70)

all_results = {}

for d in DIMS:
    print(f"\n{'=' * 50}")
    print(f"Dimension d = {d}")
    print(f"{'=' * 50}")

    # Generate data
    torch.manual_seed(SEED)
    x_db = torch.randn(N_TRAIN, d, device=DEVICE)
    x_db = x_db / torch.norm(x_db, dim=-1, keepdim=True)
    x_query = torch.randn(N_QUERY, d, device=DEVICE)
    x_query = x_query / torch.norm(x_query, dim=-1, keepdim=True)

    for b in BIT_WIDTHS:
        print(f"\n  --- Bit-width b = {b} ---")

        # ── TurboQuant_prod ──
        t0 = time.time()
        tq = TurboQuantProd(d, b, device=DEVICE, seed=SEED)
        x_hat_tq = tq.quantize_dequantize(x_db)
        time_tq = time.time() - t0

        # ── Product Quantization ──
        t0 = time.time()
        pq = SimpleProductQuantization(d, b, device=DEVICE, seed=SEED)
        pq.train(x_db)
        x_hat_pq = pq.quantize_dequantize(x_db)
        time_pq = time.time() - t0

        print(f"  Quantization time — TurboQuant: {time_tq:.4f}s, PQ: {time_pq:.4f}s")

        # ── Recall@k ──
        for k in TOP_K_VALUES:
            r_tq = recall_at_k(x_query, x_db, x_hat_tq, k)
            r_pq = recall_at_k(x_query, x_db, x_hat_pq, k)
            print(f"    Recall@{k:3d}: TurboQuant={r_tq:.4f}, PQ={r_pq:.4f}")

            key = (d, b, k)
            all_results[key] = {'tq': r_tq, 'pq': r_pq}


# ═══════════════════════════════════════════════════════════════════
# Table 2: Quantization Time
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Table 2: Quantization Time (seconds)")
print("=" * 60)

print(f"{'Method':<25} | ", end='')
for d in DIMS:
    print(f"{'d=' + str(d):>10} | ", end='')
print()
print("-" * (30 + 14 * len(DIMS)))

for method_name in ['TurboQuant', 'Product Quantization']:
    print(f"{method_name:<25} | ", end='')
    for d in DIMS:
        torch.manual_seed(SEED)
        x_test = torch.randn(N_TRAIN, d, device=DEVICE)
        x_test = x_test / torch.norm(x_test, dim=-1, keepdim=True)

        t0 = time.time()
        if method_name == 'TurboQuant':
            tq = TurboQuantProd(d, 4, device=DEVICE, seed=SEED)
            _ = tq.quantize_dequantize(x_test)
        else:
            pq = SimpleProductQuantization(d, 4, device=DEVICE, seed=SEED)
            pq.train(x_test)
            _ = pq.quantize_dequantize(x_test)
        elapsed = time.time() - t0
        print(f"{elapsed:10.4f} | ", end='')
    print()


# ═══════════════════════════════════════════════════════════════════
# Plot: Recall@k comparison
# ═══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(len(DIMS), len(BIT_WIDTHS), figsize=(6 * len(BIT_WIDTHS), 5 * len(DIMS)))
if len(DIMS) == 1:
    axes = axes.reshape(1, -1)
if len(BIT_WIDTHS) == 1:
    axes = axes.reshape(-1, 1)

for i, d in enumerate(DIMS):
    for j, b in enumerate(BIT_WIDTHS):
        ax = axes[i, j]
        tq_recalls = [all_results[(d, b, k)]['tq'] for k in TOP_K_VALUES]
        pq_recalls = [all_results[(d, b, k)]['pq'] for k in TOP_K_VALUES]

        ax.plot(TOP_K_VALUES, tq_recalls, 'b-o', label='TurboQuant', linewidth=2, markersize=6)
        ax.plot(TOP_K_VALUES, pq_recalls, 'r--s', label='PQ', linewidth=2, markersize=6)
        ax.set_xlabel('k', fontsize=12)
        ax.set_ylabel('Recall@k', fontsize=12)
        ax.set_title(f'd={d}, b={b}', fontsize=13)
        ax.legend(fontsize=11)
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3)

plt.tight_layout()
fig_path = os.path.join(OUTPUT_DIR, 'figure_nn_recall.png')
fig.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"\nRecall plot saved to: {fig_path}")
print("\nDone!")
