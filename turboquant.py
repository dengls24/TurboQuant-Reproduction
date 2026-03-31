"""
TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate

Implementation based on:
  Zandieh et al., "TurboQuant: Online Vector Quantization with Near-optimal
  Distortion Rate", ICLR 2026. arXiv:2504.19874

Two algorithms:
  1. TurboQuant_mse  (Algorithm 1) — minimizes MSE distortion
  2. TurboQuant_prod (Algorithm 2) — unbiased inner-product quantizer
"""

import math
import torch
import numpy as np
from scipy.special import gammaln
from typing import Tuple, Optional, List


# ──────────────────────────────────────────────────────────────────────
# 1. Beta distribution PDF for coordinates on the unit hypersphere
# ──────────────────────────────────────────────────────────────────────

def beta_pdf(x: np.ndarray, d: int) -> np.ndarray:
    """
    PDF of a single coordinate of a uniformly random point on S^{d-1}.
    f_X(x) = Gamma(d/2) / (sqrt(pi) * Gamma((d-1)/2)) * (1 - x^2)^{(d-3)/2}
    for x in [-1, 1].

    Uses log-gamma to avoid numerical overflow for large d.
    """
    log_coeff = gammaln(d / 2.0) - 0.5 * np.log(np.pi) - gammaln((d - 1) / 2.0)
    # For large d, (1 - x^2)^{(d-3)/2} underflows for |x| not very close to 0.
    # Use log space: (d-3)/2 * log(1 - x^2)
    one_minus_x2 = np.maximum(1.0 - x ** 2, 1e-300)
    log_body = (d - 3) / 2.0 * np.log(one_minus_x2)
    return np.exp(log_coeff + log_body)


# ──────────────────────────────────────────────────────────────────────
# 2. Lloyd-Max optimal scalar quantizer for the Beta distribution
# ──────────────────────────────────────────────────────────────────────

def lloyd_max_codebook(d: int, b: int, max_iter: int = 200, n_grid: int = 50000) -> np.ndarray:
    """
    Compute the optimal Lloyd-Max codebook (centroids) for b-bit quantization
    of the Beta distribution induced by dimension d on the unit hypersphere.

    Uses a fine grid approximation of the continuous Lloyd-Max algorithm.

    Returns:
        centroids: np.ndarray of shape (2^b,), sorted in ascending order.
    """
    n_levels = 2 ** b

    # The effective support of the Beta distribution is [-1, 1] but for large d
    # the density concentrates around 0 with scale ~1/sqrt(d).
    # Use a grid that covers the effective support well.
    sigma = 1.0 / np.sqrt(d) if d >= 10 else 0.9
    lo, hi = max(-1 + 1e-10, -6 * sigma), min(1 - 1e-10, 6 * sigma)

    xs = np.linspace(lo, hi, n_grid)
    pdf = beta_pdf(xs, d)
    pdf /= pdf.sum()  # normalize to a discrete distribution

    # Initialize centroids uniformly in the effective range
    centroids = np.linspace(lo, hi, n_levels + 2)[1:-1]  # skip endpoints

    for _ in range(max_iter):
        # Assignment: each grid point goes to nearest centroid
        dists = np.abs(xs[:, None] - centroids[None, :])  # (n_grid, n_levels)
        assignments = dists.argmin(axis=1)  # (n_grid,)

        # Update centroids: weighted mean
        new_centroids = np.zeros(n_levels)
        for k in range(n_levels):
            mask = assignments == k
            w = pdf[mask]
            if w.sum() > 0:
                new_centroids[k] = (xs[mask] * w).sum() / w.sum()
            else:
                new_centroids[k] = centroids[k]

        if np.allclose(centroids, new_centroids, atol=1e-10):
            break
        centroids = new_centroids

    centroids.sort()
    return centroids


def compute_mse_cost(d: int, b: int, centroids: np.ndarray, n_grid: int = 50000) -> float:
    """
    Compute the MSE cost C(f_X, b) for the given codebook.
    C = sum_i integral over Voronoi cell i of |x - c_i|^2 f_X(x) dx
    """
    sigma = 1.0 / np.sqrt(d) if d >= 10 else 0.9
    lo, hi = max(-1 + 1e-10, -6 * sigma), min(1 - 1e-10, 6 * sigma)

    xs = np.linspace(lo, hi, n_grid)
    dx = xs[1] - xs[0]
    pdf = beta_pdf(xs, d)

    dists = np.abs(xs[:, None] - centroids[None, :])
    assignments = dists.argmin(axis=1)
    nearest_centroid = centroids[assignments]

    cost = np.sum((xs - nearest_centroid) ** 2 * pdf) * dx
    return cost


# ──────────────────────────────────────────────────────────────────────
# 3. Codebook cache — precompute for common (d, b) pairs
# ──────────────────────────────────────────────────────────────────────

_codebook_cache = {}


def get_codebook(d: int, b: int) -> np.ndarray:
    """Get (or compute and cache) the Lloyd-Max codebook for dimension d and bit-width b."""
    key = (d, b)
    if key not in _codebook_cache:
        _codebook_cache[key] = lloyd_max_codebook(d, b)
    return _codebook_cache[key]


# ──────────────────────────────────────────────────────────────────────
# 4. Random rotation matrix via QR decomposition
# ──────────────────────────────────────────────────────────────────────

def random_rotation_matrix(d: int, device: torch.device = None, dtype=torch.float32) -> torch.Tensor:
    """
    Generate a random orthogonal (rotation) matrix Π ∈ R^{d×d}
    via QR decomposition of a random Gaussian matrix.
    """
    if device is None:
        device = torch.device('cpu')
    G = torch.randn(d, d, device=device, dtype=dtype)
    Q, R = torch.linalg.qr(G)
    # Ensure det(Q) = +1 (proper rotation)
    diag_sign = torch.sign(torch.diag(R))
    Q = Q * diag_sign.unsqueeze(0)
    return Q


# ──────────────────────────────────────────────────────────────────────
# 5. TurboQuant_mse — Algorithm 1
# ──────────────────────────────────────────────────────────────────────

class TurboQuantMSE:
    """
    MSE-optimized TurboQuant (Algorithm 1 from the paper).

    Steps:
      Setup: Generate random rotation Π, precompute codebook centroids.
      Quant(x): y = Π·x, idx_j = argmin_k |y_j - c_k| for each coord j.
      DeQuant(idx): ỹ_j = c_{idx_j}, x̃ = Π^T · ỹ.
    """

    def __init__(self, d: int, b: int, device: torch.device = None, seed: int = None):
        """
        Args:
            d: vector dimension
            b: bit-width per coordinate
            device: torch device
            seed: random seed for reproducibility
        """
        self.d = d
        self.b = b
        self.n_levels = 2 ** b
        self.device = device or torch.device('cpu')

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Random rotation matrix
        self.Pi = random_rotation_matrix(d, device=self.device)

        # Precompute Lloyd-Max codebook centroids
        centroids_np = get_codebook(d, b)
        self.centroids = torch.tensor(centroids_np, device=self.device, dtype=torch.float32)

    def quantize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize input vectors.

        Args:
            x: (n, d) tensor of input vectors (not necessarily unit-norm)

        Returns:
            idx: (n, d) integer tensor of codebook indices (b-bit integers)
            norms: (n,) tensor of L2 norms (stored for rescaling)
        """
        # Store norms for rescaling
        norms = torch.norm(x, dim=-1, keepdim=True)
        x_unit = x / (norms + 1e-12)

        # Rotate: y = Π · x
        y = x_unit @ self.Pi.T  # (n, d)

        # Find nearest centroid per coordinate
        # dists: (n, d, n_levels)
        dists = torch.abs(y.unsqueeze(-1) - self.centroids.unsqueeze(0).unsqueeze(0))
        idx = dists.argmin(dim=-1)  # (n, d)

        return idx, norms.squeeze(-1)

    def dequantize(self, idx: torch.Tensor, norms: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct vectors from quantized indices.

        Args:
            idx: (n, d) integer tensor of codebook indices
            norms: (n,) tensor of L2 norms

        Returns:
            x_hat: (n, d) reconstructed vectors
        """
        # Map indices to centroids
        y_hat = self.centroids[idx]  # (n, d)

        # Inverse rotate: x̃ = Π^T · ỹ
        x_hat = y_hat @ self.Pi  # (n, d)

        # Rescale
        x_hat = x_hat * norms.unsqueeze(-1)

        return x_hat

    def quantize_dequantize(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize and immediately dequantize (for evaluation)."""
        idx, norms = self.quantize(x)
        return self.dequantize(idx, norms)


# ──────────────────────────────────────────────────────────────────────
# 6. QJL — Quantized Johnson-Lindenstrauss transform (1-bit)
# ──────────────────────────────────────────────────────────────────────

class QJL:
    """
    1-bit Quantized Johnson-Lindenstrauss (QJL) transform.

    Q_qjl(x) = sign(S · x) where S ~ N(0,1)^{d×d}
    Q_qjl^{-1}(z) = sqrt(π/2)/d · S^T · z

    Provides unbiased inner product estimation with variance ≤ π/(2d) · ||y||^2.
    """

    def __init__(self, d: int, device: torch.device = None, seed: int = None):
        self.d = d
        self.device = device or torch.device('cpu')

        if seed is not None:
            torch.manual_seed(seed + 12345)

        # Random projection matrix S ∈ R^{d×d}, S_{i,j} ~ N(0,1)
        self.S = torch.randn(d, d, device=self.device)

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply QJL: sign(S · x)

        Args:
            x: (n, d) tensor

        Returns:
            qjl: (n, d) tensor of {-1, +1}
        """
        projected = x @ self.S.T  # (n, d)
        return torch.sign(projected)

    def dequantize(self, qjl: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        """
        Inverse QJL: sqrt(π/2)/d · γ · S^T · qjl

        Args:
            qjl: (n, d) tensor of {-1, +1}
            gamma: (n,) tensor of residual norms ||r||

        Returns:
            x_hat: (n, d) reconstructed residual contribution
        """
        coeff = math.sqrt(math.pi / 2) / self.d
        reconstructed = coeff * (qjl @ self.S)  # (n, d)
        return reconstructed * gamma.unsqueeze(-1)


# ──────────────────────────────────────────────────────────────────────
# 7. TurboQuant_prod — Algorithm 2
# ──────────────────────────────────────────────────────────────────────

class TurboQuantProd:
    """
    Inner-product optimized TurboQuant (Algorithm 2 from the paper).

    Two-stage approach:
      1. Apply TurboQuant_mse with bit-width b-1
      2. Apply QJL on the residual (1-bit per coordinate)
      Total: b bits per coordinate

    Guarantees unbiased inner product estimation.
    """

    def __init__(self, d: int, b: int, device: torch.device = None, seed: int = None):
        """
        Args:
            d: vector dimension
            b: total bit-width per coordinate (must be >= 1)
            device: torch device
            seed: random seed
        """
        assert b >= 1, "TurboQuant_prod requires b >= 1"
        self.d = d
        self.b = b
        self.device = device or torch.device('cpu')

        seed = seed or 42

        # Stage 1: MSE quantizer with b-1 bits
        self.mse_bits = b - 1
        if self.mse_bits > 0:
            self.mse_quantizer = TurboQuantMSE(d, self.mse_bits, device=self.device, seed=seed)
        else:
            self.mse_quantizer = None

        # Stage 2: QJL (1-bit per coordinate)
        self.qjl = QJL(d, device=self.device, seed=seed)

    def quantize(self, x: torch.Tensor) -> dict:
        """
        Quantize for unbiased inner product estimation.

        Args:
            x: (n, d) tensor of input vectors

        Returns:
            dict with keys:
                'idx': (n, d) MSE quantization indices (or None if b=1)
                'norms': (n,) original vector norms
                'qjl': (n, d) QJL sign bits {-1, +1}
                'gamma': (n,) residual norms ||r||
        """
        norms = torch.norm(x, dim=-1, keepdim=True)
        x_unit = x / (norms + 1e-12)

        if self.mse_quantizer is not None:
            # Stage 1: MSE quantization with b-1 bits
            idx, _ = self.mse_quantizer.quantize(x_unit)
            x_mse = self.mse_quantizer.dequantize(idx, torch.ones(x.shape[0], device=self.device))

            # Residual
            r = x_unit - x_mse  # (n, d)
        else:
            # b=1: no MSE stage, entire vector goes to QJL
            idx = None
            r = x_unit

        # Stage 2: QJL on residual
        gamma = torch.norm(r, dim=-1)  # (n,)
        r_unit = r / (gamma.unsqueeze(-1) + 1e-12)
        qjl = self.qjl.quantize(r_unit)

        return {
            'idx': idx,
            'norms': norms.squeeze(-1),
            'qjl': qjl,
            'gamma': gamma,
        }

    def dequantize(self, quant_data: dict) -> torch.Tensor:
        """
        Reconstruct vectors from quantized representation.

        Args:
            quant_data: dict from quantize()

        Returns:
            x_hat: (n, d) reconstructed vectors
        """
        n = quant_data['norms'].shape[0]

        if self.mse_quantizer is not None:
            x_mse = self.mse_quantizer.dequantize(
                quant_data['idx'],
                torch.ones(n, device=self.device)
            )
        else:
            x_mse = torch.zeros(n, self.d, device=self.device)

        # QJL reconstruction
        x_qjl = self.qjl.dequantize(quant_data['qjl'], quant_data['gamma'])

        # Combine and rescale
        x_hat = (x_mse + x_qjl) * quant_data['norms'].unsqueeze(-1)

        return x_hat

    def quantize_dequantize(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize and immediately dequantize (for evaluation)."""
        quant_data = self.quantize(x)
        return self.dequantize(quant_data)


# ──────────────────────────────────────────────────────────────────────
# 8. Utility functions
# ──────────────────────────────────────────────────────────────────────

def compute_mse_distortion(x: torch.Tensor, x_hat: torch.Tensor) -> float:
    """Compute mean-squared error: E[||x - x_hat||^2] averaged over samples."""
    return ((x - x_hat) ** 2).sum(dim=-1).mean().item()


def compute_inner_product_distortion(x: torch.Tensor, x_hat: torch.Tensor,
                                      y: torch.Tensor) -> Tuple[float, float]:
    """
    Compute inner product distortion:
      bias = |E[<y, x_hat>] - E[<y, x>]|
      variance = E[|<y, x> - <y, x_hat>|^2]

    Args:
        x: (n, d) original vectors
        x_hat: (n, d) reconstructed vectors
        y: (m, d) query vectors

    Returns:
        (bias, variance) tuple
    """
    # x: (n, d), y: (m, d)
    # ip_orig: (m, n), ip_hat: (m, n)
    ip_orig = y @ x.T
    ip_hat = y @ x_hat.T

    diff = ip_orig - ip_hat
    bias = diff.mean().abs().item()
    variance = (diff ** 2).mean().item()

    return bias, variance


def theoretical_mse_upper_bound(b: int) -> float:
    """Theoretical MSE upper bound: sqrt(3*pi)/2 * 1/4^b"""
    return math.sqrt(3 * math.pi) / 2 * (1 / 4 ** b)


def theoretical_mse_lower_bound(b: int) -> float:
    """Theoretical MSE lower bound: 1/4^b"""
    return 1.0 / (4 ** b)


def theoretical_ip_upper_bound(b: int, d: int) -> float:
    """Theoretical inner-product distortion upper bound: sqrt(3)*pi^2/d * 1/4^b"""
    return math.sqrt(3) * math.pi ** 2 / d * (1 / 4 ** b)


def theoretical_ip_lower_bound(b: int, d: int) -> float:
    """Theoretical inner-product distortion lower bound: 1/d * 1/4^b"""
    return 1.0 / d * (1 / 4 ** b)


# Refined MSE bounds for small bit-widths (from paper Table in Theorem 1)
REFINED_MSE_BOUNDS = {1: 0.36, 2: 0.117, 3: 0.03, 4: 0.009}

# Refined IP bounds for small bit-widths (from paper, per dimension)
REFINED_IP_BOUNDS = {1: 1.57, 2: 0.56, 3: 0.18, 4: 0.047}  # multiply by 1/d


if __name__ == "__main__":
    # Quick smoke test
    d, n = 256, 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}, d={d}, n={n}")

    # Generate random unit vectors
    x = torch.randn(n, d, device=device)
    x = x / torch.norm(x, dim=-1, keepdim=True)

    for b in [1, 2, 3, 4]:
        # MSE quantizer
        tq_mse = TurboQuantMSE(d, b, device=device, seed=0)
        x_hat_mse = tq_mse.quantize_dequantize(x)
        mse = compute_mse_distortion(x, x_hat_mse)

        # Prod quantizer
        tq_prod = TurboQuantProd(d, b, device=device, seed=0)
        x_hat_prod = tq_prod.quantize_dequantize(x)
        mse_prod = compute_mse_distortion(x, x_hat_prod)

        # Query vectors for IP test
        y = torch.randn(50, d, device=device)
        y = y / torch.norm(y, dim=-1, keepdim=True)
        bias_mse, var_mse = compute_inner_product_distortion(x, x_hat_mse, y)
        bias_prod, var_prod = compute_inner_product_distortion(x, x_hat_prod, y)

        ub = theoretical_mse_upper_bound(b)
        lb = theoretical_mse_lower_bound(b)
        print(f"\nb={b}: MSE_mse={mse:.6f}, MSE_prod={mse_prod:.6f}")
        print(f"  Theoretical bounds: [{lb:.6f}, {ub:.6f}]")
        print(f"  IP bias  — MSE-quant: {bias_mse:.6f}, Prod-quant: {bias_prod:.6f}")
        print(f"  IP var   — MSE-quant: {var_mse:.6f}, Prod-quant: {var_prod:.6f}")
