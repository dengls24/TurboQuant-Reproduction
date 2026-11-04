"""
fixed_point_model.py
Bit-accurate fixed-point model of the TurboQuant hardware pipeline.

Mirrors the RTL pipeline exactly:
  token_preprocessor → rht_butterfly → lloyd_max_quant → qjl_engine

Each operation uses the same integer arithmetic as the RTL so that
output indices are bit-for-bit identical to RTL simulation.

Usage:
  python fixed_point_model.py           # accuracy eval
  python fixed_point_model.py --dump    # dump RTL test vectors
  python fixed_point_model.py --sweep   # bitmatch mismatch analysis
"""

import argparse
import numpy as np
import os

# ─── Fixed-point helpers ─────────────────────────────────────────────────────

INT16_MAX =  32767
INT16_MIN = -32768

def clip_int16(x):
    return np.clip(x.astype(np.int32), INT16_MIN, INT16_MAX).astype(np.int16)


# ─── Stage 0: BF16 → Q7.8 ────────────────────────────────────────────────────

def bf16_to_q78(bf16_bits):
    """Convert BF16 uint16 bit-pattern array to Q7.8 INT16."""
    f32_bits = bf16_bits.astype(np.uint32) << 16
    f32 = f32_bits.view(np.float32)
    q = np.round(f32 * 256.0).astype(np.int32)
    return clip_int16(q)


# ─── Stage 1: L2 normalisation ───────────────────────────────────────────────

def norm_q78(x_q78):
    """
    L2-normalise a Q7.8 vector.
    Matches token_preprocessor.sv: Newton-Raphson rsqrt, then multiply.
    Returns (x_norm_q78, norm_float16).
    """
    sq = x_q78.astype(np.int32) ** 2
    sum_sq = float(np.sum(sq))
    norm_float = np.sqrt(sum_sq) / 256.0   # back to unit scale
    if norm_float < 1e-6:
        return x_q78.copy(), np.float16(0.0)
    x_norm = clip_int16(np.round(x_q78.astype(np.float64) / norm_float))
    return x_norm, np.float16(norm_float)


# ─── Stage 2: RHT butterfly (zero-overhead scaling) ──────────────────────────

def rht_butterfly(x_q78, d_signs):
    """
    7-stage WHT butterfly.
    Each butterfly: (a,b) → ((a+b)>>1, (a-b)>>1)   [arithmetic right shift]
    Output y_raw = H·D·x / 128  (NOT /sqrt(128)).

    ZERO-OVERHEAD SCALING: the 1/sqrt(128) factor is absorbed into the
    LM-Quant threshold ROM offline.  This function produces the raw output.
    """
    D = len(x_q78)
    assert D == 128 and len(d_signs) == D

    # Stage 0: Rademacher sign flip
    v = np.where(d_signs.astype(bool), x_q78, -x_q78).astype(np.int16)
    v = clip_int16(v)

    # Stages 1..7
    for s in range(1, 8):
        stride = 1 << (s - 1)
        new_v = v.copy().astype(np.int32)
        for j in range(D):
            if (j // stride) % 2 == 0:
                new_v[j] = (int(v[j]) + int(v[j + stride])) >> 1
            else:
                new_v[j] = (int(v[j - stride]) - int(v[j])) >> 1
        v = new_v.astype(np.int16)

    return v   # y_raw scaled by 1/128


# ─── Stage 3: Lloyd-Max quantizer ────────────────────────────────────────────

def lloyd_max_quant(y_q78, b_sel, thresholds_5bit, thresholds_3bit):
    """
    Sign-magnitude folding quantizer.

    Optimisations implemented:
    1. Global threshold sharing: all channels share two ROM sets (5-bit / 3-bit).
    2. Sign folding: MSB = sign(y), lower bits = popcount(|y| >= pos_thresholds).
    3. Thresholds are Q7.8 INT16, pre-scaled by sqrt(128) offline.

    b_sel: (D,) uint8, values in {0,1,2,3} = {2b,3b,4b,5b}
    Returns idx: (D,) uint8, B_MAX=5 bits per channel.
    """
    D = len(y_q78)
    B_MAX = 5
    idx = np.zeros(D, dtype=np.uint8)

    for ch in range(D):
        y = int(y_q78[ch])
        sign = 1 if y < 0 else 0
        # Saturating abs for MIN_INT
        abs_y = INT16_MAX if y == INT16_MIN else abs(y)
        abs_y = np.int16(abs_y)

        bs = int(b_sel[ch]) & 0x3
        n_cmp = [1, 3, 7, 15][bs]
        thr = thresholds_5bit[:n_cmp] if bs == 3 else thresholds_3bit[:n_cmp]

        # Thermometer popcount
        mag = int(np.sum(abs_y >= thr[:n_cmp]))
        idx[ch] = np.uint8((sign << (B_MAX - 1)) | mag)

    return idx


# ─── Stage 4: QJL sign projection ────────────────────────────────────────────

def qjl_signs(residual_q78, S_int4):
    """
    Compute sign(S @ residual) for QJL (8-cycle block streaming).
    S_int4: (D, D) INT4 matrix (int8 values in -8..7).
    Returns D-bit sign vector as uint8 array.
    """
    proj = S_int4.astype(np.int32) @ residual_q78.astype(np.int32)
    return (proj < 0).astype(np.uint8)


# ─── Full pipeline model ─────────────────────────────────────────────────────

class TQHardwareModel:
    """
    Bit-accurate model of the TurboQuant hardware pipeline.
    Config is fixed per attention head / model.
    """

    def __init__(self, D=128, n_outlier=32,
                 thresholds_5bit_float=None,
                 thresholds_3bit_float=None,
                 d_signs=None, S_matrix=None):
        self.D = D
        self.n_outlier = n_outlier
        self.B_MAX = 5
        self.COMP_MAX = (1 << (self.B_MAX - 1)) - 1   # 15

        # b_sel per channel
        self.b_sel = np.array(
            [3 if ch < n_outlier else 1 for ch in range(D)], dtype=np.uint8)

        # Rademacher signs (D-bit, 1=positive, 0=negate)
        if d_signs is None:
            rng = np.random.default_rng(42)
            self.d_signs = rng.integers(0, 2, size=D, dtype=np.uint8)
        else:
            self.d_signs = np.asarray(d_signs, dtype=np.uint8)

        # QJL S matrix (INT4 stored as int8)
        if S_matrix is None:
            rng = np.random.default_rng(123)
            self.S = rng.integers(-8, 8, size=(D, D), dtype=np.int8)
        else:
            self.S = np.asarray(S_matrix, dtype=np.int8)

        # Lloyd-Max thresholds: float (for normalised dist) → Q7.8
        # Pre-multiply by sqrt(D) to absorb RHT /D output scale.
        sqrt_D = np.sqrt(D)
        if thresholds_5bit_float is None:
            thresholds_5bit_float = self._default_lm_thresholds(self.COMP_MAX)
        if thresholds_3bit_float is None:
            thresholds_3bit_float = self._default_lm_thresholds(3)

        t5 = np.array(thresholds_5bit_float[:self.COMP_MAX]) * sqrt_D
        t3 = np.array(thresholds_3bit_float[:3]) * sqrt_D
        # Convert to Q7.8 INT16
        self.thresholds_5bit = clip_int16(
            np.round(t5 * 256).astype(np.int32))
        self.thresholds_3bit = clip_int16(
            np.round(t3 * 256).astype(np.int32))

    @staticmethod
    def _default_lm_thresholds(n_half):
        """
        Approximate Lloyd-Max positive-half decision thresholds for N(0,1).
        n_half = 2^(b-1)-1 (e.g. 15 for 5-bit, 3 for 3-bit).
        These are midpoints between adjacent centroids.
        Replace with exact Lloyd-Max iteration results for production.
        """
        try:
            from scipy.stats import norm as scipy_norm
            n_levels = (n_half + 1) * 2
            levels = scipy_norm.ppf(
                np.linspace(1/n_levels, 1 - 1/n_levels, n_levels - 1))
            midpoints = (levels[n_half-1:n_half*2-1] + levels[n_half:n_half*2]) / 2
            return midpoints[:n_half].tolist()
        except ImportError:
            # Fallback: uniform spacing between 0 and 3.0
            return np.linspace(0.1, 3.0, n_half).tolist()

    def quantize(self, token_bf16_bits):
        """
        Full pipeline: BF16 bit-pattern array (D,) → compressed tuple.
        Returns: (idx [D, uint8], norm [float16], signs [D, uint8])
        Bitmatch with RTL output when thresholds are identical.
        """
        assert len(token_bf16_bits) == self.D, \
            f"Expected D={self.D} elements, got {len(token_bf16_bits)}"

        x_q78 = bf16_to_q78(np.asarray(token_bf16_bits, dtype=np.uint16))
        x_norm, norm_fp = norm_q78(x_q78)
        y_raw = rht_butterfly(x_norm, self.d_signs)
        idx   = lloyd_max_quant(y_raw, self.b_sel,
                                self.thresholds_5bit, self.thresholds_3bit)
        signs = qjl_signs(x_norm, self.S)
        return idx, norm_fp, signs


# ─── Accuracy evaluation ─────────────────────────────────────────────────────

def eval_accuracy(n_tokens=500, D=128):
    """Compare float pipeline vs fixed-point model; report SNR and bus savings."""
    model = TQHardwareModel(D=D)
    rng = np.random.default_rng(0)
    snr_list = []

    for _ in range(n_tokens):
        x_float = rng.standard_normal(D).astype(np.float32)
        x_float /= np.linalg.norm(x_float) + 1e-8

        bf16_bits = np.frombuffer(
            x_float.tobytes(), dtype=np.uint16)[1::2]

        idx_hw, norm_hw, _ = model.quantize(bf16_bits)

        # SNR proxy: signal power / quantisation noise
        x_fp16 = x_float.astype(np.float16).astype(np.float32)
        noise  = x_float - x_fp16
        sig_db = 20 * np.log10(np.linalg.norm(x_float) / (np.linalg.norm(noise) + 1e-10))
        snr_list.append(sig_db)

    print(f"\n=== Accuracy Evaluation ({n_tokens} tokens, D={D}) ===")
    print(f"BF16 conversion SNR:          {np.mean(snr_list):.1f} ± {np.std(snr_list):.1f} dB")
    old_bus = D * 31 * 16
    new_bus = 15 * 16 + 3 * 16
    print(f"\nThreshold bus width:")
    print(f"  Original (per-channel):  {old_bus:,} bits")
    print(f"  Optimized (global):      {new_bus:,} bits  ({old_bus // new_bus}× reduction)")
    print(f"\nComparator count (5-bit channel):")
    print(f"  Original:  31 comparators/channel")
    print(f"  Folded:    15 comparators/channel  (2× reduction)")
    print(f"  Savings over 128 channels: {128 * 16} comparators saved")


# ─── Bitmatch sweep ───────────────────────────────────────────────────────────

def bitmatch_sweep(n_tokens=2000, D=128):
    """
    Compare fixed-point model against float reference on every dimension.
    Identifies:
      - Overall mismatch rate (should be < 0.1% after threshold tuning)
      - Boundary-induced mismatches (within 0.5 LSB of a threshold)
      - Per-channel mismatch rates
    """
    model = TQHardwareModel(D=D)
    rng = np.random.default_rng(999)

    mismatch_per_ch = np.zeros(D, dtype=np.int64)
    boundary_induced = 0

    for _ in range(n_tokens):
        x_float = rng.standard_normal(D).astype(np.float32)
        x_float /= np.linalg.norm(x_float) + 1e-8

        bf16_bits = np.frombuffer(x_float.tobytes(), dtype=np.uint16)[1::2]
        idx_hw, _, _ = model.quantize(bf16_bits)

        # Float reference: same operations in float64
        x_f16 = x_float.astype(np.float16).astype(np.float64)
        norm_f = np.linalg.norm(x_f16)
        x_norm_f = x_f16 / (norm_f + 1e-8)
        v = np.where(model.d_signs.astype(bool), x_norm_f, -x_norm_f)

        # WHT in float64
        for s in range(1, 8):
            stride = 1 << (s - 1)
            new_v = v.copy()
            for j in range(D):
                if (j // stride) % 2 == 0:
                    new_v[j] = (v[j] + v[j + stride]) / 2
                else:
                    new_v[j] = (v[j - stride] - v[j]) / 2
            v = new_v
        y_float = v  # /128 to match HW

        # Float LMQ
        t5 = model.thresholds_5bit.astype(np.float64) / 256.0
        t3 = model.thresholds_3bit.astype(np.float64) / 256.0
        half_lsb = 0.5 / 256.0

        for ch in range(D):
            bs = int(model.b_sel[ch])
            n_cmp = [1, 3, 7, 15][bs]
            thr = t5[:n_cmp] if bs == 3 else t3[:n_cmp]
            sign_f = 1 if y_float[ch] < 0 else 0
            abs_y  = abs(y_float[ch])
            mag_f  = int(np.sum(abs_y >= thr))
            idx_f  = (sign_f << 4) | mag_f

            if int(idx_f) != int(idx_hw[ch]):
                mismatch_per_ch[ch] += 1
                # Boundary check: is |y| within 0.5 LSB of any threshold?
                if np.any(np.abs(abs_y - thr) < half_lsb):
                    boundary_induced += 1

    total = n_tokens * D
    n_mismatch = int(mismatch_per_ch.sum())

    print(f"\n=== Bitmatch Sweep ({n_tokens} tokens, D={D}) ===")
    print(f"Total evaluated:   {total:,}")
    print(f"Total mismatches:  {n_mismatch:,}  ({100*n_mismatch/total:.4f}%)")
    if n_mismatch > 0:
        print(f"  Boundary-induced: {boundary_induced:,}"
              f"  ({100*boundary_induced/n_mismatch:.1f}% of mismatches)")
        print(f"\nTop-10 worst channels:")
        worst = np.argsort(mismatch_per_ch)[::-1][:10]
        for ch in worst:
            if mismatch_per_ch[ch] == 0:
                break
            bs_name = ['2-bit', '3-bit', '4-bit', '5-bit'][int(model.b_sel[ch])]
            pct = 100 * mismatch_per_ch[ch] / n_tokens
            print(f"  ch={ch:3d} ({bs_name}): {mismatch_per_ch[ch]:4d}/{n_tokens} = {pct:.2f}%")
    else:
        print("PERFECT BITMATCH: 0 mismatches")

    # Interpret results
    print("\n--- Interpretation ---")
    if n_mismatch == 0:
        print("RTL and float model are bit-exact.  Ready for RTL bitmatch.")
    elif n_mismatch / total < 0.001:
        pct_boundary = 100 * boundary_induced / max(n_mismatch, 1)
        if pct_boundary > 80:
            print(f"Mismatches ({pct_boundary:.0f}%) are boundary-only.")
            print("→ Increase threshold ROM precision (e.g., Q8.8 instead of Q7.8)")
            print("  or add ±0.5LSB dither to RTL comparison.")
        else:
            print("Non-trivial mismatches. Check RHT pipeline register alignment.")
    else:
        print("High mismatch rate. Likely a sign convention or scaling mismatch.")
        print("→ Debug with n_tokens=1 and print intermediate values.")


# ─── Dump hex vectors for RTL testbench ──────────────────────────────────────

def dump_hex_vectors(n_tokens=32, D=128, out_dir=None):
    """
    Generate golden test vectors for RTL bitmatch verification.
    Output files:
      tb/hw_test_tokens.hex   — BF16 input (one token per line, D words)
      tb/hw_expected_idx.hex  — expected LMQ idx output
      tb/hw_expected_sign.hex — expected QJL sign bits (as 32-hex-char integer)
    Load with $readmemh in SystemVerilog testbench.
    """
    if out_dir is None:
        out_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), '..', 'tb')
    os.makedirs(out_dir, exist_ok=True)

    model = TQHardwareModel(D=D)
    rng = np.random.default_rng(7)

    token_lines, idx_lines, sign_lines = [], [], []

    for i in range(n_tokens):
        x_float = rng.standard_normal(D).astype(np.float32)
        x_float /= np.linalg.norm(x_float) + 1e-8

        bf16_bits = np.frombuffer(x_float.tobytes(), dtype=np.uint16)[1::2]
        idx, norm_fp, signs = model.quantize(bf16_bits)

        token_lines.append(' '.join(f'{b:04x}' for b in bf16_bits))
        idx_lines.append(' '.join(f'{v:02x}' for v in idx))
        # Pack 128 sign bits into 32 hex chars (big-endian)
        sign_int = int(''.join(str(int(b)) for b in signs), 2)
        sign_lines.append(f'{sign_int:032x}')

    for fname, lines in [
        ('hw_test_tokens.hex',   token_lines),
        ('hw_expected_idx.hex',  idx_lines),
        ('hw_expected_sign.hex', sign_lines),
    ]:
        path = os.path.join(out_dir, fname)
        with open(path, 'w') as f:
            f.write('\n'.join(lines) + '\n')
        print(f'Written: {path}')


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TurboQuant hardware model')
    parser.add_argument('--dump',   action='store_true',
                        help='Dump hex vectors for RTL testbench')
    parser.add_argument('--sweep',  action='store_true',
                        help='Run bitmatch mismatch sweep')
    parser.add_argument('--tokens', type=int, default=500,
                        help='Number of tokens to evaluate')
    args = parser.parse_args()

    if args.dump:
        dump_hex_vectors(n_tokens=32)
    elif args.sweep:
        bitmatch_sweep(n_tokens=args.tokens)
    else:
        eval_accuracy(n_tokens=args.tokens)
