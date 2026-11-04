"""
Generate test vectors for tq_engine_tb.sv
Runs Python TurboQuant (golden model) and outputs BF16 hex files

Usage: python hardware/rtl/gen_test_vectors.py
Output: hardware/tb/test_vectors.hex, hardware/tb/golden_output.hex
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import numpy as np
import struct

def float32_to_bf16_int(f):
    """Convert float32 to bfloat16 integer representation."""
    b = struct.pack('>f', f)
    # BF16 = upper 2 bytes of float32
    return (b[0] << 8) | b[1]

def tensor_to_bf16_hex(t):
    """Convert a 1D tensor of float32 to space-separated BF16 hex string."""
    return ' '.join(f'{float32_to_bf16_int(v.item()):04x}' for v in t)

def main():
    os.makedirs('hardware/tb', exist_ok=True)

    D = 128
    N_VECTORS = 32
    B = 3      # regular bits (for golden reference)
    seed = 42

    torch.manual_seed(seed)

    # Generate random unit vectors (simulating KV cache tokens)
    x = torch.randn(N_VECTORS, D)
    norms = x.norm(dim=-1, keepdim=True)
    x_unit = x / norms

    # Scale to typical KV cache magnitude range [~0.1, ~5.0]
    scales = torch.rand(N_VECTORS, 1) * 4.9 + 0.1
    x_scaled = x_unit * scales

    from turboquant import TurboQuantProd
    tq = TurboQuantProd(d=D, b=B, device=torch.device('cpu'), seed=seed)
    x_hat = tq.quantize_dequantize(x_scaled)

    # Write test vectors (BF16 hex)
    with open('hardware/tb/test_vectors.hex', 'w') as f:
        for i in range(N_VECTORS):
            f.write(tensor_to_bf16_hex(x_scaled[i]) + '\n')

    # Write golden output (reconstructed BF16)
    with open('hardware/tb/golden_output.hex', 'w') as f:
        for i in range(N_VECTORS):
            f.write(tensor_to_bf16_hex(x_hat[i]) + '\n')

    # Print statistics
    mse = ((x_scaled - x_hat) ** 2).sum(dim=-1).mean().item()
    print(f"Generated {N_VECTORS} test vectors (D={D}, b={B})")
    print(f"Golden model MSE: {mse:.6f}")
    print(f"Expected MSE range: [{1/4**B:.6f}, {(3.14159**0.5)/2 * 1/4**B:.6f}]")
    print("Saved to hardware/tb/test_vectors.hex and golden_output.hex")

if __name__ == '__main__':
    main()
