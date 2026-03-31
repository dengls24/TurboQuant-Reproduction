# TurboQuant Reproduction

Reproduction of experiments from **"TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate"** (ICLR 2026, [arXiv:2504.19874](https://arxiv.org/abs/2504.19874)).

This repository also includes a technical comparison with **RaBitQ** ([SIGMOD 2024](https://arxiv.org/abs/2405.12497)), which shares the same core mathematical framework (random orthogonal rotation + scalar quantization).

## Key Results

All core claims from the paper are **verified**:

| Paper Claim | Our Result | Status |
|---|---|---|
| MSE distortion within theoretical bounds | Matches Theorem 1 bounds | Verified |
| TurboQuant_prod gives unbiased IP estimation | \|mean_err\| < 0.001 across all conditions | Verified |
| KV cache quantization preserves NIAH performance | **66/66 test points 100% match** (4K-128K tokens) | Verified |
| TurboQuant ~1000x faster than Product Quantization | 900-1000x speedup observed | Verified |
| TurboQuant Recall@k >= PQ | TQ > PQ in all configurations | Verified |

### NIAH (Needle-In-A-Haystack) Results

Tested on **Meta-Llama-3.1-8B-Instruct** across 4K-128K tokens using 4×A800-80GB GPUs:

```
Full Precision Score:      0.4508
TurboQuant 3.5-bit Score:  0.4508  (identical)
```

## Project Structure

```
├── turboquant.py                    # Core algorithms (Algorithm 1 & 2 from paper)
├── experiments/
│   ├── kv_cache_quant.py            # KV cache quantization module (multi-GPU)
│   ├── exp_empirical_validation.py  # Section 4.1: Figure 1 & 3
│   ├── exp_figure2_ip_vs_avgip.py   # Section 4.1: Figure 2
│   ├── exp_niah.py                  # Section 4.2: NIAH test (single GPU)
│   ├── exp_niah_multigpu.py         # Section 4.2: NIAH test (multi-GPU, 4K-128K)
│   ├── exp_longbench.py             # Section 4.3: Generation quality
│   ├── exp_nn_search.py             # Section 4.4: NN search comparison
│   ├── 实验报告.md                    # Detailed report (Chinese)
│   └── results/                     # Plots and numerical results
├── requirements.txt
└── README.md
```

## Core Algorithms

### TurboQuant_mse (Algorithm 1)
MSE-optimized quantizer: random orthogonal rotation → per-coordinate Lloyd-Max quantization → inverse rotation.

### TurboQuant_prod (Algorithm 2)
Unbiased inner-product quantizer: (b-1)-bit TurboQuant_mse + 1-bit QJL residual correction.

### KV Cache Quantization
Mixed-precision strategy: outlier channels get higher bits, regular channels get lower bits. Example: 32ch@5bit + 96ch@3bit = 3.5-bit effective on 128-dim heads.

## Quick Start

```bash
pip install -r requirements.txt

# Run quantization error validation (Section 4.1)
python experiments/exp_empirical_validation.py

# Run NIAH test with multi-GPU (Section 4.2)
CUDA_VISIBLE_DEVICES=0,1,2,3 python experiments/exp_niah_multigpu.py

# Run NN search comparison (Section 4.4)
python experiments/exp_nn_search.py
```

## Requirements

- Python 3.10+
- PyTorch 2.0+ with CUDA
- 1+ NVIDIA GPU (multi-GPU recommended for long-context experiments)
- ~15GB disk for Llama-3.1-8B-Instruct model

## Experimental Setup

| Experiment | Model | Hardware | Context |
|---|---|---|---|
| Section 4.1 (Quantization Error) | Synthetic data, d=1536 | 1×GPU | N/A |
| Section 4.2 (NIAH) | Llama-3.1-8B-Instruct | 4×A800-80GB | 4K-128K |
| Section 4.3 (Generation) | Qwen3-8B | 1×GPU | Short+Long QA |
| Section 4.4 (NN Search) | Synthetic data | 1×GPU | N/A |

## References

- **TurboQuant**: Zandieh et al., "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate", ICLR 2026. [arXiv:2504.19874](https://arxiv.org/abs/2504.19874)
- **RaBitQ**: Gao & Long, "RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound for Approximate Nearest Neighbor Search", SIGMOD 2024. [arXiv:2405.12497](https://arxiv.org/abs/2405.12497)

## License

MIT
