"""
Experiment 4.2 (Extended): Multi-GPU Long-Context NIAH with Llama-3.1-8B-Instruct.

Tests KV cache quantization fidelity at context lengths from 4K to 128K tokens,
comparing Full Precision vs TurboQuant at 4-bit / 3.5-bit / 2.5-bit.

Uses device_map='auto' to distribute model across multiple GPUs.
Quantizers are created on the correct device per layer.

Hook implementation: patches DynamicCache.update() — compatible with transformers >= 5.0
(which no longer passes the Cache through attention forward outputs).
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import gc
import time
import json
import math
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from turboquant import TurboQuantProd, TurboQuantMSE

# ─── Config ───
MODEL_CANDIDATES = [
    "/home/xinhuogrp/denglishuo/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3___1-8B-Instruct",
    "/home/xinhuogrp/xushaojie/2_duoattention_orgin/duo-attention/models/Llama-3-8B-Instruct-Gradient-1048k",
]

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# NIAH parameters — full paper range
TOKEN_LIMITS = [4000, 8000, 16000, 32000, 64000, 128000]
DEPTH_PERCENTS = list(range(0, 101, 10))  # 0,10,20,...,100 (11 depths)

NEEDLE = "The best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day."
NEEDLE_QUESTION = "What is the best thing to do in San Francisco?"

HAYSTACK_FILLER = (
    "The history of artificial intelligence began in antiquity, with myths, stories and rumors of "
    "artificial beings endowed with intelligence or consciousness by master craftsmen. Classical philosophers "
    "who attempted to describe the process of human thinking as the mechanical manipulation of symbols laid "
    "the foundation of AI. This led to the invention of the programmable digital computer in the 1940s, "
    "a machine based on the abstract essence of mathematical reasoning. This device and the ideas behind "
    "it inspired a handful of scientists to begin seriously discussing the possibility of building an "
    "electronic brain. The field of AI research was founded at a workshop held on the campus of Dartmouth "
    "College during the summer of 1956. Those who attended would become the leaders of AI research for decades. "
    "Many of them predicted that a machine as intelligent as a human being would exist in no more than a "
    "generation, and they were given millions of dollars to make this vision come true. Eventually it became "
    "obvious that commercial developers and researchers had grossly underestimated the difficulty of the project. "
    "In 1974, in response to the criticism from James Lighthill and ongoing pressure from congress, the U.S. and "
    "British Governments stopped funding undirected research into artificial intelligence, leading to a period "
    "known as the AI winter. Seven years later, a visionary initiative by the Japanese Government inspired "
    "governments and industry to provide AI with billions of dollars, but by the late 1980s the investors became "
    "disillusioned and withdrew funding again. Investment and interest in AI boomed in the first decades of "
    "the 21st century when machine learning was successfully applied to many problems in academia and industry. "
)


# ═══════════════════════════════════════════════════════════════════════
# KV Cache Quantization — DynamicCache.update hook (transformers >= 5.0)
# ═══════════════════════════════════════════════════════════════════════

class TurboQuantKVCacheLocal:
    """KV cache quantizer on a fixed device, with lazy device migration."""

    def __init__(self, head_dim, n_outlier_channels, outlier_bits, regular_bits,
                 device, seed=42, quantizer_type='prod'):
        self.head_dim = head_dim
        self.n_outlier = n_outlier_channels
        self.device = device

        QuantClass = TurboQuantProd if quantizer_type == 'prod' else TurboQuantMSE

        if n_outlier_channels > 0 and outlier_bits > 0:
            self.outlier_quantizer = QuantClass(
                n_outlier_channels, outlier_bits, device=device, seed=seed)
        else:
            self.outlier_quantizer = None

        n_regular = head_dim - n_outlier_channels
        if n_regular > 0 and regular_bits > 0:
            self.regular_quantizer = QuantClass(
                n_regular, regular_bits, device=device, seed=seed + 1)
        else:
            self.regular_quantizer = None

    def move_to(self, device):
        """Lazily migrate all quantizer tensors to target device."""
        if self.device == device:
            return
        self.device = device
        for q in [self.outlier_quantizer, self.regular_quantizer]:
            if q is None:
                continue
            _move_quantizer(q, device)

    def quantize_dequantize(self, kv):
        orig_shape = kv.shape
        head_dim = orig_shape[-1]
        flat = kv.reshape(-1, head_dim).float()

        outlier_indices = torch.arange(self.n_outlier, device=self.device)
        regular_mask = torch.ones(head_dim, dtype=torch.bool, device=self.device)
        regular_mask[outlier_indices] = False
        regular_indices = torch.where(regular_mask)[0]

        result = torch.zeros_like(flat)

        if self.outlier_quantizer is not None and len(outlier_indices) > 0:
            result[:, outlier_indices] = self.outlier_quantizer.quantize_dequantize(
                flat[:, outlier_indices])

        if self.regular_quantizer is not None and len(regular_indices) > 0:
            result[:, regular_indices] = self.regular_quantizer.quantize_dequantize(
                flat[:, regular_indices])

        return result.reshape(orig_shape).to(kv.dtype)


def _move_quantizer(q, device):
    """Move all tensor attributes of a TurboQuant quantizer to device."""
    for attr in ['Pi', 'centroids']:
        if hasattr(q, attr):
            setattr(q, attr, getattr(q, attr).to(device))
    if hasattr(q, 'device'):
        q.device = device
    if hasattr(q, 'qjl') and q.qjl is not None:
        if hasattr(q.qjl, 'S'):
            q.qjl.S = q.qjl.S.to(device)
        q.qjl.device = device
    if hasattr(q, 'mse_quantizer') and q.mse_quantizer is not None:
        _move_quantizer(q.mse_quantizer, device)


def _remove_dynamic_cache_patch():
    """Restore original DynamicCache.update."""
    try:
        from transformers.cache_utils import DynamicCache
        if hasattr(DynamicCache, '_tq_original_update'):
            DynamicCache.update = DynamicCache._tq_original_update
            del DynamicCache._tq_original_update
    except Exception:
        pass


def apply_turboquant_multigpu(model, effective_bits=3.5, n_outlier_channels=32,
                               quantizer_type='prod', seed=42):
    """
    Apply TurboQuant KV cache quantization via DynamicCache.update patch.

    Compatible with transformers >= 5.0 where attention forward no longer
    returns the Cache object. Quantizers are created on each layer's actual
    device to support device_map='auto' multi-GPU setups.
    """
    config = model.config
    head_dim = getattr(config, 'head_dim', config.hidden_size // config.num_attention_heads)
    n_layers = config.num_hidden_layers
    n_regular = head_dim - n_outlier_channels

    # Compute bit allocation
    regular_bits = int(math.floor(effective_bits))
    outlier_bits = regular_bits + 1
    actual_eff = (n_outlier_channels * outlier_bits + n_regular * regular_bits) / head_dim
    if abs(actual_eff - effective_bits) > 0.01:
        for rb in range(1, 8):
            for ob in range(rb, rb + 3):
                eff = (n_outlier_channels * ob + n_regular * rb) / head_dim
                if abs(eff - effective_bits) < 0.01:
                    regular_bits, outlier_bits = rb, ob
                    break

    print(f"TurboQuant KV: head_dim={head_dim}, "
          f"outlier={n_outlier_channels}ch@{outlier_bits}bit, "
          f"regular={n_regular}ch@{regular_bits}bit, "
          f"effective={effective_bits:.1f}bit")

    # Build per-layer quantizers on each layer's actual device
    kv_quantizers = {}
    for layer_idx, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        layer_device = next(attn.parameters()).device

        kv_quantizers[(layer_idx, 'k')] = TurboQuantKVCacheLocal(
            head_dim, n_outlier_channels, outlier_bits, regular_bits,
            device=layer_device, seed=seed + layer_idx * 2,
            quantizer_type=quantizer_type)
        kv_quantizers[(layer_idx, 'v')] = TurboQuantKVCacheLocal(
            head_dim, n_outlier_channels, outlier_bits, regular_bits,
            device=layer_device, seed=seed + layer_idx * 2 + 1,
            quantizer_type=quantizer_type)

    # Patch DynamicCache.update — restoring original first to avoid double-patch
    try:
        from transformers.cache_utils import DynamicCache

        if hasattr(DynamicCache, '_tq_original_update'):
            original_update = DynamicCache._tq_original_update
        else:
            original_update = DynamicCache.update
            DynamicCache._tq_original_update = original_update

        def quantized_update(self, key_states, value_states, layer_idx, *args, **kwargs):
            if (layer_idx, 'k') in kv_quantizers:
                k_q = kv_quantizers[(layer_idx, 'k')]
                v_q = kv_quantizers[(layer_idx, 'v')]
                dev = key_states.device
                k_q.move_to(dev)
                v_q.move_to(dev)
                orig_k = key_states.shape
                orig_v = value_states.shape
                key_states   = k_q.quantize_dequantize(key_states.reshape(-1, orig_k[-1])).reshape(orig_k)
                value_states = v_q.quantize_dequantize(value_states.reshape(-1, orig_v[-1])).reshape(orig_v)
            return original_update(self, key_states, value_states, layer_idx, *args, **kwargs)

        DynamicCache.update = quantized_update
        print(f"  Patched DynamicCache.update for {n_layers} layers")

    except Exception as e:
        print(f"  WARNING: DynamicCache patch failed ({e})")

    return model


# ═══════════════════════════════════════════════════════════════════════
# NIAH helpers
# ═══════════════════════════════════════════════════════════════════════

def build_prompt_llama(tokenizer, target_tokens, depth_percent):
    """Build NIAH prompt in Llama-3 chat format."""
    filler_tokens = tokenizer.encode(HAYSTACK_FILLER, add_special_tokens=False)
    needle_tokens = tokenizer.encode(NEEDLE, add_special_tokens=False)

    available_tokens = target_tokens - len(needle_tokens) - 200

    n_repeats = (available_tokens // len(filler_tokens)) + 1
    full_filler_ids = tokenizer.encode(
        HAYSTACK_FILLER * n_repeats, add_special_tokens=False)[:available_tokens]

    insert_pos = int(len(full_filler_ids) * depth_percent / 100)
    context_ids = full_filler_ids[:insert_pos] + needle_tokens + full_filler_ids[insert_pos:]
    context = tokenizer.decode(context_ids, skip_special_tokens=True)

    messages = [
        {"role": "system", "content": "You are a helpful assistant. Answer questions based only on the provided context. Be concise."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {NEEDLE_QUESTION}\nAnswer concisely:"},
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt


def evaluate_response(response):
    """Score based on keyword presence. Returns float in [0, 1]."""
    resp = response.lower()
    keywords = ["sandwich", "dolores park", "san francisco", "sunny"]
    return sum(1 for kw in keywords if kw in resp) / len(keywords)


def run_single_niah(model, tokenizer, token_limit, depth_percent, max_new_tokens=100):
    """Run one NIAH test point."""
    prompt = build_prompt_llama(tokenizer, token_limit, depth_percent)

    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=token_limit)
    input_device = model.model.embed_tokens.weight.device
    input_ids = inputs['input_ids'].to(input_device)
    attention_mask = inputs['attention_mask'].to(input_device)
    actual_len = input_ids.shape[1]

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )

    response = tokenizer.decode(outputs[0][actual_len:], skip_special_tokens=True)
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
    score = evaluate_response(response)

    return score, response, actual_len


def run_niah_config(model_path, tokenizer, effective_bits, config_label):
    """Load model, optionally apply TurboQuant, run full NIAH grid, return results."""
    print(f"\n{'=' * 60}")
    print(f"Config: {config_label}")
    print(f"{'=' * 60}")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        trust_remote_code=True,
        attn_implementation='sdpa',  # sdpa (flash attn) passes through DynamicCache.update
    )
    model.eval()

    if hasattr(model, 'hf_device_map'):
        devices_used = set(str(v) for v in model.hf_device_map.values())
        print(f"  Devices: {devices_used}")

    if effective_bits is not None:
        model = apply_turboquant_multigpu(
            model,
            effective_bits=effective_bits,
            n_outlier_channels=32,
            quantizer_type='prod',
            seed=42,
        )

    results = []
    for tl in TOKEN_LIMITS:
        print(f"\n  --- Context: {tl//1000}K tokens ---")
        for dp in DEPTH_PERCENTS:
            t0 = time.time()
            try:
                score, response, actual_len = run_single_niah(model, tokenizer, tl, dp)
                elapsed = time.time() - t0
                results.append((tl, dp, score))
                print(f"    depth={dp:3d}%, score={score:.2f}, len={actual_len}, "
                      f"time={elapsed:.1f}s | {response[:80]}")
            except torch.cuda.OutOfMemoryError:
                print(f"    depth={dp:3d}% — OOM at {tl//1000}K tokens, skipping")
                results.append((tl, dp, -1))
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"    depth={dp:3d}% — ERROR: {e}")
                results.append((tl, dp, 0.0))
            torch.cuda.empty_cache()

        tl_results = [r for r in results if r[0] == tl]
        if all(r[2] == -1 for r in tl_results):
            print(f"  All OOM at {tl//1000}K, stopping")
            break

    # Cleanup
    del model
    _remove_dynamic_cache_patch()
    torch.cuda.empty_cache()
    gc.collect()
    time.sleep(2)

    valid = [(tl, dp, s) for tl, dp, s in results if s >= 0]
    avg_score = float(np.mean([s for _, _, s in valid])) if valid else 0.0
    print(f"  Average score: {avg_score:.4f}")
    return results, avg_score


# ═══════════════════════════════════════════════════════════════════════
# Figure rendering — academic style
# ═══════════════════════════════════════════════════════════════════════

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'font.size': 10,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


def render_niah_comparison(all_configs, output_path):
    """
    Render side-by-side NIAH heatmaps for multiple configurations.
    all_configs: list of (label, results, avg_score)
    """
    n = len(all_configs)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        'niah', ['#FDECEC', '#FDD49E', '#ADDD8E', '#31A354', '#006837'])

    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 4.0), sharey=True)
    if n == 1:
        axes = [axes]

    all_tls = sorted(set(r[0] for cfg in all_configs for r in cfg[1] if r[2] >= 0))
    all_dps = sorted(set(r[1] for cfg in all_configs for r in cfg[1]))

    token_limits = TOKEN_LIMITS
    depths = DEPTH_PERCENTS
    token_labels = [f'{t//1000}K' for t in token_limits]

    for ax, (label, results, avg_score) in zip(axes, all_configs):
        matrix = np.zeros((len(depths), len(token_limits)))
        for tl, dp, s in results:
            if tl in token_limits and dp in depths:
                i = depths.index(dp)
                j = token_limits.index(tl)
                matrix[i, j] = max(0, s)

        im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=0, vmax=1,
                       interpolation='nearest')

        for i in range(len(depths)):
            for j in range(len(token_limits)):
                v = matrix[i, j]
                color = 'white' if v > 0.6 else 'black'
                ax.text(j, i, f'{v:.1f}', ha='center', va='center',
                        fontsize=5.5, color=color, fontweight='medium')

        ax.set_xticks(range(len(token_limits)))
        ax.set_xticklabels(token_labels, fontsize=7.5)
        ax.set_xlabel('Context Length')
        if ax == axes[0]:
            ax.set_yticks(range(len(depths)))
            ax.set_yticklabels([f'{d}%' for d in depths], fontsize=7.5)
            ax.set_ylabel('Needle Depth')
        else:
            ax.set_yticks([])
        ax.set_title(f'{label}\nAvg: {avg_score:.3f}', fontsize=9, pad=4)

    plt.subplots_adjust(left=0.06, right=0.88, wspace=0.08)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Retrieval Score', fontsize=8)

    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved: {output_path}")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def find_model_path():
    import glob
    ms_patterns = [
        os.path.expanduser("~/.cache/modelscope/hub/LLM-Research/Meta-Llama-3*8B-Instruct*"),
        "/home/xinhuogrp/denglishuo/.cache/modelscope/**/Meta-Llama-3*8B-Instruct*",
    ]
    for pat in ms_patterns:
        for m in glob.glob(pat, recursive=True):
            if os.path.isdir(m) and os.path.exists(os.path.join(m, 'config.json')):
                return m
    for path in MODEL_CANDIDATES:
        if os.path.exists(path) and os.path.exists(os.path.join(path, 'config.json')):
            return path
    return None


def main():
    print("=" * 70)
    print("Experiment 4.2 (Extended): Multi-bitwidth NIAH Comparison")
    print("Configs: Full Precision | 4-bit | 3.5-bit | 2.5-bit")
    print("=" * 70)

    model_path = find_model_path()
    if model_path is None:
        print("ERROR: No Llama model found. Trying to download from ModelScope...")
        try:
            from modelscope import snapshot_download
            model_path = snapshot_download(
                'LLM-Research/Meta-Llama-3.1-8B-Instruct',
                cache_dir=os.path.expanduser('~/.cache/modelscope'))
            print(f"  Downloaded to: {model_path}")
        except Exception as e:
            print(f"  Download failed: {e}")
            sys.exit(1)

    print(f"Using model: {model_path}")
    n_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {n_gpus}")
    for i in range(n_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name}, {props.total_memory // 1024**3}GB")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Configurations to evaluate
    configs = [
        ("Full Precision\n(16-bit)", None),
        ("TurboQuant\n(4-bit)",      4.0),
        ("TurboQuant\n(3.5-bit)",    3.5),
        ("TurboQuant\n(2.5-bit)",    2.5),
    ]

    all_results = {}

    for config_label, effective_bits in configs:
        clean_label = config_label.replace('\n', ' ')
        results, avg_score = run_niah_config(
            model_path, tokenizer, effective_bits, clean_label)
        all_results[clean_label] = {
            'effective_bits': effective_bits,
            'results': results,
            'score': avg_score,
        }

    # ─── Save raw results ───
    serializable = {}
    for label, data in all_results.items():
        serializable[label] = {
            'effective_bits': data['effective_bits'],
            'score': data['score'],
            'results': [[int(tl), int(dp), float(s)] for tl, dp, s in data['results']],
        }
    # Also write niah_llama_results.json in the format render_figures.py expects
    token_limits = TOKEN_LIMITS
    depth_percents = DEPTH_PERCENTS
    fp_data = all_results.get('Full Precision (16-bit)', {})
    tq35_data = all_results.get('TurboQuant (3.5-bit)', {})
    legacy_json = {
        'model': model_path,
        'token_limits': token_limits,
        'depth_percents': depth_percents,
        'full_precision': {
            'results': [[int(tl), int(dp), float(s)] for tl, dp, s in fp_data.get('results', [])],
            'score': fp_data.get('score', 0.0),
        },
        'turboquant_3.5bit': {
            'results': [[int(tl), int(dp), float(s)] for tl, dp, s in tq35_data.get('results', [])],
            'score': tq35_data.get('score', 0.0),
        },
    }
    with open(os.path.join(OUTPUT_DIR, 'niah_llama_results.json'), 'w') as f:
        json.dump(legacy_json, f, indent=2)

    extended_json = {
        'model': model_path,
        'token_limits': token_limits,
        'depth_percents': depth_percents,
        'configs': serializable,
    }
    with open(os.path.join(OUTPUT_DIR, 'niah_extended_results.json'), 'w') as f:
        json.dump(extended_json, f, indent=2)
    print(f"\nResults saved to {OUTPUT_DIR}/niah_extended_results.json")

    # ─── Render figures ───
    # Figure 4a: Full Precision vs TurboQuant 3.5-bit (paper comparison)
    configs_2 = [
        ('Full Precision (16-bit)', all_results['Full Precision (16-bit)']['results'],
         all_results['Full Precision (16-bit)']['score']),
        ('TurboQuant (3.5-bit)', all_results['TurboQuant (3.5-bit)']['results'],
         all_results['TurboQuant (3.5-bit)']['score']),
    ]
    render_niah_comparison(configs_2,
                           os.path.join(OUTPUT_DIR, 'fig4_niah.png'))

    # Figure 4b: All 4 configs
    configs_4 = [
        (lbl.replace('\n', ' '), all_results[lbl.replace('\n', ' ')]['results'],
         all_results[lbl.replace('\n', ' ')]['score'])
        for lbl, _ in configs
    ]
    render_niah_comparison(configs_4,
                           os.path.join(OUTPUT_DIR, 'fig4_niah_multibit.png'))

    # ─── Summary table ───
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    header = f"{'Config':<30} | {'Avg':>6} | " + " | ".join(f"{t//1000}K" for t in TOKEN_LIMITS)
    print(header)
    print("-" * len(header))
    for label, data in all_results.items():
        valid = [(tl, dp, s) for tl, dp, s in data['results'] if s >= 0]
        per_tl = []
        for tl in TOKEN_LIMITS:
            scores = [s for t, d, s in valid if t == tl]
            per_tl.append(f"{np.mean(scores):5.3f}" if scores else "  N/A")
        row = f"{label:<30} | {data['score']:6.3f} | " + " | ".join(per_tl)
        print(row)

    print("\nDone!")


if __name__ == "__main__":
    main()
