"""
Experiment 4.2 (Extended): Multi-GPU Long-Context NIAH with Llama-3.1-8B-Instruct.

Tests KV cache quantization fidelity at context lengths from 4K to 128K tokens,
matching the full range in the TurboQuant paper (arXiv:2504.19874, Figure 4).

Uses device_map='auto' to distribute model across multiple GPUs.
Quantizers are created on the correct device per layer.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import matplotlib.pyplot as plt
import gc
import time
import json
import math
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from turboquant import TurboQuantProd, TurboQuantMSE

# ─── Config ───
MODEL_PATH = None  # Will be resolved at runtime
MODEL_CANDIDATES = [
    "/home/xinhuogrp/denglishuo/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3___1-8B-Instruct",
    # Fallback: local Gradient model (same architecture, 1M context)
    "/home/xinhuogrp/xushaojie/2_duoattention_orgin/duo-attention/models/Llama-3-8B-Instruct-Gradient-1048k",
]

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# NIAH parameters — full paper range
TOKEN_LIMITS = [4000, 8000, 16000, 32000, 64000, 128000]
DEPTH_PERCENTS = list(range(0, 101, 10))  # finer granularity: 0,10,20,...,100

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
# Device-aware KV Cache Quantization (multi-GPU compatible)
# ═══════════════════════════════════════════════════════════════════════

class TurboQuantKVCacheLocal:
    """KV cache quantizer that lives on a specific device."""

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


def apply_turboquant_multigpu(model, effective_bits=3.5, n_outlier_channels=32,
                               quantizer_type='prod', seed=42):
    """
    Apply TurboQuant KV cache quantization, creating quantizers on each layer's
    actual device. Works with device_map='auto' across multiple GPUs.
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

    # Patch each layer, detecting its actual device
    for layer_idx, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        # Detect device from layer parameters
        layer_device = next(attn.parameters()).device

        # Create quantizers on the correct device
        k_quantizer = TurboQuantKVCacheLocal(
            head_dim, n_outlier_channels, outlier_bits, regular_bits,
            device=layer_device, seed=seed + layer_idx * 2, quantizer_type=quantizer_type)
        v_quantizer = TurboQuantKVCacheLocal(
            head_dim, n_outlier_channels, outlier_bits, regular_bits,
            device=layer_device, seed=seed + layer_idx * 2 + 1, quantizer_type=quantizer_type)

        # Patch forward
        original_forward = attn.forward

        def make_quantized_forward(orig_fwd, k_q, v_q, l_idx):
            def quantized_forward(*args, **kwargs):
                outputs = orig_fwd(*args, **kwargs)
                if isinstance(outputs, tuple) and len(outputs) >= 3:
                    attn_output, attn_weights, past_kv = outputs[0], outputs[1], outputs[2]
                    if past_kv is not None and hasattr(past_kv, 'key_cache'):
                        if l_idx < len(past_kv.key_cache):
                            k = past_kv.key_cache[l_idx]
                            v = past_kv.value_cache[l_idx]
                            batch, n_heads, seq_len, hd = k.shape
                            past_kv.key_cache[l_idx] = k_q.quantize_dequantize(
                                k.reshape(-1, hd)).reshape(k.shape)
                            past_kv.value_cache[l_idx] = v_q.quantize_dequantize(
                                v.reshape(-1, hd)).reshape(v.shape)
                        outputs = (attn_output, attn_weights, past_kv)
                return outputs
            return quantized_forward

        attn.forward = make_quantized_forward(original_forward, k_quantizer, v_quantizer, layer_idx)

    print(f"  Patched {n_layers} layers across devices")
    return model


# ═══════════════════════════════════════════════════════════════════════
# NIAH helpers
# ═══════════════════════════════════════════════════════════════════════

def build_prompt_llama(tokenizer, target_tokens, depth_percent):
    """Build NIAH prompt in Llama-3 chat format."""
    filler_tokens = tokenizer.encode(HAYSTACK_FILLER, add_special_tokens=False)
    needle_tokens = tokenizer.encode(NEEDLE, add_special_tokens=False)

    # Reserve for needle + question + chat template overhead
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
    # For multi-GPU: move to the device of the first embedding layer
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
    # Strip any thinking tags (Qwen3 compat, shouldn't appear for Llama)
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
    score = evaluate_response(response)

    return score, response, actual_len


def plot_niah_heatmap(results, title, output_path, overall_score=None):
    """Plot NIAH heatmap."""
    token_limits = sorted(set(r[0] for r in results))
    depths = sorted(set(r[1] for r in results))

    score_matrix = np.zeros((len(depths), len(token_limits)))
    for tl, dp, score in results:
        i = depths.index(dp)
        j = token_limits.index(tl)
        score_matrix[i, j] = score

    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(score_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

    token_labels = [f"{t//1000}K" for t in token_limits]
    ax.set_xticks(range(len(token_limits)))
    ax.set_xticklabels(token_labels, rotation=45, ha='right')
    ax.set_yticks(range(len(depths)))
    ax.set_yticklabels([f"{d}%" for d in depths])
    ax.set_xlabel("Context Length (tokens)", fontsize=12)
    ax.set_ylabel("Needle Depth", fontsize=12)

    # Annotate cells with scores
    for i in range(len(depths)):
        for j in range(len(token_limits)):
            val = score_matrix[i, j]
            color = 'white' if val < 0.5 else 'black'
            ax.text(j, i, f"{val:.1f}", ha='center', va='center', fontsize=7, color=color)

    score_str = f"\nAvg Score: {overall_score:.3f}" if overall_score is not None else ""
    ax.set_title(f"{title}{score_str}", fontsize=14)
    plt.colorbar(im, ax=ax, label='Score')
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def find_model_path():
    """Find the first available model path."""
    # Also check modelscope default download locations
    import glob
    ms_patterns = [
        os.path.expanduser("~/.cache/modelscope/hub/LLM-Research/Meta-Llama-3*8B-Instruct*"),
        os.path.expanduser("~/.cache/modelscope/models/LLM-Research/Meta-Llama-3*8B-Instruct*"),
        "/home/xinhuogrp/denglishuo/.cache/modelscope/**/Meta-Llama-3*8B-Instruct*",
    ]
    for pat in ms_patterns:
        matches = glob.glob(pat, recursive=True)
        for m in matches:
            if os.path.isdir(m) and os.path.exists(os.path.join(m, 'config.json')):
                return m

    for path in MODEL_CANDIDATES:
        if os.path.exists(path) and os.path.exists(os.path.join(path, 'config.json')):
            return path

    return None


def main():
    print("=" * 70)
    print("Experiment 4.2 (Extended): Multi-GPU Long-Context NIAH")
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
            print("  Falling back to local Gradient model...")
            model_path = "/home/xinhuogrp/xushaojie/2_duoattention_orgin/duo-attention/models/Llama-3-8B-Instruct-Gradient-1048k"

    print(f"\nUsing model: {model_path}")

    # Check available GPUs
    n_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {n_gpus}")
    for i in range(n_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name}, total={props.total_memory // 1024**3}GB")

    # Select GPUs with most free memory (use nvidia-smi for accurate free mem)
    print(f"\nLoading model with device_map='auto' across {n_gpus} GPUs...")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ─── Phase 1: Full Precision ───
    print("\n" + "=" * 60)
    print("Phase 1: Full Precision Baseline")
    print("=" * 60)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        trust_remote_code=True,
        attn_implementation='sdpa',  # PyTorch native scaled dot product attention
    )
    model.eval()

    # Show device map
    if hasattr(model, 'hf_device_map'):
        devices_used = set(str(v) for v in model.hf_device_map.values())
        print(f"  Model distributed across: {devices_used}")

    results_fp = []
    for tl in TOKEN_LIMITS:
        print(f"\n  --- Context: {tl//1000}K tokens ---")
        for dp in DEPTH_PERCENTS:
            t0 = time.time()
            try:
                score, response, actual_len = run_single_niah(model, tokenizer, tl, dp)
                elapsed = time.time() - t0
                results_fp.append((tl, dp, score))
                print(f"    depth={dp:3d}%, score={score:.2f}, len={actual_len}, "
                      f"time={elapsed:.1f}s | {response[:80]}...")
            except torch.cuda.OutOfMemoryError:
                print(f"    depth={dp:3d}% — OOM at {tl//1000}K tokens, skipping larger sizes")
                results_fp.append((tl, dp, -1))  # -1 = OOM
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"    depth={dp:3d}% — ERROR: {e}")
                results_fp.append((tl, dp, 0.0))
            torch.cuda.empty_cache()

        # If all depths at this token limit were OOM, skip larger sizes
        tl_results = [r for r in results_fp if r[0] == tl]
        if all(r[2] == -1 for r in tl_results):
            print(f"  All OOM at {tl//1000}K, stopping FP tests at larger sizes")
            max_fp_tokens = tl
            break

    # Filter out OOM results for scoring
    valid_fp = [(tl, dp, s) for tl, dp, s in results_fp if s >= 0]
    overall_fp = np.mean([r[2] for r in valid_fp]) if valid_fp else 0.0

    plot_niah_heatmap(
        [(tl, dp, max(0, s)) for tl, dp, s in results_fp],
        f"Full-Precision ({os.path.basename(model_path)})",
        os.path.join(OUTPUT_DIR, 'niah_llama_full_precision.png'),
        overall_fp)

    # ─── Phase 2: TurboQuant 3.5-bit ───
    print("\n" + "=" * 60)
    print("Phase 2: TurboQuant 3.5-bit KV Cache")
    print("=" * 60)

    # Reload model fresh
    del model
    torch.cuda.empty_cache()
    gc.collect()
    time.sleep(2)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        trust_remote_code=True,
        attn_implementation='sdpa',
    )
    model.eval()

    # Apply device-aware TurboQuant
    model = apply_turboquant_multigpu(
        model,
        effective_bits=3.5,
        n_outlier_channels=32,
        quantizer_type='prod',
        seed=42,
    )

    results_tq = []
    for tl in TOKEN_LIMITS:
        print(f"\n  --- Context: {tl//1000}K tokens ---")
        for dp in DEPTH_PERCENTS:
            t0 = time.time()
            try:
                score, response, actual_len = run_single_niah(model, tokenizer, tl, dp)
                elapsed = time.time() - t0
                results_tq.append((tl, dp, score))
                print(f"    depth={dp:3d}%, score={score:.2f}, len={actual_len}, "
                      f"time={elapsed:.1f}s | {response[:80]}...")
            except torch.cuda.OutOfMemoryError:
                print(f"    depth={dp:3d}% — OOM at {tl//1000}K tokens")
                results_tq.append((tl, dp, -1))
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"    depth={dp:3d}% — ERROR: {e}")
                results_tq.append((tl, dp, 0.0))
            torch.cuda.empty_cache()

        tl_results = [r for r in results_tq if r[0] == tl]
        if all(r[2] == -1 for r in tl_results):
            print(f"  All OOM at {tl//1000}K, stopping TQ tests at larger sizes")
            break

    valid_tq = [(tl, dp, s) for tl, dp, s in results_tq if s >= 0]
    overall_tq = np.mean([r[2] for r in valid_tq]) if valid_tq else 0.0

    plot_niah_heatmap(
        [(tl, dp, max(0, s)) for tl, dp, s in results_tq],
        f"TurboQuant 3.5-bit ({os.path.basename(model_path)})",
        os.path.join(OUTPUT_DIR, 'niah_llama_turboquant.png'),
        overall_tq)

    # ─── Combined Figure ───
    print("\n" + "=" * 60)
    print("Generating combined Figure 4 (Extended)")
    print("=" * 60)

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    for ax, (results, label, score) in zip(axes, [
        (results_fp, "Full-Precision", overall_fp),
        (results_tq, "TurboQuant 3.5-bit", overall_tq),
    ]):
        token_limits = sorted(set(r[0] for r in results))
        depths = sorted(set(r[1] for r in results))
        score_matrix = np.full((len(depths), len(token_limits)), np.nan)
        for tl, dp, s in results:
            i = depths.index(dp)
            j = token_limits.index(tl)
            score_matrix[i, j] = max(0, s)

        im = ax.imshow(score_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        ax.set_xticks(range(len(token_limits)))
        ax.set_xticklabels([f"{t//1000}K" for t in token_limits], rotation=45, ha='right')
        ax.set_yticks(range(len(depths)))
        ax.set_yticklabels([f"{d}%" for d in depths])
        ax.set_xlabel("Context Length (tokens)")
        ax.set_ylabel("Needle Depth")
        ax.set_title(f"{label}\nAvg Score: {score:.3f}", fontsize=13)

        for i in range(len(depths)):
            for j in range(len(token_limits)):
                val = score_matrix[i, j]
                if not np.isnan(val):
                    color = 'white' if val < 0.5 else 'black'
                    ax.text(j, i, f"{val:.1f}", ha='center', va='center', fontsize=6, color=color)

    plt.colorbar(im, ax=axes, shrink=0.8, label='Score')
    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, 'figure4_niah_extended.png')
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Combined figure saved to: {fig_path}")

    # ─── Save results ───
    results_json = {
        'model': model_path,
        'token_limits': TOKEN_LIMITS,
        'depth_percents': DEPTH_PERCENTS,
        'full_precision': {'results': results_fp, 'score': float(overall_fp)},
        'turboquant_3.5bit': {'results': results_tq, 'score': float(overall_tq)},
    }
    json_path = os.path.join(OUTPUT_DIR, 'niah_llama_results.json')
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"Results saved to: {json_path}")

    # ─── Summary ───
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Full Precision Score:      {overall_fp:.4f}")
    print(f"  TurboQuant 3.5-bit Score:  {overall_tq:.4f}")
    print(f"  Difference:                {abs(overall_fp - overall_tq):.4f}")

    # Per-token-limit comparison
    print(f"\n  {'Tokens':>8} | {'FP Score':>10} | {'TQ Score':>10} | {'Diff':>8}")
    print("  " + "-" * 45)
    for tl in TOKEN_LIMITS:
        fp_scores = [s for t, d, s in valid_fp if t == tl]
        tq_scores = [s for t, d, s in valid_tq if t == tl]
        fp_avg = np.mean(fp_scores) if fp_scores else float('nan')
        tq_avg = np.mean(tq_scores) if tq_scores else float('nan')
        diff = abs(fp_avg - tq_avg) if not (np.isnan(fp_avg) or np.isnan(tq_avg)) else float('nan')
        print(f"  {tl:8d} | {fp_avg:10.4f} | {tq_avg:10.4f} | {diff:8.4f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
