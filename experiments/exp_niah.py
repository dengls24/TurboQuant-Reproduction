"""
Experiment 4.2: Needle-In-A-Haystack Test with TurboQuant KV Cache Quantization.
Reproduces Figure 4 from the paper.

Tests Llama-3.1-8B-Instruct on the NIAH benchmark with:
  - Full precision (baseline)
  - TurboQuant KV cache quantization (3.5-bit effective)

Document sizes from 4K to 104K tokens, needle placed at various depth percentages.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from transformers import AutoTokenizer, AutoModelForCausalLM
import gc
import time
import json
from kv_cache_quant import apply_turboquant_to_kv_cache

# ─── Config ───
MODEL_NAME = "Qwen/Qwen3-8B"
# Try local / alternative names
MODEL_ALTERNATIVES = [
    "Qwen/Qwen3-8B",
]

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# NIAH parameters — use smaller token limits for faster testing
TOKEN_LIMITS = [4000, 8000, 16000, 28000]
DEPTH_PERCENTS = list(range(0, 101, 20))  # 0, 20, 40, 60, 80, 100

# The "needle" sentence to hide
NEEDLE = "The best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day."
NEEDLE_QUESTION = "What is the best thing to do in San Francisco?"

# Haystack text (repeated filler)
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


def build_haystack_with_needle(tokenizer, target_tokens, depth_percent):
    """
    Build a text of approximately target_tokens length with the needle
    inserted at the specified depth percentage.
    """
    # Tokenize filler to know its length
    filler_tokens = tokenizer.encode(HAYSTACK_FILLER, add_special_tokens=False)
    needle_tokens = tokenizer.encode(NEEDLE, add_special_tokens=False)
    question_tokens = tokenizer.encode(NEEDLE_QUESTION, add_special_tokens=False)

    # Reserve space for needle + question + special tokens
    available_tokens = target_tokens - len(needle_tokens) - len(question_tokens) - 50

    # Repeat filler to fill available_tokens
    n_repeats = (available_tokens // len(filler_tokens)) + 1
    full_filler = HAYSTACK_FILLER * n_repeats

    # Tokenize and truncate
    filler_token_ids = tokenizer.encode(full_filler, add_special_tokens=False)[:available_tokens]

    # Insert needle at depth_percent
    insert_pos = int(len(filler_token_ids) * depth_percent / 100)

    # Build final token sequence
    final_ids = filler_token_ids[:insert_pos] + needle_tokens + filler_token_ids[insert_pos:]

    # Decode back to text
    context = tokenizer.decode(final_ids, skip_special_tokens=True)

    # Build prompt (Qwen3/ChatML format)
    prompt = (
        f"<|im_start|>system\n"
        f"You are a helpful assistant. Answer the question based on the context below.\n"
        f"<|im_end|>\n"
        f"<|im_start|>user\n"
        f"Context: {context}\n\n"
        f"Question: {NEEDLE_QUESTION}\n"
        f"Answer the question concisely based on the context.\n"
        f"<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    return prompt


def evaluate_response(response: str) -> float:
    """Score the response: 1.0 if it mentions key elements, 0.0 otherwise."""
    response_lower = response.lower()
    keywords = ["sandwich", "dolores park", "san francisco", "sunny"]
    score = sum(1 for kw in keywords if kw in response_lower) / len(keywords)
    return score


def run_niah_test(model, tokenizer, token_limit, depth_percent, max_new_tokens=50):
    """Run a single NIAH test and return the score."""
    prompt = build_haystack_with_needle(tokenizer, token_limit, depth_percent)

    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=token_limit)
    input_ids = inputs['input_ids'].to(model.device)
    actual_len = input_ids.shape[1]

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            use_cache=True,
        )

    response = tokenizer.decode(outputs[0][actual_len:], skip_special_tokens=True)
    # Strip <think>...</think> tags if present (Qwen3 thinking mode)
    import re
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
    score = evaluate_response(response)

    return score, response, actual_len


def plot_niah_heatmap(results, title, output_path, overall_score=None):
    """Plot NIAH results as a heatmap."""
    token_limits = sorted(set(r[0] for r in results))
    depths = sorted(set(r[1] for r in results))

    # Build score matrix
    score_matrix = np.zeros((len(depths), len(token_limits)))
    for tl, dp, score in results:
        i = depths.index(dp)
        j = token_limits.index(tl)
        score_matrix[i, j] = score

    fig, ax = plt.subplots(figsize=(10, 8))

    # Custom colormap: red -> yellow -> green
    cmap = plt.cm.RdYlGn

    im = ax.imshow(score_matrix, cmap=cmap, aspect='auto', vmin=0, vmax=1)

    # Labels
    token_labels = [f"{t//1000}K" for t in token_limits]
    ax.set_xticks(range(len(token_limits)))
    ax.set_xticklabels(token_labels, rotation=45, ha='right')
    ax.set_yticks(range(len(depths)))
    ax.set_yticklabels(depths)

    ax.set_xlabel("Token Limit", fontsize=12)
    ax.set_ylabel("Depth Percent", fontsize=12)

    score_str = f"\nScore: {overall_score:.3f}" if overall_score is not None else ""
    ax.set_title(f"{title}{score_str}", fontsize=14)

    plt.colorbar(im, ax=ax, label='Score')
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Heatmap saved to: {output_path}")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("Experiment 4.2: Needle-In-A-Haystack Test")
    print("=" * 70)

    # Load model and tokenizer
    print(f"\nLoading model: {MODEL_NAME}")
    tokenizer = None
    model = None

    for model_name in MODEL_ALTERNATIVES:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map='auto',
                trust_remote_code=True,
            )
            print(f"  Loaded: {model_name}")
            break
        except Exception as e:
            print(f"  Failed to load {model_name}: {e}")
            continue

    if model is None:
        print("\nERROR: Could not load any model. Exiting.")
        return

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ─── Test 1: Full Precision ───
    print("\n" + "=" * 50)
    print("Test 1: Full Precision Baseline")
    print("=" * 50)

    results_fp = []
    for tl in TOKEN_LIMITS:
        for dp in DEPTH_PERCENTS:
            try:
                score, response, actual_len = run_niah_test(model, tokenizer, tl, dp)
                results_fp.append((tl, dp, score))
                print(f"  tokens={tl:6d}, depth={dp:3d}%, score={score:.2f}, actual_len={actual_len}")
            except Exception as e:
                print(f"  tokens={tl:6d}, depth={dp:3d}% — ERROR: {e}")
                results_fp.append((tl, dp, 0.0))
            torch.cuda.empty_cache()

    overall_fp = np.mean([r[2] for r in results_fp])
    plot_niah_heatmap(results_fp, "Full-Precision",
                      os.path.join(OUTPUT_DIR, 'niah_full_precision.png'), overall_fp)

    # ─── Test 2: TurboQuant 3.5-bit ───
    print("\n" + "=" * 50)
    print("Test 2: TurboQuant 3.5-bit KV Cache")
    print("=" * 50)

    # Reload model fresh
    del model
    torch.cuda.empty_cache()
    gc.collect()

    for model_name in MODEL_ALTERNATIVES:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map='auto',
                trust_remote_code=True,
            )
            break
        except:
            continue

    # Apply TurboQuant
    model = apply_turboquant_to_kv_cache(
        model,
        effective_bits=3.5,
        n_outlier_channels=32,
        quantizer_type='prod',
        device=DEVICE,
        seed=42,
    )

    results_tq = []
    for tl in TOKEN_LIMITS:
        for dp in DEPTH_PERCENTS:
            try:
                score, response, actual_len = run_niah_test(model, tokenizer, tl, dp)
                results_tq.append((tl, dp, score))
                print(f"  tokens={tl:6d}, depth={dp:3d}%, score={score:.2f}, actual_len={actual_len}")
            except Exception as e:
                print(f"  tokens={tl:6d}, depth={dp:3d}% — ERROR: {e}")
                results_tq.append((tl, dp, 0.0))
            torch.cuda.empty_cache()

    overall_tq = np.mean([r[2] for r in results_tq])
    plot_niah_heatmap(results_tq, "TurboQuant (3.5-bit)",
                      os.path.join(OUTPUT_DIR, 'niah_turboquant.png'), overall_tq)

    # ─── Combined Figure 4 ───
    print("\n" + "=" * 50)
    print("Generating combined Figure 4")
    print("=" * 50)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    all_data = [
        (results_fp, f"Full-Precision\nScore: {overall_fp:.3f}"),
        (results_tq, f"TurboQuant\nScore: {overall_tq:.3f}"),
    ]

    for ax, (results, title) in zip(axes, all_data):
        token_limits = sorted(set(r[0] for r in results))
        depths = sorted(set(r[1] for r in results))
        score_matrix = np.zeros((len(depths), len(token_limits)))
        for tl, dp, score in results:
            i = depths.index(dp)
            j = token_limits.index(tl)
            score_matrix[i, j] = score

        im = ax.imshow(score_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        token_labels = [f"{t//1000}K" for t in token_limits]
        ax.set_xticks(range(len(token_limits)))
        ax.set_xticklabels(token_labels, rotation=45, ha='right', fontsize=9)
        ax.set_yticks(range(len(depths)))
        ax.set_yticklabels(depths, fontsize=9)
        ax.set_xlabel("Token Limit")
        ax.set_ylabel("Depth Percent")
        ax.set_title(title, fontsize=13)

    plt.colorbar(im, ax=axes, shrink=0.8, label='Score')
    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, 'figure4_niah.png')
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Figure 4 saved to: {fig_path}")

    # Save results as JSON
    results_json = {
        'full_precision': {'results': results_fp, 'score': overall_fp},
        'turboquant_3.5bit': {'results': results_tq, 'score': overall_tq},
    }
    json_path = os.path.join(OUTPUT_DIR, 'niah_results.json')
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"Results saved to: {json_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
