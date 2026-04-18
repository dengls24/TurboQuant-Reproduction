"""
Experiment 4.3: End-to-end Generation Quality with TurboQuant KV Cache Quantization.
Simplified version of Table 1 from the paper.

Since LongBench dataset is unavailable, we use a set of representative QA/summarization
tasks to demonstrate that TurboQuant KV cache quantization preserves generation quality.

Evaluates: Full Cache, TurboQuant 2.5-bit, TurboQuant 3.5-bit
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import json
import gc
import re
import string
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM
from kv_cache_quant import apply_turboquant_to_kv_cache

# ─── Config ───
MODEL_NAME = "Qwen/Qwen3-8B"
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_NEW_TOKENS = 512  # enough room for Qwen3 thinking + answer


# ─── Test data: representative tasks ───

QA_TASKS = [
    {
        "context": (
            "The Great Wall of China is a series of fortifications that were built across "
            "the historical northern borders of ancient Chinese states and Imperial China as "
            "protection against various nomadic groups. Several walls were built as early as "
            "the 7th century BC, with the most well-known sections built by the Ming Dynasty "
            "(1368-1644). The wall spans approximately 21,196 kilometers."
        ),
        "question": "How long is the Great Wall of China?",
        "answer": "approximately 21,196 kilometers",
    },
    {
        "context": (
            "Photosynthesis is a process used by plants and other organisms to convert light "
            "energy into chemical energy that can be later released to fuel the organism's "
            "activities. This chemical energy is stored in carbohydrate molecules, such as "
            "sugars and starches, which are synthesized from carbon dioxide and water. In most "
            "cases, oxygen is released as a waste product."
        ),
        "question": "What is released as a waste product of photosynthesis?",
        "answer": "oxygen",
    },
    {
        "context": (
            "The Pythagorean theorem states that in a right triangle, the square of the "
            "length of the hypotenuse equals the sum of the squares of the lengths of the "
            "other two sides. This is written as a^2 + b^2 = c^2, where c is the hypotenuse. "
            "The theorem is named after the ancient Greek mathematician Pythagoras."
        ),
        "question": "What does the Pythagorean theorem state?",
        "answer": "the square of the hypotenuse equals the sum of squares of the other two sides",
    },
    {
        "context": (
            "DNA, or deoxyribonucleic acid, is the molecule that contains the genetic code "
            "of organisms. DNA is in each cell in the organism and tells cells what proteins "
            "to make. The structure of DNA was first described by James Watson and Francis "
            "Crick in 1953. They proposed the double helix model."
        ),
        "question": "Who first described the structure of DNA?",
        "answer": "James Watson and Francis Crick",
    },
    {
        "context": (
            "The Amazon River is the largest river by discharge volume of water in the world, "
            "and the disputed longest. The headwaters of the Apurímac River on Nevado Mismi "
            "had been considered for nearly a century as the source. The river flows through "
            "the Amazon Rainforest, which makes up the largest tropical rainforest in the world."
        ),
        "question": "What distinction does the Amazon River hold?",
        "answer": "largest river by discharge volume",
    },
]

# Longer context QA (to test with more KV cache pressure)
LONG_QA_TASKS = []
filler = (
    "The history of science is marked by numerous breakthroughs and discoveries that have "
    "shaped our understanding of the world. From ancient Greek philosophy to modern quantum "
    "mechanics, each era has contributed to the vast body of human knowledge. "
) * 50  # ~200 words * 50 = ~10K words

for i, qa in enumerate(QA_TASKS[:3]):
    LONG_QA_TASKS.append({
        "context": filler + qa["context"] + filler,
        "question": qa["question"],
        "answer": qa["answer"],
    })


# ─── Metrics ───

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        return ''.join(ch for ch in text if ch not in string.punctuation)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(prediction_tokens) if prediction_tokens else 0
    recall = 1.0 * num_same / len(ground_truth_tokens) if ground_truth_tokens else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1


def evaluate_qa(model, tokenizer, tasks, task_name="QA"):
    """Evaluate on QA tasks, return average F1."""
    scores = []
    for i, task in enumerate(tasks):
        prompt = (
            f"<|im_start|>user\n"
            f"Answer the question based on the context. Be concise.\n\n"
            f"Context: {task['context']}\n\n"
            f"Question: {task['question']}\n"
            f"<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=8192)
        input_ids = inputs['input_ids'].to(model.device)

        try:
            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    use_cache=True,
                )
            response = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
            response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
            score = f1_score(response, task['answer'])
            scores.append(score * 100)
            print(f"    [{task_name} {i+1}] F1={score*100:.1f} | ans='{task['answer'][:50]}' | pred='{response[:80]}'")
        except Exception as e:
            print(f"    [{task_name} {i+1}] ERROR: {e}")
            scores.append(0.0)
        torch.cuda.empty_cache()

    return np.mean(scores) if scores else 0.0


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("Experiment 4.3: Generation Quality with TurboQuant KV Cache")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_results = {}

    configs = [
        ("Full Cache (16-bit)", None),
        ("TurboQuant (4-bit)",  4.0),
        ("TurboQuant (3.5-bit)", 3.5),
        ("TurboQuant (2.5-bit)", 2.5),
    ]

    for config_name, eff_bits in configs:
        print(f"\n{'=' * 50}")
        print(f"Evaluating: {config_name}")
        print(f"{'=' * 50}")

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map='auto',
            trust_remote_code=True,
            attn_implementation='eager',
        )

        if eff_bits is not None:
            model = apply_turboquant_to_kv_cache(
                model,
                effective_bits=eff_bits,
                n_outlier_channels=32,
                quantizer_type='prod',
                device=DEVICE,
            )

        # Short-context QA
        print("\n  Short-context QA:")
        short_qa_score = evaluate_qa(model, tokenizer, QA_TASKS, "ShortQA")

        # Long-context QA
        print("\n  Long-context QA:")
        long_qa_score = evaluate_qa(model, tokenizer, LONG_QA_TASKS, "LongQA")

        all_results[config_name] = {
            'ShortQA': short_qa_score,
            'LongQA': long_qa_score,
            'Average': (short_qa_score + long_qa_score) / 2,
        }

        print(f"\n  Results: ShortQA={short_qa_score:.1f}, LongQA={long_qa_score:.1f}, "
              f"Avg={(short_qa_score + long_qa_score) / 2:.1f}")

        del model
        torch.cuda.empty_cache()
        gc.collect()

    # ─── Print Table ───
    print("\n" + "=" * 70)
    print("Table: Generation Quality Results")
    print("=" * 70)
    categories = ['ShortQA', 'LongQA', 'Average']
    header = f"{'Method':<25} | " + " | ".join(f"{c:>10}" for c in categories)
    print(header)
    print("-" * len(header))
    for method, scores in all_results.items():
        row = f"{method:<25} | " + " | ".join(f"{scores[c]:>10.1f}" for c in categories)
        print(row)

    # Save results
    json_path = os.path.join(OUTPUT_DIR, 'longbench_results.json')
    serializable = {k: {kk: float(vv) for kk, vv in v.items()} for k, v in all_results.items()}
    with open(json_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to: {json_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()
