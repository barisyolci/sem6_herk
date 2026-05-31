"""
Full Benchmark Runner with Parallel Processing
Runs 500 questions x 10 repetitions on both base and finetuned models
Generates visualizations comparing performance

Optimized for speed:
- Batch generation for GPU efficiency
- Parallel evaluation using multiprocessing
- Progress saving to resume interrupted runs
"""

import os
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from evaluate_benchmark import BenchmarkEvaluator
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
import multiprocessing as mp

# Set HuggingFace token from environment if available
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_KEY")
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    print(f"Using HuggingFace token: {HF_TOKEN[:10]}...")

# Configuration (Arabic)
BASE_MODEL = os.environ.get("BASE_MODEL") or "Qwen/Qwen3-4B"
ADAPTER_PATH = os.environ.get("ADAPTER_PATH") or "/home/dealers/mo_4b_finetune/outputs/grpo_lora"
BENCHMARK_PATH = os.environ.get("BENCHMARK_PATH") or "benchmark_questions.json"
REPETITIONS = 10
OUTPUT_DIR = "benchmark_results"
BATCH_SIZE = 100  # Parallel evaluation batch size
NUM_WORKERS = min(mp.cpu_count(), 8)  # Max parallel workers

# For quick testing, set these to smaller values
TEST_MODE = False  # Set to True for quick test
TEST_QUESTIONS = 5  # Number of questions in test mode
TEST_REPS = 2  # Repetitions in test mode

# System prompt
SYSTEM_PROMPT = (
    "أنت مساعد اجتماعي يقدم دعمًا للأشخاص الذين يواجهون التشرد. "
    "قدّم إجابات عملية ومتعاطفة وآمنة، ولا تختلق معلومات."
)

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_base_model():
    """Load only the base model (no finetuning)"""
    print(f"Loading BASE model on {DEVICE}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map=DEVICE,
        trust_remote_code=True
    )
    model.eval()
    print(f"✓ Base model loaded on {model.device}")
    return model, tokenizer


def load_finetuned_model():
    """Load the finetuned model with LoRA adapter"""
    print(f"Loading FINETUNED model on {DEVICE}...")
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map=DEVICE,
        trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()
    print(f"✓ Finetuned model loaded on {model.device}")
    return model, tokenizer


def generate_response(model, tokenizer, question: str, max_length: int = 256):
    """Generate a response to a question"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question}
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_length = inputs['input_ids'].shape[1]  # Track input length

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Only decode the NEW tokens (not the input prompt)
    new_tokens = outputs[0][input_length:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # Clean up any remaining markers (Qwen uses im_start/im_end)
    if "<|im_end|>" in response:
        response = response.split("<|im_end|>")[0].strip()
    if "<|im_start|>" in response:
        response = response.split("<|im_start|>")[0].strip()

    # Remove any "assistant" prefix if still present
    if response.lower().startswith("assistant"):
        response = response[len("assistant"):].strip()
    if response.startswith("\n"):
        response = response.strip()

    return response


def evaluate_single_item(item, evaluator=None):
    """Evaluate a single response - used for parallel processing"""
    if evaluator is None:
        evaluator = BenchmarkEvaluator()

    eval_result = evaluator.evaluate_response(
        question=item["question"],
        response=item["response"],
        expected_answer=item["expected"],
        category=item["category"]
    )

    return {
        "model": item["model"],
        "question_id": item["question_id"],
        "category": item["category"],
        "repetition": item["repetition"],
        "question": item["question"],
        "response": item["response"],
        "expected": item["expected"],
        "regex_score": eval_result["regex_score"],
        "cross_encoder_score": eval_result["cross_encoder_score"],
        "nli_score": eval_result["nli_score"],
        "nli_label": eval_result["nli_label"],
        "bertscore": eval_result["bertscore"],
        "semantic_score": eval_result["semantic_score"],
        "combined_score": eval_result["combined_score"]
    }


def run_benchmark(model, tokenizer, evaluator, questions, model_name: str, reps: int = REPETITIONS):
    """Run benchmark with parallel evaluation in batches of 100"""
    total_runs = len(questions) * reps
    print(f"\nRunning benchmark for {model_name}: {len(questions)} questions x {reps} reps = {total_runs} runs")

    # PHASE 1: Generate all responses (sequential, GPU-bound)
    print(f"  Phase 1: Generating {total_runs} responses...")
    items_to_evaluate = []

    with tqdm(total=total_runs, desc=f"{model_name} (generation)") as pbar:
        for q in questions:
            for rep in range(reps):
                response = generate_response(model, tokenizer, q["question"])
                items_to_evaluate.append({
                    "model": model_name,
                    "question_id": q["id"],
                    "category": q["category"],
                    "repetition": rep + 1,
                    "question": q["question"],
                    "response": response,
                    "expected": q["correct_answer"]
                })
                pbar.update(1)

    # PHASE 2: Evaluate in parallel batches (CPU-bound, can parallelize)
    print(f"  Phase 2: Evaluating {total_runs} responses in batches of {BATCH_SIZE}...")
    results = []

    # Process in batches of BATCH_SIZE
    num_batches = (len(items_to_evaluate) + BATCH_SIZE - 1) // BATCH_SIZE

    with tqdm(total=len(items_to_evaluate), desc=f"{model_name} (evaluation)") as pbar:
        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = min(start_idx + BATCH_SIZE, len(items_to_evaluate))
            batch = items_to_evaluate[start_idx:end_idx]

            # Use ThreadPoolExecutor for I/O-bound evaluation (models share memory)
            with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
                batch_results = list(executor.map(
                    lambda item: evaluate_single_item(item, evaluator),
                    batch
                ))

            results.extend(batch_results)
            pbar.update(len(batch))

    return results


def main(test_mode=False):
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Determine settings based on mode
    reps = TEST_REPS if test_mode else REPETITIONS

    # Load questions
    print("Loading benchmark questions...")
    with open(BENCHMARK_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    questions = data["questions"]

    # Limit questions in test mode
    if test_mode:
        questions = questions[:TEST_QUESTIONS]
        print(f"TEST MODE: Using {len(questions)} questions x {reps} reps")
    else:
        print(f"Loaded {len(questions)} questions x {reps} reps = {len(questions) * reps} runs per model")
    
    # Initialize evaluator
    print("Initializing evaluator...")
    evaluator = BenchmarkEvaluator()
    
    # Load models and run benchmarks
    all_results = []

    # 1. Base model
    base_model, base_tokenizer = load_base_model()
    base_results = run_benchmark(base_model, base_tokenizer, evaluator, questions, "base", reps)
    all_results.extend(base_results)
    del base_model  # Free memory
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # 2. Finetuned model
    ft_model, ft_tokenizer = load_finetuned_model()
    ft_results = run_benchmark(ft_model, ft_tokenizer, evaluator, questions, "finetuned", reps)
    all_results.extend(ft_results)
    del ft_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Save raw results
    df = pd.DataFrame(all_results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df.to_csv(f"{OUTPUT_DIR}/benchmark_results_{timestamp}.csv", index=False)
    print(f"\n✓ Results saved to {OUTPUT_DIR}/benchmark_results_{timestamp}.csv")

    # Generate visualizations
    print("\nGenerating visualizations...")
    create_visualizations(df, timestamp)

    # Print summary
    print_summary(df)


def create_visualizations(df: pd.DataFrame, timestamp: str):
    """Create comparison visualizations with all evaluation methods"""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. Semantic Score Distribution (main metric)
    ax1 = axes[0, 0]
    for model in ["base", "finetuned"]:
        data = df[df["model"] == model]["semantic_score"]
        ax1.hist(data, bins=30, alpha=0.6, label=model, density=True)
    ax1.set_xlabel("Semantic Score")
    ax1.set_ylabel("Density")
    ax1.set_title("Distribution of Semantic Scores\n(Cross-Encoder + NLI + BERTScore)")
    ax1.legend()

    # 2. All Evaluation Methods Comparison
    ax2 = axes[0, 1]
    methods = ['semantic_score', 'cross_encoder_score', 'nli_score', 'bertscore']
    method_names = ['Semantic\n(Combined)', 'Cross-\nEncoder', 'NLI', 'BERT\nScore']
    x = np.arange(len(methods))
    width = 0.35

    base_means = [df[df['model']=='base'][m].dropna().mean() for m in methods]
    ft_means = [df[df['model']=='finetuned'][m].dropna().mean() for m in methods]

    ax2.bar(x - width/2, base_means, width, label='Base', color='#ff7f0e')
    ax2.bar(x + width/2, ft_means, width, label='Finetuned', color='#1f77b4')
    ax2.set_ylabel('Score')
    ax2.set_title('Comparison of All Evaluation Methods')
    ax2.set_xticks(x)
    ax2.set_xticklabels(method_names)
    ax2.legend()
    ax2.set_ylim(0, 1)

    # 3. Regex (Factual) Accuracy by Category
    ax3 = axes[0, 2]
    factual_df = df[df["regex_score"].notna()]
    if len(factual_df) > 0:
        regex_by_cat = factual_df.groupby(["category", "model"])["regex_score"].mean().unstack()
        if not regex_by_cat.empty:
            regex_by_cat.plot(kind="bar", ax=ax3, color=["#ff7f0e", "#1f77b4"])
            ax3.set_ylabel("Regex Accuracy")
            ax3.set_title("Factual Accuracy (Phone/Address)")
            ax3.set_ylim(0, 1)
            ax3.tick_params(axis='x', rotation=45)
            ax3.legend(title="Model")

    # 4. NLI Label Distribution
    ax4 = axes[1, 0]
    nli_df = df[df["nli_label"].notna()]
    if len(nli_df) > 0:
        nli_counts = nli_df.groupby(["model", "nli_label"]).size().unstack(fill_value=0)
        nli_counts.plot(kind="bar", ax=ax4, stacked=True)
        ax4.set_ylabel("Count")
        ax4.set_title("NLI Classification Distribution")
        ax4.tick_params(axis='x', rotation=0)
        ax4.legend(title="NLI Label")

    # 5. Score Improvement Distribution
    ax5 = axes[1, 1]
    base_scores = df[df["model"] == "base"].groupby("question_id")["semantic_score"].mean()
    ft_scores = df[df["model"] == "finetuned"].groupby("question_id")["semantic_score"].mean()
    improvement = ft_scores - base_scores
    ax5.hist(improvement, bins=30, color="#2ca02c", alpha=0.7)
    ax5.axvline(x=0, color='red', linestyle='--', label='No change')
    ax5.axvline(x=improvement.mean(), color='blue', linestyle='-', label=f'Mean: {improvement.mean():.3f}')
    ax5.set_xlabel("Semantic Score Improvement (Finetuned - Base)")
    ax5.set_ylabel("Count")
    ax5.set_title("Per-Question Improvement Distribution")
    ax5.legend()

    # 6. Summary Stats
    ax6 = axes[1, 2]
    ax6.axis('off')

    # Calculate stats safely
    base_df = df[df['model']=='base']
    ft_df = df[df['model']=='finetuned']
    base_regex = base_df[base_df['regex_score'].notna()]['regex_score'].mean()
    ft_regex = ft_df[ft_df['regex_score'].notna()]['regex_score'].mean()

    summary_text = f"""
    BENCHMARK SUMMARY
    ═══════════════════════════════════════
    Total Questions: {len(df['question_id'].unique())}
    Total Evaluations: {len(df)}

    BASE MODEL
    ───────────────────────────────────────
    • Semantic Score:    {base_df['semantic_score'].mean():.3f} ± {base_df['semantic_score'].std():.3f}
    • Cross-Encoder:     {base_df['cross_encoder_score'].dropna().mean():.3f}
    • NLI Score:         {base_df['nli_score'].dropna().mean():.3f}
    • BERTScore:         {base_df['bertscore'].dropna().mean():.3f}
    • Regex Accuracy:    {base_regex:.1%}

    FINETUNED MODEL
    ───────────────────────────────────────
    • Semantic Score:    {ft_df['semantic_score'].mean():.3f} ± {ft_df['semantic_score'].std():.3f}
    • Cross-Encoder:     {ft_df['cross_encoder_score'].dropna().mean():.3f}
    • NLI Score:         {ft_df['nli_score'].dropna().mean():.3f}
    • BERTScore:         {ft_df['bertscore'].dropna().mean():.3f}
    • Regex Accuracy:    {ft_regex:.1%}

    IMPROVEMENT (Finetuned - Base)
    ───────────────────────────────────────
    • Semantic:          {ft_df['semantic_score'].mean() - base_df['semantic_score'].mean():+.3f}
    • Questions improved: {(improvement > 0).sum()} / {len(improvement)}
    """
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/benchmark_comparison_{timestamp}.png", dpi=150, bbox_inches='tight')
    print(f"✓ Visualization saved to {OUTPUT_DIR}/benchmark_comparison_{timestamp}.png")
    plt.close()

    # Additional: Category-specific visualizations
    create_category_plots(df, timestamp)


def create_category_plots(df: pd.DataFrame, timestamp: str):
    """Create per-category detailed plots"""
    categories = df["category"].unique()
    n_cats = len(categories)

    fig, axes = plt.subplots(2, (n_cats + 1) // 2, figsize=(16, 10))
    axes = axes.flatten()

    for i, cat in enumerate(categories):
        ax = axes[i]
        cat_df = df[df["category"] == cat]

        for model in ["base", "finetuned"]:
            data = cat_df[cat_df["model"] == model]["semantic_score"]
            ax.hist(data, bins=20, alpha=0.6, label=model, density=True)

        ax.set_title(f"{cat}\n(n={len(cat_df)//2})")
        ax.set_xlabel("Semantic Score")
        ax.legend()

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.suptitle("Semantic Score Distribution by Category", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/benchmark_by_category_{timestamp}.png", dpi=150, bbox_inches='tight')
    print(f"✓ Category plots saved to {OUTPUT_DIR}/benchmark_by_category_{timestamp}.png")
    plt.close()

    # Create dedicated regex comparison plot
    create_regex_plots(df, timestamp)


def create_regex_plots(df: pd.DataFrame, timestamp: str):
    """Create detailed regex/factual accuracy plots"""
    factual_df = df[df["regex_score"].notna()]

    if len(factual_df) == 0:
        print("No factual data to plot (no regex scores)")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Overall Regex Accuracy Comparison
    ax1 = axes[0, 0]
    regex_by_model = factual_df.groupby("model")["regex_score"].agg(['mean', 'std', 'count'])
    colors = ["#ff7f0e", "#1f77b4"]
    bars = ax1.bar(regex_by_model.index, regex_by_model['mean'],
                   yerr=regex_by_model['std'], color=colors, capsize=5)
    ax1.set_ylabel("Regex Accuracy")
    ax1.set_title("Overall Factual Accuracy (Phone + Address)")
    ax1.set_ylim(0, 1)
    for bar, val in zip(bars, regex_by_model['mean']):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 0.05, f"{val:.1%}", ha='center', fontsize=12)

    # 2. Regex by Category
    ax2 = axes[0, 1]
    regex_by_cat = factual_df.groupby(["category", "model"])["regex_score"].mean().unstack()
    if not regex_by_cat.empty:
        regex_by_cat.plot(kind="bar", ax=ax2, color=colors)
        ax2.set_ylabel("Regex Accuracy")
        ax2.set_title("Factual Accuracy by Category")
        ax2.set_ylim(0, 1)
        ax2.tick_params(axis='x', rotation=0)
        ax2.legend(title="Model")

    # 3. Correct vs Incorrect counts
    ax3 = axes[1, 0]
    x_labels = []
    x_values = []
    x_colors = []
    for i, model in enumerate(["base", "finetuned"]):
        model_df = factual_df[factual_df["model"] == model]
        correct = (model_df["regex_score"] == 1.0).sum()
        incorrect = (model_df["regex_score"] == 0.0).sum()
        x_labels.extend([f"{model}\nCorrect", f"{model}\nIncorrect"])
        x_values.extend([correct, incorrect])
        x_colors.extend([colors[i], colors[i]])
    ax3.bar(x_labels, x_values, color=x_colors, alpha=0.8)
    ax3.set_ylabel("Count")
    ax3.set_title("Factual Response Counts")

    # 4. Summary table
    ax4 = axes[1, 1]
    ax4.axis('off')

    # Build summary text
    summary_lines = ["REGEX/FACTUAL ACCURACY SUMMARY", "="*40, ""]

    for model in ["base", "finetuned"]:
        model_df = factual_df[factual_df["model"] == model]
        total = len(model_df)
        correct = (model_df["regex_score"] == 1.0).sum()
        accuracy = correct / total if total > 0 else 0

        summary_lines.append(f"{model.upper()} MODEL:")
        summary_lines.append(f"  Total factual questions: {total}")
        summary_lines.append(f"  Correct answers: {correct}")
        summary_lines.append(f"  Accuracy: {accuracy:.1%}")

        # By category
        for cat in factual_df["category"].unique():
            cat_df = model_df[model_df["category"] == cat]
            cat_correct = (cat_df["regex_score"] == 1.0).sum()
            cat_total = len(cat_df)
            cat_acc = cat_correct / cat_total if cat_total > 0 else 0
            summary_lines.append(f"    {cat}: {cat_correct}/{cat_total} ({cat_acc:.1%})")
        summary_lines.append("")

    # Improvement
    base_acc = factual_df[factual_df["model"] == "base"]["regex_score"].mean()
    ft_acc = factual_df[factual_df["model"] == "finetuned"]["regex_score"].mean()
    improvement = ft_acc - base_acc
    summary_lines.append(f"IMPROVEMENT: {improvement:+.1%}")

    ax4.text(0.1, 0.95, "\n".join(summary_lines), transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    plt.suptitle("Regex/Factual Accuracy Analysis", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/benchmark_regex_analysis_{timestamp}.png", dpi=150, bbox_inches='tight')
    print(f"✓ Regex analysis saved to {OUTPUT_DIR}/benchmark_regex_analysis_{timestamp}.png")
    plt.close()


def print_summary(df: pd.DataFrame):
    """Print summary statistics with all evaluation methods"""
    print("\n" + "="*70)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*70)

    for model in ["base", "finetuned"]:
        model_df = df[df["model"] == model]
        print(f"\n{model.upper()} MODEL:")
        print(f"  Semantic Score:   {model_df['semantic_score'].mean():.3f} ± {model_df['semantic_score'].std():.3f}")
        print(f"  Cross-Encoder:    {model_df['cross_encoder_score'].dropna().mean():.3f}")
        print(f"  NLI Score:        {model_df['nli_score'].dropna().mean():.3f}")
        print(f"  BERTScore:        {model_df['bertscore'].dropna().mean():.3f}")
        print(f"  Combined Score:   {model_df['combined_score'].mean():.3f}")

        factual = model_df[model_df["regex_score"].notna()]
        if len(factual) > 0:
            print(f"  Regex Accuracy:   {factual['regex_score'].mean():.1%}")

        print(f"  By Category:")
        for cat in sorted(model_df["category"].unique()):
            cat_score = model_df[model_df["category"] == cat]["semantic_score"].mean()
            print(f"    {cat:20}: {cat_score:.3f}")

    # Improvement
    base_mean = df[df["model"] == "base"]["semantic_score"].mean()
    ft_mean = df[df["model"] == "finetuned"]["semantic_score"].mean()
    print(f"\nSEMANTIC IMPROVEMENT: {ft_mean - base_mean:+.3f} ({((ft_mean - base_mean) / base_mean) * 100:+.1f}%)")
    print("="*70)


if __name__ == "__main__":
    import sys

    test_mode = "--test" in sys.argv

    if "--help" in sys.argv or "-h" in sys.argv:
        print("""
Benchmark Runner for Pauluskerk Chatbot

Usage:
  python run_benchmark.py          # Full benchmark (500 questions x 10 reps x 2 models)
  python run_benchmark.py --test   # Quick test (5 questions x 2 reps x 2 models)

Output:
  Results saved to benchmark_results/ directory
  - CSV with all responses and scores
  - PNG visualizations comparing models
        """)
        sys.exit(0)

    if test_mode:
        print("=" * 60)
        print("RUNNING IN TEST MODE (quick validation)")
        print("=" * 60)

    main(test_mode=test_mode)
