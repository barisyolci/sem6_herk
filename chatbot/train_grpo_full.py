"""
GRPO full fine-tuning for Qwen3-4B on homeless assistance preference data.
"""

import argparse
import json
from pathlib import Path

import mlflow
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback, set_seed
from trl import GRPOConfig, GRPOTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GRPO full fine-tuning")
    parser.add_argument("--model_name", default="Qwen/Qwen3-4B")
    parser.add_argument("--train_file", default="data/train.jsonl")
    parser.add_argument("--val_file", default="data/val.jsonl")
    parser.add_argument("--output_dir", default="outputs/grpo_full")
    parser.add_argument("--experiment_name", default="qwen3-4b-grpo-safety")
    parser.add_argument("--run_name", default="grpo_full")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_prompt_length", type=int, default=512)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--save_total_limit", type=int, default=2)

    parser.add_argument("--early_stopping_patience", type=int, default=3)
    parser.add_argument("--early_stopping_threshold", type=float, default=0.002)

    parser.add_argument("--max_steps", type=int, default=1000)
    return parser.parse_args()


def response_logprob(model, tokenizer, prompt, response, max_prompt_length, max_length, device):
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    response_ids = tokenizer(response, add_special_tokens=False).input_ids

    prompt_ids = prompt_ids[-max_prompt_length:]
    max_response_len = max_length - len(prompt_ids)
    if max_response_len <= 0:
        return float("-inf")

    response_ids = response_ids[:max_response_len]
    if not response_ids:
        return float("-inf")

    input_ids = torch.tensor([prompt_ids + response_ids], device=device)
    labels = torch.tensor([[-100] * len(prompt_ids) + response_ids], device=device)
    attention_mask = torch.ones_like(input_ids, device=device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
    return -loss.item() * len(response_ids)


def preference_win_rate(model, tokenizer, eval_dataset, max_prompt_length, max_length):
    model.eval()
    device = next(model.parameters()).device
    wins = 0
    total = len(eval_dataset)

    for row in eval_dataset:
        chosen_score = response_logprob(
            model,
            tokenizer,
            row["prompt"],
            row["chosen"],
            max_prompt_length,
            max_length,
            device,
        )
        rejected_score = response_logprob(
            model,
            tokenizer,
            row["prompt"],
            row["rejected"],
            max_prompt_length,
            max_length,
            device,
        )
        if chosen_score > rejected_score:
            wins += 1

    return wins / total if total else 0.0


def resolve_path(base_dir: Path, path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return base_dir / path


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    project_root = Path(__file__).resolve().parent
    mlflow.set_tracking_uri(f\"file:{project_root / 'mlruns'}\")
    mlflow.set_experiment(args.experiment_name)
    train_file = resolve_path(project_root, args.train_file)
    val_file = resolve_path(project_root, args.val_file)
    output_dir = resolve_path(project_root, args.output_dir)

    train_dataset = load_dataset("json", data_files=str(train_file), split="train")
    val_dataset = load_dataset("json", data_files=str(val_file), split="train")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map="auto",
    )

    config = GRPOConfig(
        output_dir=str(output_dir),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        beta=args.beta,
        max_prompt_length=args.max_prompt_length,
        max_length=args.max_length,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        evaluation_strategy="steps",
        save_strategy="steps",
        max_steps=args.max_steps,
        load_best_model_at_end=True,
        metric_for_best_model="pref_win_rate",
        greater_is_better=True,
        report_to="none",
        remove_unused_columns=False,
    )

    def compute_metrics(_):
        win_rate = preference_win_rate(
            model, tokenizer, val_dataset, args.max_prompt_length, args.max_length
        )
        return {"pref_win_rate": win_rate}

    trainer = GRPOTrainer(
        model=model,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.add_callback(
        EarlyStoppingCallback(
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_threshold=args.early_stopping_threshold,
        )
    )

    run_config = {
        "variant": "grpo_full",
        "model_name": args.model_name,
        "train_file": str(train_file),
        "val_file": str(val_file),
        "training": {
            "learning_rate": args.learning_rate,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "per_device_eval_batch_size": args.per_device_eval_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "max_prompt_length": args.max_prompt_length,
            "max_length": args.max_length,
            "beta": args.beta,
            "eval_steps": args.eval_steps,
            "logging_steps": args.logging_steps,
            "save_steps": args.save_steps,
            "save_total_limit": args.save_total_limit,
            "max_steps": args.max_steps,
        },
        "early_stopping": {
            "patience": args.early_stopping_patience,
            "threshold": args.early_stopping_threshold,
            "metric": "pref_win_rate",
        },
        "seed": args.seed,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "run_config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_params({
            "variant": "grpo_full",
            "model_name": args.model_name,
            "train_size": len(train_dataset),
            "val_size": len(val_dataset),
        })
        mlflow.log_params(run_config["training"])
        mlflow.log_params({
            "early_stopping_patience": args.early_stopping_patience,
            "early_stopping_threshold": args.early_stopping_threshold,
        })

        trainer.train()
        final_metrics = trainer.evaluate()

        mlflow.log_metrics({
            "final_pref_win_rate": final_metrics.get("eval_pref_win_rate", 0.0),
            "train_steps": trainer.state.global_step,
        })

        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))

        mlflow.log_artifact(str(config_path))
        mlflow.log_artifacts(str(output_dir), artifact_path="model")


if __name__ == "__main__":
    main()
