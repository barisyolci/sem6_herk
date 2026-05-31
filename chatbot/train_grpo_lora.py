import argparse
import json
import os
import random
import time
from pathlib import Path

import mlflow
import requests
import torch
from datasets import Dataset
from dotenv import load_dotenv
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, set_seed
from trl import GRPOConfig, GRPOTrainer

load_dotenv()

ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)

OPENROUTER_KEY = (os.getenv("OPENROUTER_KEY") or "").strip()
OPENROUTER_URL = (os.getenv("OPENROUTER_URL") or "https://openrouter.ai/api/v1/chat/completions").strip()
JUDGE_MODEL_DEFAULT = (os.getenv("JUDGE_MODEL") or "meta-llama/llama-3.1-8b-instruct").strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GRPO LoRA reward-based fine-tuning")
    parser.add_argument("--model_name", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--raw_file", default="data/dataset_raw_merged.json")
    parser.add_argument("--output_dir", default="outputs/grpo_lora")
    parser.add_argument("--experiment_name", default="llama3-8b-grpo-safety")
    parser.add_argument("--run_name", default="grpo_lora_reward")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_prompt_length", type=int, default=256)
    parser.add_argument("--max_completion_length", type=int, default=256)
    parser.add_argument("--num_generations", type=int, default=2)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=300)

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument("--judge_model", default=JUDGE_MODEL_DEFAULT)
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--eval_size", type=int, default=64)
    parser.add_argument("--early_stopping_patience", type=int, default=3)
    parser.add_argument("--early_stopping_threshold", type=float, default=0.05)
    parser.add_argument("--bf16", action="store_true", default=None)
    parser.add_argument("--no-bf16", action="store_false", dest="bf16")
    parser.add_argument("--tf32", action="store_true", default=None)
    parser.add_argument("--no-tf32", action="store_false", dest="tf32")
    return parser.parse_args()


def resolve_path(base_dir: Path, path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return base_dir / path


def resolve_raw_path(project_root: Path, raw_file: str) -> Path:
    candidate = resolve_path(project_root, raw_file)
    if candidate.exists():
        return candidate
    data_dir = project_root / "data"
    matches = sorted(data_dir.glob("dataset_raw_*.json"))
    if matches:
        return matches[-1]
    raise FileNotFoundError(f"No dataset_raw_*.json found in {data_dir}")


def load_prompts(raw_path: Path) -> list[str]:
    with raw_path.open("r", encoding="utf-8") as f:
        raw_data = json.load(f)

    prompts: list[str] = []
    for conv in raw_data:
        turns = conv.get("turns") or []
        if not turns:
            continue
        first = turns[0]
        if first.get("role") != "homeless":
            continue
        prompt = (
            "You are a social support assistant helping people experiencing homelessness. "
            "Be empathetic, practical, and do not invent information.\n\n"
            f"Person: {first.get('content', '')}\n\n"
            "Assistant:"
        )
        prompts.append(prompt)

    return prompts


def split_prompts(prompts: list[str], eval_size: int, seed: int) -> tuple[list[str], list[str]]:
    if eval_size <= 0 or eval_size >= len(prompts):
        return prompts, []
    rng = random.Random(seed)
    indices = list(range(len(prompts)))
    rng.shuffle(indices)
    eval_indices = set(indices[:eval_size])
    train_prompts = [p for i, p in enumerate(prompts) if i not in eval_indices]
    eval_prompts = [p for i, p in enumerate(prompts) if i in eval_indices]
    return train_prompts, eval_prompts


def call_judge_model(prompt: str, response: str, judge_model: str) -> float:
    if not OPENROUTER_KEY:
        raise ValueError("OPENROUTER_KEY is missing. Set it in .env")

    judge_prompt = (
        "You are evaluating a response from a social support assistant providing help to people experiencing homelessness.\n"
        "Rate the response from 0 to 10 based on: empathy, practicality, safety, and clarity.\n\n"
        f"Person's message:\n{prompt}\n\n"
        f"Assistant's response:\n{response}\n\n"
        "Answer with only a number from 0 to 10."
    )

    headers = {
        "Authorization": f"Bearer {OPENROUTER_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": judge_model,
        "messages": [{"role": "user", "content": judge_prompt}],
        "max_tokens": 10,
        "temperature": 0.1,
        "stream": False,
    }

    last_err = None
    for attempt in range(3):
        try:
            resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            score_text = resp.json()["choices"][0]["message"]["content"].strip()
            score = float("".join(c for c in score_text if c.isdigit() or c == ".") or "0")
            return max(0.0, min(10.0, score))
        except Exception as e:
            last_err = e
            time.sleep(1.5 * (attempt + 1))

    raise RuntimeError(f"Judge model call failed: {last_err}")


def reward_function(completions, prompts=None, judge_model: str = JUDGE_MODEL_DEFAULT, **kwargs):
    rewards = []
    if prompts is None:
        prompts = ["" for _ in completions]

    for prompt, completion in zip(prompts, completions):
        prompt_text = prompt.get("content", str(prompt)) if isinstance(prompt, dict) else str(prompt)
        completion_text = completion.get("content", str(completion)) if isinstance(completion, dict) else str(completion)
        score = call_judge_model(prompt_text, completion_text, judge_model)
        rewards.append((score - 5.0) / 5.0)

    return rewards


def eval_reward_score(
    model,
    tokenizer,
    prompts: list[str],
    judge_model: str,
    max_prompt_length: int,
    max_completion_length: int,
) -> float:
    if not prompts:
        return 0.0

    model.eval()
    device = next(model.parameters()).device
    scores = []
    with torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_prompt_length,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            output_ids = model.generate(
                **inputs,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                num_beams=1,
                max_new_tokens=max_completion_length,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            generated = output_ids[0][inputs["input_ids"].shape[1]:]
            response = tokenizer.decode(generated, skip_special_tokens=True).strip()
            score = call_judge_model(prompt, response, judge_model)
            scores.append(score)

    return sum(scores) / len(scores)


class RewardEarlyStopCallback(TrainerCallback):
    def __init__(
        self,
        eval_prompts: list[str],
        judge_model: str,
        eval_steps: int,
        patience: int,
        threshold: float,
        max_prompt_length: int,
        max_completion_length: int,
    ) -> None:
        self.eval_prompts = eval_prompts
        self.judge_model = judge_model
        self.eval_steps = max(1, eval_steps)
        self.patience = max(1, patience)
        self.threshold = threshold
        self.max_prompt_length = max_prompt_length
        self.max_completion_length = max_completion_length
        self.best_score = None
        self.bad_rounds = 0

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 0 or state.global_step % self.eval_steps != 0:
            return control
        model = kwargs.get("model")
        tokenizer = kwargs.get("tokenizer")
        if model is None or tokenizer is None:
            return control

        score = eval_reward_score(
            model,
            tokenizer,
            self.eval_prompts,
            self.judge_model,
            self.max_prompt_length,
            self.max_completion_length,
        )
        mlflow.log_metric("eval_reward_score", score, step=state.global_step)

        if self.best_score is None or score > self.best_score + self.threshold:
            self.best_score = score
            self.bad_rounds = 0
        else:
            self.bad_rounds += 1
            if self.bad_rounds >= self.patience:
                control.should_training_stop = True
        return control


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    project_root = Path(__file__).resolve().parent
    mlflow.set_tracking_uri(f"file:{project_root / 'mlruns'}")
    mlflow.set_experiment(args.experiment_name)

    raw_path = resolve_raw_path(project_root, args.raw_file)
    output_dir = resolve_path(project_root, args.output_dir)

    all_prompts = load_prompts(raw_path)
    train_prompts, eval_prompts = split_prompts(all_prompts, args.eval_size, args.seed)
    train_dataset = Dataset.from_list([{"prompt": p} for p in train_prompts])

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    cuda_available = torch.cuda.is_available()
    if args.bf16 is None:
        use_bf16 = cuda_available and torch.cuda.is_bf16_supported()
    else:
        use_bf16 = args.bf16
    if args.tf32 is None:
        use_tf32 = cuda_available
    else:
        use_tf32 = args.tf32

    if use_tf32 and cuda_available:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    torch_dtype = torch.bfloat16 if use_bf16 else torch.float16
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map="auto",
    )

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(base_model, lora_config)

    config = GRPOConfig(
        output_dir=str(output_dir),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        beta=args.beta,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        max_steps=args.max_steps,
        report_to="none",
        remove_unused_columns=False,
        bf16=use_bf16 if cuda_available else False,
        fp16=(cuda_available and not use_bf16),
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
    )

    trainer = GRPOTrainer(
        model=model,
        args=config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        reward_funcs=[lambda completions, prompts=None, **kw: reward_function(
            completions, prompts=prompts, judge_model=args.judge_model
        )],
    )
    early_stop_callback = None
    if eval_prompts:
        early_stop_callback = RewardEarlyStopCallback(
            eval_prompts=eval_prompts,
            judge_model=args.judge_model,
            eval_steps=args.eval_steps,
            patience=args.early_stopping_patience,
            threshold=args.early_stopping_threshold,
            max_prompt_length=args.max_prompt_length,
            max_completion_length=args.max_completion_length,
        )
        trainer.add_callback(early_stop_callback)

    run_config = {
        "variant": "grpo_lora_reward",
        "model_name": args.model_name,
        "raw_file": str(raw_path),
        "judge_model": args.judge_model,
        "training": {
            "learning_rate": args.learning_rate,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "max_prompt_length": args.max_prompt_length,
            "max_completion_length": args.max_completion_length,
            "num_generations": args.num_generations,
            "beta": args.beta,
            "logging_steps": args.logging_steps,
            "save_steps": args.save_steps,
            "save_total_limit": args.save_total_limit,
            "max_steps": args.max_steps,
            "eval_steps": args.eval_steps,
            "eval_size": args.eval_size,
            "early_stopping_patience": args.early_stopping_patience,
            "early_stopping_threshold": args.early_stopping_threshold,
            "bf16": use_bf16 if cuda_available else False,
            "fp16": (cuda_available and not use_bf16),
            "tf32": use_tf32 if cuda_available else False,
        },
        "lora": {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        },
        "seed": args.seed,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "run_config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_params({
            "variant": "grpo_lora_reward",
            "model_name": args.model_name,
            "raw_file": str(raw_path),
            "train_size": len(train_dataset),
            "judge_model": args.judge_model,
        })
        mlflow.log_params(run_config["training"])
        mlflow.log_params(run_config["lora"])

        trainer.train()

        mlflow.log_metrics({
            "train_steps": trainer.state.global_step,
        })
        if early_stop_callback is not None:
            mlflow.log_metrics({
                "early_stop_best_reward": early_stop_callback.best_score or 0.0,
            })

        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))

        mlflow.log_artifact(str(config_path))
        mlflow.log_artifacts(str(output_dir), artifact_path="model")


if __name__ == "__main__":
    main()