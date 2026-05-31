# Qwen3-4B GRPO (Arabic)

Arabic-only GRPO training and evaluation with required Arabic data generation.

## 1) Install

```bash
pip install -r requirements.txt
pip install flask python-dotenv requests
```

## 2) Set API key

Create `/home/dealers/mo_4b_finetune/.env`:

```bash
OPENROUTER_KEY=your_key_here
# Optional
OPENROUTER_MODEL=deepseek/deepseek-chat
OPENROUTER_MAX_TOKENS=800
OPENROUTER_TEMPERATURE=0.25
TARGET_CONVERSATIONS=2000
PARALLEL_WORKERS=8
PORT=5000
```

## 3) Generate Arabic data (required)

```bash
python app_arabic.py
```

Start generation (set total with `target`):

```bash
curl -X POST http://localhost:5000/start -H "Content-Type: application/json" -d '{"target": 200}'
# or easier:
curl "http://localhost:5000/start?target=200"
```

Merge when done:

```bash
curl -X POST http://localhost:5000/merge
```

Progress (0..n):

```bash
curl http://localhost:5000/progress
```

Outputs:
- `dataset_raw_<run_id>.json`
- `dataset_rl_training_<run_id>.jsonl`
- `dataset_raw_merged.json`
- `dataset_rl_training_merged.jsonl`

Files are written continuously during generation. `dataset_raw_<run_id>.json` is finalized with a closing bracket when the run stops.

## 4) Create GRPO preference data (Arabic)

GRPO needs JSONL lines with:

```json
{"prompt": "...", "chosen": "...", "rejected": "..."}
```

Place Arabic files here:
- `data/train.jsonl`
- `data/val.jsonl`

Also prepare Arabic benchmarks:
- `data/benchmark_harmful.jsonl`
- `data/benchmark_benign.jsonl`
- `data/benchmark_borderline.jsonl`
- `data/benchmark_prefs.jsonl`

## 5) Train (GRPO only)

Full:
```bash
python train_grpo_full.py
```

LoRA:
```bash
python train_grpo_lora.py
```

## 6) Evaluate

```bash
python evaluate.py
```

Results:
- `benchmarks.csv`
- MLflow runs in `./mlruns`

Open MLflow:
```bash
mlflow ui --backend-store-uri /home/dealers/mo_4b_finetune/mlruns
```

## Benchmarking folder

Full benchmark pipeline is now in:

- `/home/dealers/mo_4b_finetune/benchmarking`

## 7) UI (Open WebUI)

Install Open WebUI:

```bash
pip install open-webui
```

Start Ollama with the project model store (only models created here will show up in the UI):

```bash
/home/dealers/open_webui/scripts/run_ollama.sh
```

If you already run Ollama as a service, stop it and restart with the script so it uses the project store.

Add a model to the shared folder (example with a local GGUF):

```bash
mkdir -p /home/dealers/open_webui/team_models
cd /home/dealers/open_webui/team_models
cat > Modelfile <<'EOF'
FROM ./my_model.gguf
EOF
OLLAMA_MODELS=/home/dealers/open_webui/team_models/ollama_store ollama create my-model -f Modelfile
```

Start the UI:

```bash
/home/dealers/open_webui/scripts/run_open_webui.sh
```

Open `http://localhost:8080` in a browser. The model list will only include models created in `/home/dealers/open_webui/team_models/ollama_store`.
