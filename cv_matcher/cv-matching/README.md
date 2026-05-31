# CV Matching Benchmark

A reproducible benchmark comparing three CV-job matching approaches:

1. TF-IDF + cosine similarity
2. Sentence-BERT (all-MiniLM-L6-v2) + cosine similarity
3. LLM scoring with Groq using `openai/gpt-oss-120b` (pairwise job description vs CV)

## Setup

**Python version:** Python 3.13 is supported with the pinned dependencies below. If you see dependency conflicts, make sure `requirements.txt` is installed exactly as-is. If you see build errors, upgrade pip first:

```bash
python -m pip install --upgrade pip
```

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If you see `TypeError: Client.__init__() got an unexpected keyword argument 'proxies'`, reinstall to ensure `httpx==0.27.2` is used.

If you want to use Groq for LLM scoring, set the API key:

```bash
copy .env.example .env
# then edit .env and add your Groq key
```

The app loads `.env` automatically via `python-dotenv`.

## Run

```bash
streamlit run app.py
```

The app will load the dataset in `data/`, score all CVs with each model, and write benchmark results to:

`results/benchmark_results.csv`

## Upload in the UI

The Streamlit app also supports uploads:

- **One job vs many CVs:** upload a job description and multiple CVs, optionally with a `labels.csv` file to compute benchmark percentages.
- **One job vs one CV:** upload a single CV and a single job description to get 1:1 scores from all three models.

## Supported CV formats

Place CV files in `data/cvs/` as `.txt`, `.pdf`, or `.docx`. The loader extracts text from each file and uses the filename (without extension) as the `cv_id` to match labels in `data/labels.csv`.

## Benchmarks

- **Top-K Accuracy (K = 1, 3, 5):** checks if at least one *relevant* CV appears in the top K (reported as 0 or 100).
- **Precision@3:** counts `relevant` + `partially_relevant` in the top 3, divided by 3, reported as a percentage.
- **Mean Rank of Relevant CVs:** average rank (1-based) of all CVs labeled `relevant`.

## Reproducibility Notes

- TF-IDF and SBERT scoring are deterministic for fixed inputs.
- LLM scoring uses Groq with `temperature=0`. If `GROQ_API_KEY` is not set, a deterministic mock scorer is used to keep the benchmark runnable offline.
