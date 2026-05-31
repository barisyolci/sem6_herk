import csv
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from data_loader import load_cvs
from llm_model import LLMScorer
from sbert_model import SBERTScorer
from tfidf_model import score_cvs as tfidf_score


load_dotenv()

def set_deterministic(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_job_description(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def load_labels(path: str) -> Dict[str, str]:
    labels = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels[row["cv_id"].strip()] = row["label"].strip()
    return labels


def rank_scores(cv_ids: List[str], scores: List[float], labels: Dict[str, str]) -> pd.DataFrame:
    df = pd.DataFrame({"cv_id": cv_ids, "score": scores})
    df["label"] = df["cv_id"].map(labels).fillna("unknown")
    df = df.sort_values(by=["score", "cv_id"], ascending=[False, True]).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    return df


def top_k_accuracy(ranked_ids: List[str], labels: Dict[str, str], k: int) -> float:
    top_ids = ranked_ids[:k]
    hit = any(labels.get(cv_id) == "relevant" for cv_id in top_ids)
    return 100.0 if hit else 0.0


def precision_at_k(ranked_ids: List[str], labels: Dict[str, str], k: int) -> float:
    top_ids = ranked_ids[:k]
    good = sum(labels.get(cv_id) in {"relevant", "partially_relevant"} for cv_id in top_ids)
    return (good / float(k)) * 100.0


def mean_rank_relevant(ranked_ids: List[str], labels: Dict[str, str]) -> float:
    ranks = [i + 1 for i, cv_id in enumerate(ranked_ids) if labels.get(cv_id) == "relevant"]
    if not ranks:
        return float("nan")
    return float(np.mean(ranks))


def compute_metrics(ranking: pd.DataFrame, labels: Dict[str, str]) -> Dict[str, float]:
    ranked_ids = ranking["cv_id"].tolist()
    metrics = {
        "top1_accuracy": top_k_accuracy(ranked_ids, labels, 1),
        "top3_accuracy": top_k_accuracy(ranked_ids, labels, 3),
        "top5_accuracy": top_k_accuracy(ranked_ids, labels, 5),
        "precision_at_3": precision_at_k(ranked_ids, labels, 3),
        "mean_rank_relevant": mean_rank_relevant(ranked_ids, labels),
    }
    return metrics


def run_benchmark(
    data_dir: str,
    use_groq: bool = False,
    sbert_scorer: Optional[SBERTScorer] = None,
    llm_scorer: Optional[LLMScorer] = None,
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    set_deterministic()

    job_path = os.path.join(data_dir, "job_description.txt")
    labels_path = os.path.join(data_dir, "labels.csv")
    cvs_dir = os.path.join(data_dir, "cvs")

    job_description = load_job_description(job_path)
    cv_ids, cv_texts = load_cvs(cvs_dir)
    labels = load_labels(labels_path)

    sbert_scorer = sbert_scorer or SBERTScorer()
    llm_scorer = llm_scorer or LLMScorer(use_groq=use_groq)

    model_scores = {
        "tfidf": tfidf_score(job_description, cv_texts),
        "sbert": sbert_scorer.score_cvs(job_description, cv_texts),
        "llm": llm_scorer.score_cvs(job_description, cv_texts),
    }

    rankings: Dict[str, pd.DataFrame] = {}
    results_rows = []

    for model_name, scores in model_scores.items():
        ranking = rank_scores(cv_ids, scores, labels)
        metrics = compute_metrics(ranking, labels)
        metrics["model"] = model_name
        rankings[model_name] = ranking
        results_rows.append(metrics)

    results_df = pd.DataFrame(results_rows)
    results_df = results_df[
        ["model", "top1_accuracy", "top3_accuracy", "top5_accuracy", "precision_at_3", "mean_rank_relevant"]
    ]

    results_dir = os.path.join(os.path.dirname(data_dir), "results")
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, "benchmark_results.csv")
    results_df.to_csv(results_path, index=False)

    return rankings, results_df


def run_benchmark_from_texts(
    job_description: str,
    cv_ids: List[str],
    cv_texts: List[str],
    labels: Dict[str, str],
    use_groq: bool = False,
    sbert_scorer: Optional[SBERTScorer] = None,
    llm_scorer: Optional[LLMScorer] = None,
    results_dir: Optional[str] = None,
    save_results: bool = False,
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    set_deterministic()

    sbert_scorer = sbert_scorer or SBERTScorer()
    llm_scorer = llm_scorer or LLMScorer(use_groq=use_groq)

    model_scores = {
        "tfidf": tfidf_score(job_description, cv_texts),
        "sbert": sbert_scorer.score_cvs(job_description, cv_texts),
        "llm": llm_scorer.score_cvs(job_description, cv_texts),
    }

    rankings: Dict[str, pd.DataFrame] = {}
    results_rows = []

    for model_name, scores in model_scores.items():
        ranking = rank_scores(cv_ids, scores, labels)
        metrics = compute_metrics(ranking, labels)
        metrics["model"] = model_name
        rankings[model_name] = ranking
        results_rows.append(metrics)

    results_df = pd.DataFrame(results_rows)
    results_df = results_df[
        ["model", "top1_accuracy", "top3_accuracy", "top5_accuracy", "precision_at_3", "mean_rank_relevant"]
    ]

    if save_results:
        if results_dir is None:
            results_dir = os.path.join(os.path.dirname(__file__), "results")
        os.makedirs(results_dir, exist_ok=True)
        results_path = os.path.join(results_dir, "benchmark_results.csv")
        results_df.to_csv(results_path, index=False)

    return rankings, results_df
