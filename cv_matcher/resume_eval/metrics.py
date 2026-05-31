import csv
import os
import random
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from text_utils import load_resumes
from semantic_engine import SemanticScorer
from llm_evaluator import LLMJudge
from vector_scorer import compute_match_scores as tfidf_compute

load_dotenv()


def initialize_reproducibility(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_target_description(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def load_ground_truth(path: str) -> Dict[str, str]:
    mapping = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mapping[row["candidate_id"].strip()] = row["relevance"].strip()
    return mapping


def generate_ranking(
    candidate_ids: List[str], 
    scores: List[float], 
    truth: Dict[str, str]
) -> pd.DataFrame:
    df = pd.DataFrame({"candidate_id": candidate_ids, "match_score": scores})
    df["relevance"] = df["candidate_id"].map(truth).fillna("unknown")
    df = df.sort_values(by=["match_score", "candidate_id"], ascending=[False, True]).reset_index(drop=True)
    df.insert(0, "position", range(1, len(df) + 1))
    return df


def hit_rate_at_k(ranked_ids: List[str], truth: Dict[str, str], k: int) -> float:
    top_k = ranked_ids[:k]
    found = any(truth.get(cid) == "relevant" for cid in top_k)
    return 100.0 if found else 0.0


def precision_top_k(ranked_ids: List[str], truth: Dict[str, str], k: int) -> float:
    top_k = ranked_ids[:k]
    relevant = sum(truth.get(cid) in {"relevant", "partially_relevant"} for cid in top_k)
    return (relevant / float(k)) * 100.0


def avg_position_relevant(ranked_ids: List[str], truth: Dict[str, str]) -> float:
    positions = [i + 1 for i, cid in enumerate(ranked_ids) if truth.get(cid) == "relevant"]
    if not positions:
        return float("nan")
    return float(np.mean(positions))


def calculate_metrics(ranking: pd.DataFrame, truth: Dict[str, str]) -> Dict[str, float]:
    ranked_ids = ranking["candidate_id"].tolist()
    return {
        "hit_rate_k1": hit_rate_at_k(ranked_ids, truth, 1),
        "hit_rate_k3": hit_rate_at_k(ranked_ids, truth, 3),
        "hit_rate_k5": hit_rate_at_k(ranked_ids, truth, 5),
        "precision_k3": precision_top_k(ranked_ids, truth, 3),
        "avg_pos_relevant": avg_position_relevant(ranked_ids, truth),
    }


def execute_evaluation(
    data_path: str,
    enable_llm: bool = False,
    semantic_model: Optional[SemanticScorer] = None,
    llm_model: Optional[LLMJudge] = None,
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    
    initialize_reproducibility()
    
    target_path = os.path.join(data_path, "target_role.txt")
    labels_path = os.path.join(data_path, "ground_truth.csv")
    resumes_path = os.path.join(data_path, "resumes")
    
    target_text = load_target_description(target_path)
    candidate_ids, candidate_texts = load_resumes(resumes_path)
    ground_truth = load_ground_truth(labels_path)
    
    semantic_model = semantic_model or SemanticScorer()
    llm_model = llm_model or LLMJudge(use_api=enable_llm)
    
    all_scores = {
        "tfidf": tfidf_compute(target_text, candidate_texts),
        "semantic": semantic_model.compute_scores(target_text, candidate_texts),
        "llm": llm_model.compute_scores(target_text, candidate_texts),
    }
    
    rankings = {}
    summary_rows = []
    
    for method_name, scores in all_scores.items():
        ranking_df = generate_ranking(candidate_ids, scores, ground_truth)
        metrics = calculate_metrics(ranking_df, ground_truth)
        metrics["approach"] = method_name
        rankings[method_name] = ranking_df
        summary_rows.append(metrics)
    
    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df[[
        "approach", "hit_rate_k1", "hit_rate_k3", "hit_rate_k5", 
        "precision_k3", "avg_pos_relevant"
    ]]
    
    output_dir = os.path.join(os.path.dirname(data_path), "output")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "evaluation_report.csv")
    summary_df.to_csv(output_path, index=False)
    
    return rankings, summary_df


def execute_evaluation_from_input(
    target_text: str,
    candidate_ids: List[str],
    candidate_texts: List[str],
    ground_truth: Dict[str, str],
    enable_llm: bool = False,
    semantic_model: Optional[SemanticScorer] = None,
    llm_model: Optional[LLMJudge] = None,
    output_dir: Optional[str] = None,
    persist_results: bool = False,
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    
    initialize_reproducibility()
    
    semantic_model = semantic_model or SemanticScorer()
    llm_model = llm_model or LLMJudge(use_api=enable_llm)
    
    all_scores = {
        "tfidf": tfidf_compute(target_text, candidate_texts),
        "semantic": semantic_model.compute_scores(target_text, candidate_texts),
        "llm": llm_model.compute_scores(target_text, candidate_texts),
    }
    
    rankings = {}
    summary_rows = []
    
    for method_name, scores in all_scores.items():
        ranking_df = generate_ranking(candidate_ids, scores, ground_truth)
        metrics = calculate_metrics(ranking_df, ground_truth)
        metrics["approach"] = method_name
        rankings[method_name] = ranking_df
        summary_rows.append(metrics)
    
    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df[[
        "approach", "hit_rate_k1", "hit_rate_k3", "hit_rate_k5",
        "precision_k3", "avg_pos_relevant"
    ]]
    
    if persist_results:
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(__file__), "output")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "evaluation_report.csv")
        summary_df.to_csv(output_path, index=False)
    
    return rankings, summary_df