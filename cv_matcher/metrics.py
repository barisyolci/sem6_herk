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
from vector_scorer import calculate_alignment as tfidf_score

load_dotenv()

def _seed_all(s: int = 42):
    random.seed(s)
    np.random.seed(s)

def _load_target(p: str) -> str:
    with open(p, "r", encoding="utf-8") as f:
        return f.read().strip()

def _load_labels(p: str) -> Dict[str, str]:
    m = {}
    with open(p, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cid = row["candidate_id"].strip()
            rel = row["relevance"].strip()
            if cid:
                m[cid] = rel
    return m

def _build_ranking(ids: List[str], scores: List[float], truth: Dict[str, str]) -> pd.DataFrame:
    df = pd.DataFrame({"candidate_id": ids, "match_score": scores})
    df["relevance"] = df["candidate_id"].map(truth).fillna("unknown")
    df = df.sort_values(by=["match_score", "candidate_id"], ascending=[False, True]).reset_index(drop=True)
    df.insert(0, "position", range(1, len(df) + 1))
    return df

def _hit_k(ranked: List[str], truth: Dict[str, str], k: int) -> float:
    top = ranked[:k]
    return 100.0 if any(truth.get(c) == "relevant" for c in top) else 0.0

def _prec_k(ranked: List[str], truth: Dict[str, str], k: int) -> float:
    top = ranked[:k]
    cnt = sum(truth.get(c) in {"relevant", "partially_relevant"} for c in top)
    return (cnt / float(k)) * 100.0

def _avg_pos(ranked: List[str], truth: Dict[str, str]) -> float:
    pos = [i+1 for i, c in enumerate(ranked) if truth.get(c) == "relevant"]
    return float(np.mean(pos)) if pos else float("nan")

def _calc_metrics(ranking: pd.DataFrame, truth: Dict[str, str]) -> Dict[str, float]:
    ids = ranking["candidate_id"].tolist()
    return {
        "hit_rate_k1": _hit_k(ids, truth, 1),
        "hit_rate_k3": _hit_k(ids, truth, 3),
        "hit_rate_k5": _hit_k(ids, truth, 5),
        "precision_k3": _prec_k(ids, truth, 3),
        "avg_pos_relevant": _avg_pos(ids, truth),
    }

def run_eval(data_dir: str, use_llm: bool = False, sem_mod=None, llm_mod=None):
    _seed_all()
    tgt_p = os.path.join(data_dir, "target_role.txt")
    lbl_p = os.path.join(data_dir, "ground_truth.csv")
    res_p = os.path.join(data_dir, "resumes")
    tgt_txt = _load_target(tgt_p)
    cids, ctexts = load_resumes(res_p)
    truth = _load_labels(lbl_p)
    sem_mod = sem_mod or SemanticScorer()
    llm_mod = llm_mod or LLMJudge()
    all_scores = {
        "tfidf": tfidf_score(tgt_txt, ctexts),
        "semantic": sem_mod.compute_scores(tgt_txt, ctexts),
        "llm": llm_mod.compute_scores(tgt_txt, ctexts),
    }
    rankings = {}
    rows = []
    for name, scs in all_scores.items():
        rk = _build_ranking(cids, scs, truth)
        mets = _calc_metrics(rk, truth)
        mets["approach"] = name
        rankings[name] = rk
        rows.append(mets)
    summ = pd.DataFrame(rows)[["approach", "hit_rate_k1", "hit_rate_k3", "hit_rate_k5", "precision_k3", "avg_pos_relevant"]]
    out_dir = os.path.join(os.path.dirname(data_dir), "output")
    os.makedirs(out_dir, exist_ok=True)
    out_p = os.path.join(out_dir, "evaluation_report.csv")
    summ.to_csv(out_p, index=False)
    return rankings, summ

def run_eval_input(tgt: str, cids: List[str], ctexts: List[str], truth: Dict[str, str], use_llm: bool = False, sem_mod=None, llm_mod=None, out_dir: Optional[str] = None, save: bool = False):
    _seed_all()
    sem_mod = sem_mod or SemanticScorer()
    llm_mod = llm_mod or LLMJudge()
    all_scores = {
        "tfidf": tfidf_score(tgt, ctexts),
        "semantic": sem_mod.compute_scores(tgt, ctexts),
        "llm": llm_mod.compute_scores(tgt, ctexts),
    }
    rankings = {}
    rows = []
    for name, scs in all_scores.items():
        rk = _build_ranking(cids, scs, truth)
        mets = _calc_metrics(rk, truth)
        mets["approach"] = name
        rankings[name] = rk
        rows.append(mets)
    summ = pd.DataFrame(rows)[["approach", "hit_rate_k1", "hit_rate_k3", "hit_rate_k5", "precision_k3", "avg_pos_relevant"]]
    if save:
        if out_dir is None:
            out_dir = os.path.join(os.path.dirname(__file__), "output")
        os.makedirs(out_dir, exist_ok=True)
        out_p = os.path.join(out_dir, "evaluation_report.csv")
        summ.to_csv(out_p, index=False)
    return rankings, summ