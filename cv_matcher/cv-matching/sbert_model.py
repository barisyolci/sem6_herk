from collections import Counter
from typing import List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

import re


class SBERTScorer:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name, device="cpu")

    def score_cvs(self, job_description: str, cv_texts: List[str]) -> List[float]:
        embeddings = self.model.encode(
            [job_description] + cv_texts,
            normalize_embeddings=True,
        )
        job_vec = embeddings[0:1]
        cv_vecs = embeddings[1:]
        sims = util.cos_sim(job_vec, cv_vecs)[0].cpu().numpy()
        sims = (sims + 1.0) / 2.0
        sims = np.clip(sims, 0.0, 1.0)
        return sims.tolist()

    def explain_pair(
        self, job_description: str, cv_text: str, top_n: int = 10
    ) -> Tuple[float, List[Tuple[str, int]]]:
        embeddings = self.model.encode(
            [job_description, cv_text],
            normalize_embeddings=True,
        )
        job_vec = embeddings[0:1]
        cv_vec = embeddings[1:2]
        similarity = float(util.cos_sim(job_vec, cv_vec)[0][0].cpu().numpy())
        normalized_similarity = float((similarity + 1.0) / 2.0)

        job_tokens = self._extract_keywords(job_description)
        cv_tokens = self._extract_keywords(cv_text)
        job_counts = Counter(job_tokens)
        cv_counts = Counter(cv_tokens)
        common = job_counts.keys() & cv_counts.keys()
        ranked = sorted(
            ((token, job_counts[token] + cv_counts[token]) for token in common),
            key=lambda item: item[1],
            reverse=True,
        )
        return normalized_similarity, ranked[:top_n]

    @staticmethod
    def _extract_keywords(text: str) -> List[str]:
        tokens = re.findall(r"[a-zA-Z]+", text.lower())
        return [token for token in tokens if token not in ENGLISH_STOP_WORDS]
