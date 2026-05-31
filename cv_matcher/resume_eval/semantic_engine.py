from collections import Counter
from typing import List, Tuple
import numpy as np
import re
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


class SemanticScorer:

    def __init__(self, model_identifier: str = "all-MiniLM-L6-v2") -> None:
        self.encoder = SentenceTransformer(model_identifier, device="cpu")
    
    def compute_scores(self, target: str, candidates: List[str]) -> List[float]:
        embeddings = self.encoder.encode(
            [target] + candidates,
            normalize_embeddings=True,
        )
        target_emb = embeddings[0:1]
        candidate_embs = embeddings[1:]
        similarities = util.cos_sim(target_emb, candidate_embs)[0].cpu().numpy()
        normalized = (similarities + 1.0) / 2.0
        return np.clip(normalized, 0.0, 1.0).tolist()
    
    def explain_match(
        self, target: str, candidate: str, top_terms: int = 10
    ) -> Tuple[float, List[Tuple[str, int]]]:
        embeddings = self.encoder.encode(
            [target, candidate],
            normalize_embeddings=True,
        )
        sim = float(util.cos_sim(embeddings[0:1], embeddings[1:2])[0][0].cpu().numpy())
        normalized_sim = float((sim + 1.0) / 2.0)
        
        target_keywords = self._extract_keywords(target)
        candidate_keywords = self._extract_keywords(candidate)
        target_counts = Counter(target_keywords)
        candidate_counts = Counter(candidate_keywords)
        
        shared = target_counts.keys() & candidate_counts.keys()
        ranked = sorted(
            ((kw, target_counts[kw] + candidate_counts[kw]) for kw in shared),
            key=lambda x: x[1],
            reverse=True
        )
        return normalized_sim, ranked[:top_terms]
    
    @staticmethod
    def _extract_keywords(text: str) -> List[str]:
        tokens = re.findall(r"[a-zA-Z]+", text.lower())
        return [t for t in tokens if t not in ENGLISH_STOP_WORDS]