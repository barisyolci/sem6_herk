from collections import Counter
from typing import List, Tuple
import numpy as np
import re
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

class SemanticScorer:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.enc = SentenceTransformer(model_name, device="cpu")

    def compute_scores(self, q: str, docs: List[str]) -> List[float]:
        embs = self.enc.encode([q] + docs, normalize_embeddings=True)
        q_emb = embs[0:1]
        d_embs = embs[1:]
        sims = util.cos_sim(q_emb, d_embs)[0].cpu().numpy()
        norm = (sims + 1.0) / 2.0
        return np.clip(norm, 0.0, 1.0).tolist()

    def explain_match(self, a: str, b: str, top_n: int = 10) -> Tuple[float, List[Tuple[str, int]]]:
        embs = self.enc.encode([a, b], normalize_embeddings=True)
        sim = float(util.cos_sim(embs[0:1], embs[1:2])[0][0].cpu().numpy())
        norm_sim = float((sim + 1.0) / 2.0)
        ka = self._kw(a)
        kb = self._kw(b)
        ca = Counter(ka)
        cb = Counter(kb)
        shared = ca.keys() & cb.keys()
        ranked = sorted(((k, ca[k] + cb[k]) for k in shared), key=lambda x: x[1], reverse=True)
        return norm_sim, ranked[:top_n]

    @staticmethod
    def _kw(t: str) -> List[str]:
        toks = re.findall(r"[a-zA-Z]+", t.lower())
        return [t for t in toks if t not in ENGLISH_STOP_WORDS]