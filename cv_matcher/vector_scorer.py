from typing import List
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import linear_kernel

def calculate_alignment(q: str, docs: List[str]) -> List[float]:
    corpus = [q] + docs
    hv = HashingVectorizer(n_features=4096, alternate_sign=False, norm=None)
    X = hv.transform(corpus)
    X_norm = normalize(X, axis=1)
    q_vec = X_norm[0:1]
    d_vecs = X_norm[1:]
    sims = linear_kernel(q_vec, d_vecs).ravel()
    return np.maximum(sims, 0).tolist()

def highlight_key_terms(a: str, b: str, k: int = 10):
    hv = HashingVectorizer(n_features=1024, alternate_sign=False)
    X = hv.transform([a, b])
    Xn = normalize(X, axis=1)
    s = float(linear_kernel(Xn[0:1], Xn[1:2])[0][0])
    return s, []