from typing import List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def score_cvs(job_description: str, cv_texts: List[str]) -> List[float]:
    texts = [job_description] + cv_texts
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform(texts)

    job_vec = tfidf[0:1]
    cv_vecs = tfidf[1:]
    sims = cosine_similarity(job_vec, cv_vecs)[0]
    sims = np.clip(sims, 0.0, 1.0)
    return sims.tolist()


def explain_pair(job_description: str, cv_text: str, top_n: int = 10) -> Tuple[float, List[Tuple[str, float]]]:
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform([job_description, cv_text])

    job_vec = tfidf[0:1]
    cv_vec = tfidf[1:2]
    similarity = float(cosine_similarity(job_vec, cv_vec)[0][0])

    contribution = job_vec.multiply(cv_vec)
    if contribution.nnz == 0:
        return similarity, []

    feature_names = vectorizer.get_feature_names_out()
    indices = contribution.indices
    values = contribution.data
    ranked = sorted(
        ((feature_names[idx], float(val)) for idx, val in zip(indices, values)),
        key=lambda item: item[1],
        reverse=True,
    )
    return similarity, ranked[:top_n]
