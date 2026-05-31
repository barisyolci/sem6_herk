from typing import List, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def compute_match_scores(target_text: str, candidate_texts: List[str]) -> List[float]:
    """
    Calculate similarity scores between a target role and candidate resumes
    using TF-IDF vectorization and cosine similarity.
    """
    all_texts = [target_text] + candidate_texts
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    
    target_vector = tfidf_matrix[0:1]
    candidate_vectors = tfidf_matrix[1:]
    
    raw_scores = cosine_similarity(target_vector, candidate_vectors)[0]
    normalized_scores = np.clip(raw_scores, 0.0, 1.0)
    
    return normalized_scores.tolist()


def get_keyword_contributions(
    target_text: str, 
    candidate_text: str, 
    max_terms: int = 10
) -> Tuple[float, List[Tuple[str, float]]]:
    """
    Return overall similarity plus top contributing terms for interpretability.
    """
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform([target_text, candidate_text])
    
    target_vec = tfidf[0:1]
    candidate_vec = tfidf[1:2]
    
    similarity = float(cosine_similarity(target_vec, candidate_vec)[0][0])
    
    elementwise = target_vec.multiply(candidate_vec)
    if elementwise.nnz == 0:
        return similarity, []
    
    feature_names = vectorizer.get_feature_names_out()
    indices = elementwise.indices
    values = elementwise.data
    
    term_scores = sorted(
        ((feature_names[idx], float(val)) for idx, val in zip(indices, values)),
        key=lambda x: x[1],
        reverse=True
    )
    
    return similarity, term_scores[:max_terms]