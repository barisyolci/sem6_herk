# Reflection

## Why TF-IDF is a baseline
TF-IDF is fast, transparent, and easy to implement, which makes it a useful baseline for text matching. It captures keyword overlap and term importance, providing a simple signal for relevance without any training or external dependencies beyond standard ML tooling.

## Why embeddings outperform TF-IDF
Sentence-BERT embeddings encode semantic meaning beyond exact word overlap. This lets the model connect related concepts (e.g., "model evaluation" and "metrics") and handle paraphrases, which improves ranking quality in CV matching tasks where phrasing varies across candidates.

## Strengths and weaknesses of LLM scoring
LLM scoring can consider broader context and nuanced skill alignment, making it flexible for complex job requirements. However, it is slower, depends on external APIs, and can introduce nondeterminism or cost, which limits its suitability for repeated benchmarking without careful control.

## Limitations of small-scale benchmarks
A dataset with one job description and a small number of CVs is useful for classroom demonstration but does not represent real-world diversity. Results can be sensitive to the specific examples chosen, and the metrics may not generalize to other roles or larger candidate pools.
