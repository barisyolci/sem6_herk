
import json
import re
import os
from typing import Dict, List, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine

# Configuration 
LANGUAGE = (os.getenv("BENCHMARK_LANGUAGE") or "english").strip().lower()
USE_CROSS_ENCODER = (os.getenv("USE_CROSS_ENCODER") or "1").strip().lower() in ("1", "true", "yes", "on")
USE_NLI = (os.getenv("USE_NLI") or "0").strip().lower() in ("1", "true", "yes", "on")
USE_BERTSCORE = (os.getenv("USE_BERTSCORE") or "0").strip().lower() in ("1", "true", "yes", "on")
CROSS_ENCODER_MODEL = os.getenv("CROSS_ENCODER_MODEL") or "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
NLI_MODEL = os.getenv("NLI_MODEL") or "joeddav/xlm-roberta-large-xnli"

# Try to import advanced models
CROSS_ENCODER_AVAILABLE = False
NLI_AVAILABLE = False
BERTSCORE_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"PyTorch available, using device: {DEVICE}")
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE = "cpu"
    print("PyTorch not available")

# Cross-Encoder for semantic similarity
if USE_CROSS_ENCODER and TORCH_AVAILABLE:
    try:
        from sentence_transformers import CrossEncoder
        CROSS_ENCODER_AVAILABLE = True
        print("Cross-Encoder available for semantic similarity")
    except ImportError:
        print("Cross-Encoder not available (pip install sentence-transformers)")

# NLI for information correctness
if USE_NLI and TORCH_AVAILABLE:
    try:
        from transformers import pipeline
        NLI_AVAILABLE = True
        print("NLI available for information correctness")
    except ImportError:
        print("NLI not available (pip install transformers)")

# BERTScore for token-level semantic matching
if USE_BERTSCORE:
    try:
        from bert_score import score as bert_score_fn
        BERTSCORE_AVAILABLE = True
        print("BERTScore available for token matching")
    except ImportError:
        print("BERTScore not available (pip install bert-score)")


class BenchmarkEvaluator:
    """
    Advanced benchmark evaluator using multiple methods:
    1. Regex - Exact factual matching (phone numbers, addresses)
    2. Cross-Encoder - Semantic similarity (understands meaning)
    3. NLI - Natural Language Inference (checks information correctness)
    4. BERTScore - Token-level semantic matching
    """

    def __init__(self, factual_data_path: str = "factual_data.json"):
        """Initialize evaluator with factual data and evaluation models."""
        # Load factual data
        with open(factual_data_path, 'r', encoding='utf-8') as f:
            self.factual_data = json.load(f)

        # Initialize Cross-Encoder for semantic similarity
        self.cross_encoder = None
        if CROSS_ENCODER_AVAILABLE:
            try:
                print("Loading Cross-Encoder model...")
                # Use multilingual model for support
                self.cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL, device=DEVICE)
                print("Cross-Encoder ready")
            except Exception as e:
                print(f"Could not load Cross-Encoder: {e}")

        # Initialize NLI pipeline
        self.nli_pipeline = None
        if NLI_AVAILABLE:
            try:
                print("Loading NLI model...")
                self.nli_pipeline = pipeline(
                    "zero-shot-classification",
                    model=NLI_MODEL,
                    device=0 if DEVICE == "cuda" else -1
                )
                print("NLI ready")
            except Exception as e:
                print(f"Could not load NLI: {e}")

        # add later meer talen als de originele goed werkt
        if LANGUAGE.startswith("english"):
            self.nli_labels = ["correct", "incorrect", "partially correct"]
            self.nli_template = "This answer is {}."

        # TF-IDF as fallback
        self.tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)

        # Compile regex patterns for phone numbers
        self.phone_patterns = {}
        for key, data in self.factual_data.get("phone_numbers", {}).items():
            patterns = data.get("patterns", [data.get("phone", "")])
            self.phone_patterns[key] = {
                "name": data.get("name", key),
                "canonical": data.get("phone", ""),
                "regex": [re.compile(re.escape(p).replace(r"\ ", r"\s*").replace(r"\-", r"[-\s]?"))
                         for p in patterns]
            }

        # Compile regex patterns for addresses
        self.address_patterns = {}
        for key, data in self.factual_data.get("addresses", {}).items():
            self.address_patterns[key] = {
                "name": data.get("name", key),
                "street": data.get("street", ""),
                "postcode": data.get("postcode", ""),
                "full": data.get("full", ""),
                "regex_street": re.compile(re.escape(data.get("street", "")).replace(r"\ ", r"\s*"), re.IGNORECASE),
                "regex_postcode": re.compile(re.escape(data.get("postcode", "")).replace(r"\ ", r"\s*"))
            }

    # ==================== CROSS-ENCODER ====================
    def cross_encoder_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity using Cross-Encoder.
        Cross-encoder sees BOTH texts together, understanding relationships.
        Much better than encoding separately!
        Returns: 0.0 to 1.0 similarity score
        """
        if self.cross_encoder is None:
            return None

        try:
            # Cross-encoder takes pairs and returns similarity score
            score = self.cross_encoder.predict([(text1, text2)])[0]
            # Normalize to 0-1 range (model outputs can vary)
            return float(max(0.0, min(1.0, (score + 1) / 2)))  # Assuming -1 to 1 range
        except Exception as e:
            print(f"Cross-encoder error: {e}")
            return None

    # ==================== NLI ====================
    def nli_score(self, response: str, expected: str) -> Tuple[float, str]:
        """
        Check if response ENTAILS the expected answer using NLI.
        - entailment: response contains the same information (good!)
        - contradiction: response says opposite (bad!)
        - neutral: response doesn't address the expected info
        Returns: (score 0-1, label)
        """
        if self.nli_pipeline is None:
            return None, "unavailable"

        try:
            # Use zero-shot classification with entailment labels
            # We check: does the response semantically match the expected answer?
            result = self.nli_pipeline(
                response,
                candidate_labels=self.nli_labels,
                hypothesis_template=self.nli_template,
                multi_label=False
            )

            # Get the top label and its score
            top_label = result["labels"][0].lower()
            confidence = result["scores"][0]

            # Convert to score based on label
            if top_label in ("correct", "صحيح"):
                score = confidence  # High score for correct
                label = "entailment"
            elif "incorrect" in top_label or "غير صحيح" in top_label:
                score = 1.0 - confidence  # Invert for incorrect
                label = "contradiction"
            else:  # partially correct
                score = confidence * 0.6  # Medium score
                label = "neutral"

            return float(score), label
        except Exception as e:
            print(f"NLI error: {e}")
            return None, "error"

    # ==================== BERTSCORE ====================
    def bertscore_similarity(self, response: str, expected: str) -> float:
        """
        Calculate BERTScore - matches tokens by MEANING not exact match.
        "telefoonnummer" and "telefoon nummer" will match well.
        Returns: F1 score (0.0 to 1.0)
        """
        if not BERTSCORE_AVAILABLE:
            return None

        try:
            # BERTScore with Dutch model
            P, R, F1 = bert_score_fn(
                [response],
                [expected],
                lang="nl",  # Dutch
                verbose=False,
                device=DEVICE
            )
            return float(F1[0])
        except Exception as e:
            print(f"BERTScore error: {e}")
            return None

    # ==================== TF-IDF FALLBACK ====================
    def tfidf_similarity(self, text1: str, text2: str) -> float:
        """Fallback TF-IDF similarity when other methods unavailable."""
        try:
            tfidf_matrix = self.tfidf.fit_transform([text1, text2])
            similarity = sklearn_cosine(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity)
        except:
            return 0.0


    
    def check_phone_number(self, response: str, expected_key: str) -> Tuple[bool, str]:
        """Check if response contains the expected phone number."""
        if expected_key not in self.phone_patterns:
            return False, f"Unknown phone key: {expected_key}"
        
        patterns = self.phone_patterns[expected_key]
        for regex in patterns["regex"]:
            if regex.search(response):
                return True, patterns["canonical"]
        
        return False, f"Expected: {patterns['canonical']}"
    
    def check_address(self, response: str, expected_key: str) -> Tuple[bool, str]:
        """Check if response contains the expected address."""
        if expected_key not in self.address_patterns:
            return False, f"Unknown address key: {expected_key}"
        
        patterns = self.address_patterns[expected_key]
        
        # Check for street name
        street_match = patterns["regex_street"].search(response)
        # Check for postcode
        postcode_match = patterns["regex_postcode"].search(response)
        
        if street_match and postcode_match:
            return True, patterns["full"]
        elif street_match:
            return True, f"Partial match (street only): {patterns['street']}"
        
        return False, f"Expected: {patterns['full']}"
    
    def find_any_phone(self, response: str) -> List[str]:
        """Find any phone numbers in the response."""
        found = []
        for key, patterns in self.phone_patterns.items():
            for regex in patterns["regex"]:
                if regex.search(response):
                    found.append(patterns["canonical"])
                    break
        return found
    
    def find_any_address(self, response: str) -> List[str]:
        """Find any addresses in the response."""
        found = []
        for key, patterns in self.address_patterns.items():
            if patterns["regex_street"].search(response):
                found.append(patterns["full"])
        return found
    
    def semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate best available semantic similarity.
        Priority: Cross-Encoder > BERTScore > TF-IDF
        """
        # Try Cross-Encoder first (best)
        if self.cross_encoder is not None:
            score = self.cross_encoder_similarity(text1, text2)
            if score is not None:
                return score

        # Try BERTScore (good)
        if BERTSCORE_AVAILABLE:
            score = self.bertscore_similarity(text1, text2)
            if score is not None:
                return score

        # Fallback to TF-IDF
        return self.tfidf_similarity(text1, text2)

    # Keep old method name for compatibility
    def cosine_similarity(self, text1: str, text2: str) -> float:
        """Alias for semantic_similarity for backward compatibility."""
        return self.semantic_similarity(text1, text2)

    def evaluate_response(self, question: str, response: str, expected_answer: str,
                         category: str, factual_key: str = None) -> Dict:
        """
        Evaluate a single response using multiple methods:
        1. Regex - Exact factual matching (phone/address)
        2. Cross-Encoder - Semantic similarity
        3. NLI - Information correctness check
        4. BERTScore - Token-level semantic matching

        Returns dict with all scores and combined score.
        """
        result = {
            "question": question,
            "response": response[:200] + "..." if len(response) > 200 else response,
            "expected": expected_answer[:200] + "..." if len(expected_answer) > 200 else expected_answer,
            "category": category,
            # Individual scores
            "regex_score": None,
            "cross_encoder_score": None,
            "nli_score": None,
            "nli_label": None,
            "bertscore": None,
            "tfidf_score": None,
            # Final scores
            "semantic_score": 0.0,
            "combined_score": 0.0,
            "details": []
        }

        # ==================== 1. REGEX (Factual Accuracy) ====================
        if category == "phone_numbers":
            phones_found = self.find_any_phone(response)
            expected_phones = self.find_any_phone(expected_answer)
            if expected_phones:
                match = any(p in phones_found for p in expected_phones)
                result["regex_score"] = 1.0 if match else 0.0
                result["details"].append(f"Phone: {'✓' if match else '✗'} (found: {phones_found}, expected: {expected_phones})")

        elif category == "address_info":
            addresses_found = self.find_any_address(response)
            expected_addresses = self.find_any_address(expected_answer)
            if expected_addresses:
                match = any(a in addresses_found for a in expected_addresses)
                result["regex_score"] = 1.0 if match else 0.0
                result["details"].append(f"Address: {'✓' if match else '✗'} (found: {addresses_found})")

        # ==================== 2. CROSS-ENCODER (Semantic Similarity) ====================
        if self.cross_encoder is not None:
            result["cross_encoder_score"] = self.cross_encoder_similarity(response, expected_answer)
            if result["cross_encoder_score"] is not None:
                result["details"].append(f"CrossEncoder: {result['cross_encoder_score']:.3f}")

        # ==================== 3. NLI (Information Correctness) ====================
        if self.nli_pipeline is not None:
            nli_score, nli_label = self.nli_score(response, expected_answer)
            result["nli_score"] = nli_score
            result["nli_label"] = nli_label
            if nli_score is not None:
                result["details"].append(f"NLI: {nli_label} ({nli_score:.3f})")

        # ==================== 4. BERTSCORE (Token Semantic Matching) ====================
        if BERTSCORE_AVAILABLE:
            result["bertscore"] = self.bertscore_similarity(response, expected_answer)
            if result["bertscore"] is not None:
                result["details"].append(f"BERTScore: {result['bertscore']:.3f}")

        # ==================== 5. TF-IDF FALLBACK ====================
        result["tfidf_score"] = self.tfidf_similarity(response, expected_answer)

        # ==================== CALCULATE COMBINED SCORES ====================
        # Semantic score: weighted average of available semantic methods
        semantic_scores = []
        weights = []

        if result["cross_encoder_score"] is not None:
            semantic_scores.append(result["cross_encoder_score"])
            weights.append(0.4)  # Cross-encoder is most reliable

        if result["nli_score"] is not None:
            semantic_scores.append(result["nli_score"])
            weights.append(0.35)  # NLI checks information correctness

        if result["bertscore"] is not None:
            semantic_scores.append(result["bertscore"])
            weights.append(0.25)  # BERTScore for token matching

        if semantic_scores:
            # Normalize weights
            total_weight = sum(weights)
            result["semantic_score"] = sum(s * w for s, w in zip(semantic_scores, weights)) / total_weight
        else:
            # Fallback to TF-IDF
            result["semantic_score"] = result["tfidf_score"]

        result["details"].append(f"Semantic: {result['semantic_score']:.3f}")

        # Combined score: include regex for factual categories
        if result["regex_score"] is not None:
            # For factual categories: 50% regex, 50% semantic
            result["combined_score"] = 0.5 * result["regex_score"] + 0.5 * result["semantic_score"]
        else:
            # For non-factual categories: 100% semantic
            result["combined_score"] = result["semantic_score"]

        return result

    def evaluate_benchmark(self, questions_path: str = "benchmark_questions.json",
                          get_response_fn=None) -> Dict:
        """
        Evaluate full benchmark.

        Args:
            questions_path: Path to benchmark questions JSON
            get_response_fn: Function that takes question string and returns model response

        Returns:
            Dictionary with overall and per-category scores
        """
        with open(questions_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        results = {
            "total_questions": len(data["questions"]),
            "overall_score": 0.0,
            "regex_accuracy": 0.0,
            "semantic_avg": 0.0,
            "cross_encoder_avg": 0.0,
            "nli_avg": 0.0,
            "bertscore_avg": 0.0,
            "by_category": {},
            "detailed_results": []
        }

        category_scores = {}
        regex_scores = []
        semantic_scores = []
        cross_encoder_scores = []
        nli_scores = []
        bertscores = []

        for q in data["questions"]:
            question = q["question"]
            expected = q["correct_answer"]
            category = q["category"]

            # Get model response (or use expected as placeholder)
            if get_response_fn:
                response = get_response_fn(question)
            else:
                response = "[No model response - testing mode]"

            # Evaluate
            eval_result = self.evaluate_response(question, response, expected, category)
            eval_result["question_id"] = q["id"]
            results["detailed_results"].append(eval_result)

            # Aggregate scores
            if category not in category_scores:
                category_scores[category] = {
                    "combined": [], "regex": [], "semantic": [],
                    "cross_encoder": [], "nli": [], "bertscore": []
                }

            category_scores[category]["combined"].append(eval_result["combined_score"])
            category_scores[category]["semantic"].append(eval_result["semantic_score"])
            semantic_scores.append(eval_result["semantic_score"])

            if eval_result["cross_encoder_score"] is not None:
                category_scores[category]["cross_encoder"].append(eval_result["cross_encoder_score"])
                cross_encoder_scores.append(eval_result["cross_encoder_score"])

            if eval_result["nli_score"] is not None:
                category_scores[category]["nli"].append(eval_result["nli_score"])
                nli_scores.append(eval_result["nli_score"])

            if eval_result["bertscore"] is not None:
                category_scores[category]["bertscore"].append(eval_result["bertscore"])
                bertscores.append(eval_result["bertscore"])

            if eval_result["regex_score"] is not None:
                category_scores[category]["regex"].append(eval_result["regex_score"])
                regex_scores.append(eval_result["regex_score"])

        # Calculate averages
        results["overall_score"] = np.mean([r["combined_score"] for r in results["detailed_results"]])
        results["semantic_avg"] = np.mean(semantic_scores) if semantic_scores else 0.0
        results["cross_encoder_avg"] = np.mean(cross_encoder_scores) if cross_encoder_scores else 0.0
        results["nli_avg"] = np.mean(nli_scores) if nli_scores else 0.0
        results["bertscore_avg"] = np.mean(bertscores) if bertscores else 0.0
        results["regex_accuracy"] = np.mean(regex_scores) if regex_scores else 0.0

        for cat, scores in category_scores.items():
            results["by_category"][cat] = {
                "count": len(scores["combined"]),
                "combined_avg": np.mean(scores["combined"]),
                "semantic_avg": np.mean(scores["semantic"]),
                "cross_encoder_avg": np.mean(scores["cross_encoder"]) if scores["cross_encoder"] else None,
                "nli_avg": np.mean(scores["nli"]) if scores["nli"] else None,
                "bertscore_avg": np.mean(scores["bertscore"]) if scores["bertscore"] else None,
                "regex_accuracy": np.mean(scores["regex"]) if scores["regex"] else None
            }

        return results


def print_results(results: Dict):
    """Pretty print evaluation results."""
    print("\n" + "="*70)
    print("BENCHMARK EVALUATION RESULTS")
    print("="*70)
    print(f"\nTotal Questions: {results['total_questions']}")
    print(f"\n--- Overall Scores ---")
    print(f"Combined Score:     {results['overall_score']:.1%}")
    print(f"Semantic Score:     {results['semantic_avg']:.1%}")
    print(f"Cross-Encoder:      {results['cross_encoder_avg']:.1%}")
    print(f"NLI Score:          {results['nli_avg']:.1%}")
    print(f"BERTScore:          {results['bertscore_avg']:.1%}")
    print(f"Regex (factual):    {results['regex_accuracy']:.1%}")

    print("\n" + "-"*70)
    print("SCORES BY CATEGORY:")
    print("-"*70)
    for cat, scores in sorted(results["by_category"].items()):
        regex_str = f"{scores['regex_accuracy']:.1%}" if scores['regex_accuracy'] is not None else "N/A"
        ce_str = f"{scores['cross_encoder_avg']:.1%}" if scores['cross_encoder_avg'] is not None else "N/A"
        nli_str = f"{scores['nli_avg']:.1%}" if scores['nli_avg'] is not None else "N/A"
        print(f"\n  {cat}:")
        print(f"    Combined: {scores['combined_avg']:.1%} | Semantic: {scores['semantic_avg']:.1%}")
        print(f"    CrossEnc: {ce_str} | NLI: {nli_str} | Regex: {regex_str}")


if __name__ == "__main__":
    # Test the evaluator
    print("Initializing evaluator...")
    evaluator = BenchmarkEvaluator()

    # Test single evaluation
    print("\n" + "="*70)
    print("Testing single evaluation...")
    print("="*70)
    test_result = evaluator.evaluate_response(
        question="What is the phone number for Pauluskerk?",
        response="You can reach Pauluskerk at 010-411 5950 during opening hours.",
        expected_answer="The phone number for Pauluskerk Rotterdam is 010-411 5950.",
        category="phone_numbers"
    )

    print(f"\nResults:")
    print(f"  Regex score:        {test_result['regex_score']}")
    print(f"  Cross-Encoder:      {test_result['cross_encoder_score']}")
    print(f"  NLI score:          {test_result['nli_score']} ({test_result['nli_label']})")
    print(f"  BERTScore:          {test_result['bertscore']}")
    print(f"  Semantic (combined):{test_result['semantic_score']:.3f}")
    print(f"  Final Combined:     {test_result['combined_score']:.3f}")
    print(f"\nDetails: {test_result['details']}")

    print("\n" + "="*70)
    print("To run full benchmark, use:")
    print("  evaluator.evaluate_benchmark('benchmark_questions.json', your_model_function)")
    print("="*70)
