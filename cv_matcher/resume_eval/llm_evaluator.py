from typing import List, Optional, Tuple, Union
import hashlib
import os
import re

try:
    from groq import Groq
except Exception:
    Groq = None


class LLMJudge:
    """
    LLM-based evaluator for candidate-role matching.
    Falls back to deterministic mock scoring when API unavailable.
    """
    
    def __init__(
        self,
        use_api: Optional[bool] = None,
        model_id: str = "openai/gpt-oss-120b",
        effort_level: str = "medium",
    ) -> None:
        api_key = os.getenv("GROQ_API_KEY")
        if use_api is None:
            use_api = bool(api_key)
        self.use_api = bool(use_api and api_key and Groq is not None)
        self.model_id = model_id
        self.effort_level = effort_level
        self.client = None
        if self.use_api:
            try:
                self.client = Groq(api_key=api_key)
            except Exception as e:
                print(f"[LLMJudge] Init failed: {e}. Using mock mode.")
                self.use_api = False
    
    def compute_scores(self, target: str, candidates: List[str]) -> List[float]:
        return [self.evaluate_pair(target, cv) for cv in candidates]
    
    def evaluate_pair(self, target: str, candidate: str) -> float:
        if self.client is None:
            return self._mock_evaluate(target, candidate)
        
        prompt = f"""
Assess the alignment between the candidate profile and the target role.
Return a score from 0 to 100 based on skills, experience, and qualifications.

Guidelines:
- 0-20: Minimal alignment
- 21-40: Weak alignment
- 41-60: Moderate alignment  
- 61-80: Strong alignment
- 81-100: Excellent alignment

Output ONLY the integer score, nothing else.

TARGET ROLE:
{target}

CANDIDATE PROFILE:
{candidate}

Score (0-100):
"""
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=[
                {"role": "system", "content": "Output only a numeric score."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_completion_tokens=1000,
            top_p=1,
            reasoning_effort=self.effort_level,
        )
        raw = response.choices[0].message.content.strip()
        return self._parse_numeric(raw)
    
    def provide_rationale(
        self, target: str, candidate: str, include_meta: bool = False
    ) -> Union[str, Tuple[str, Optional[str]]]:
        if self.client is None:
            msg = "LLM service unavailable; using mock explanation."
            return (msg, None) if include_meta else msg
        
        prompt = (
            "Provide 2-4 concise bullet points explaining the match quality. "
            "Focus on specific skills or experience gaps.\n\n"
            f"TARGET ROLE:\n{target}\n\n"
            f"CANDIDATE PROFILE:\n{candidate}"
        )
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=[
                {"role": "system", "content": "Be concise and specific."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_completion_tokens=1000,
            top_p=1,
            reasoning_effort=self.effort_level,
        )
        content = response.choices[0].message.content or ""
        finish = getattr(response.choices[0], "finish_reason", None)
        return (content.strip(), finish) if include_meta else content.strip()
    
    @staticmethod
    def _parse_numeric(text: str) -> float:
        match = re.search(r"-?\d+(?:\.\d+)?", text)
        if not match:
            return 0.0
        val = float(match.group(0))
        val = min(max(val, 0.0), 100.0)
        return val / 100.0
    
    @staticmethod
    def _mock_evaluate(target: str, candidate: str) -> float:
        target_tokens = set(re.findall(r"[a-zA-Z]+", target.lower()))
        candidate_tokens = set(re.findall(r"[a-zA-Z]+", candidate.lower()))
        overlap = len(target_tokens & candidate_tokens)
        base = overlap / max(len(target_tokens), 1)
        digest = hashlib.sha256((target + candidate).encode("utf-8")).hexdigest()
        jitter = (int(digest[:8], 16) % 11) / 100.0
        return min(1.0, base + jitter)