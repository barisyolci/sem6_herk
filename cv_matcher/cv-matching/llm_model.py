from typing import List, Optional, Tuple, Union
import hashlib
import os
import re

try:
    from groq import Groq
except Exception:
    Groq = None


class LLMScorer:
    def __init__(
        self,
        use_groq: Optional[bool] = None,
        model_name: str = "openai/gpt-oss-120b",
        reasoning_effort: str = "medium",
    ) -> None:
        api_key = os.getenv("GROQ_API_KEY")
        if use_groq is None:
            use_groq = bool(api_key)
        self.use_groq = bool(use_groq and api_key and Groq is not None)
        self.model_name = model_name
        self.reasoning_effort = reasoning_effort
        self.client = None
        if self.use_groq:
            try:
                self.client = Groq(api_key=api_key)
            except Exception as exc:
                print(f"Groq client init failed: {exc}. Falling back to mock scoring.")
                self.use_groq = False

    def score_cvs(self, job_description: str, cv_texts: List[str]) -> List[float]:
        return [self.score_pair(job_description, cv_text) for cv_text in cv_texts]

    def score_pair(self, job_description: str, cv_text: str) -> float:
        if self.client is None:
            return self.mock_score(job_description, cv_text)

        prompt = (f"""
                        Evaluate how well the CV matches the job description and provide a numerical score from 0 to 100.

                        SCORING GUIDELINES:
                        - 0-20: Poor match (minimal to no relevant skills/experience)
                        - 21-40: Weak match (some relevant skills but major gaps)
                        - 41-60: Moderate match (meets basic requirements with some gaps)
                        - 61-80: Good match (strong alignment with most requirements)
                        - 81-100: Excellent match (exceeds requirements, ideal candidate)

                        IMPORTANT:
                        - Consider skills, experience, qualifications, and domain knowledge
                        - Output MUST be ONLY the integer score with no additional text
                        - Do not include explanation, commentary, or formatting
                        - Output format example: "85"

                        JOB DESCRIPTION:
                        {job_description}

                        CV:
                        {cv_text}

                        Score (0-100):
                        """
        )

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a strict evaluator. Output only a number."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_completion_tokens=1000,
            top_p=1,
            reasoning_effort=self.reasoning_effort,
        )
        text = response.choices[0].message.content.strip()
        return self.parse_score(text)

    def explain_pair(
        self,
        job_description: str,
        cv_text: str,
        return_meta: bool = False,
    ) -> Union[str, Tuple[str, Optional[str]]]:
        if self.client is None:
            message = "LLM unavailable; using deterministic mock scoring."
            return (message, None) if return_meta else message

        prompt = (
            "Explain in 2-4 short bullet points why the CV matches or does not match the job. "
            "Be specific about skills, tools, or experience. Keep it concise.\n\n"
            f"JOB DESCRIPTION:\n{job_description}\n\n"
            f"CV:\n{cv_text}"
        )
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a strict evaluator."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_completion_tokens=1000,
            top_p=1,
            reasoning_effort=self.reasoning_effort,
        )
        content = response.choices[0].message.content or ""
        text = content.strip()
        finish_reason = getattr(response.choices[0], "finish_reason", None)
        return (text, finish_reason) if return_meta else text

    @staticmethod
    def parse_score(text: str) -> float:
        match = re.search(r"-?\d+(?:\.\d+)?", text)
        if not match:
            return 0.0
        value = float(match.group(0))
        value = min(max(value, 0.0), 100.0)
        return value / 100.0

    @staticmethod
    def mock_score(job_description: str, cv_text: str) -> float:
        job_tokens = set(re.findall(r"[a-zA-Z]+", job_description.lower()))
        cv_tokens = set(re.findall(r"[a-zA-Z]+", cv_text.lower()))
        overlap = len(job_tokens & cv_tokens)
        base = overlap / max(len(job_tokens), 1)
        digest = hashlib.sha256((job_description + cv_text).encode("utf-8")).hexdigest()
        jitter = (int(digest[:8], 16) % 11) / 100.0
        return min(1.0, base + jitter)
