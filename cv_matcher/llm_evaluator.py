from typing import List
import hashlib
import re

class LLMJudge:
    def __init__(self):
        self.active = False

    def compute_scores(self, q: str, docs: List[str]) -> List[float]:
        return [self._eval_pair(q, d) for d in docs]

    def _eval_pair(self, q: str, d: str) -> float:
        return self._mock(q, d)

    def provide_rationale(self, q: str, d: str, meta: bool = False) -> str:
        return "Mock explanation: alignment based on keyword overlap."

    @staticmethod
    def _mock(q: str, d: str) -> float:
        tq = set(re.findall(r"[a-zA-Z]+", q.lower()))
        td = set(re.findall(r"[a-zA-Z]+", d.lower()))
        ov = len(tq & td)
        base = ov / max(len(tq), 1)
        dg = hashlib.sha256((q + d).encode("utf-8")).hexdigest()
        jitter = (int(dg[:8], 16) % 11) / 100.0
        return min(1.0, base + jitter)