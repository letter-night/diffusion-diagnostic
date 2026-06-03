"""Optional LLM-judge quality/consistency score (Case 3/4).

ANNOTATION ONLY. This score is observational metadata — it is never used as a reward,
never backpropagated, never used to filter or reweight samples. That would cross into the
GDC/reward-model scope this study explicitly excludes.

Default backend is `offline`: a deterministic, hash-based pseudo-score so the pipeline runs
without API calls. Swap in a real judge by implementing `_score_api` and passing
backend="api". Scores are cached by completion hash to avoid recomputation.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any


class LLMJudge:
    def __init__(self, backend: str = "offline", cache_path: str | None = None,
                 model: str = "offline-stub"):
        self.backend = backend
        self.model = model
        self.cache_path = cache_path
        self.cache: dict[str, Any] = {}
        if cache_path and os.path.exists(cache_path):
            with open(cache_path) as f:
                self.cache = json.load(f)

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def score(self, text: str, ctx: dict | None = None) -> dict:
        k = self._key(text)
        if k in self.cache:
            return self.cache[k]
        if self.backend == "offline":
            out = self._score_offline(text)
        elif self.backend == "api":
            out = self._score_api(text, ctx or {})
        else:
            raise ValueError(f"Unknown judge backend: {self.backend!r}")
        out["judge_model"] = self.model
        self.cache[k] = out
        return out

    def _score_offline(self, text: str) -> dict:
        # Deterministic pseudo-score in [1,5] from a hash; purely a placeholder.
        h = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
        quality = 1 + (h % 41) / 10.0  # 1.0 .. 5.0
        return {"quality": round(quality, 2), "backend": "offline"}

    def _score_api(self, text: str, ctx: dict) -> dict:  # pragma: no cover
        raise NotImplementedError(
            "Wire up a real judge API here (annotation-only). Return {'quality': float}.")

    def flush(self) -> None:
        if self.cache_path:
            with open(self.cache_path, "w") as f:
                json.dump(self.cache, f, indent=2)
