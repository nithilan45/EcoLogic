"""Load the production EcoLogic classifier from ``backend/main.py`` source.

Importing ``backend.main`` pulls FastAPI, httpx, and dotenv. The deployment
endpoint only needs ``classify_prompt_local_nlp`` and ``MODELS``, so those
objects are executed from the production file with a stdlib prelude. Logic is
not reimplemented here.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable

from woais_experiments.paths import ROOT

_PRELUDE = """
from __future__ import annotations
import re
from typing import List, Set, Tuple

class ClassificationResult:
    def __init__(self, difficulty, risk, recommended_tier, reason):
        self.difficulty = difficulty
        self.risk = risk
        self.recommended_tier = recommended_tier
        self.reason = reason
"""


@dataclass(frozen=True)
class RouterDecision:
    difficulty: str
    risk: str
    recommended_tier: int
    reason: str
    selected_model: str
    provider: str
    tier_key: str


class ClassifierLoadError(RuntimeError):
    pass


def _backend_main_text() -> str:
    path = ROOT / "backend" / "main.py"
    if not path.is_file():
        raise ClassifierLoadError(f"missing production classifier at {path}")
    return path.read_text(encoding="utf-8")


def _slice_production_classifier(src: str) -> str:
    start = src.index("MODELS = {")
    models_end = src.index("# GPT-5 baseline")
    models_block = src[start:models_end]
    fn_start = src.index("# Programming language keywords")
    fn_end = src.index("\nasync def query_together")
    fn_block = src[fn_start:fn_end]
    return models_block + "\n" + fn_block


@lru_cache(maxsize=1)
def production_classifier_namespace() -> dict[str, Any]:
    """Execute production classifier source (no FastAPI). Read-only wrap."""
    src = _slice_production_classifier(_backend_main_text())
    ns: dict[str, Any] = {}
    exec(compile(_PRELUDE + src, "backend/main.py:classifier", "exec"), ns, ns)
    if "classify_prompt_local_nlp" not in ns:
        raise ClassifierLoadError("failed to load classify_prompt_local_nlp")
    return ns


@lru_cache(maxsize=1)
def load_production_router() -> tuple[Callable[[str], Any], dict[str, Any]]:
    """Return ``(classify_prompt_local_nlp, MODELS)`` from production source."""
    ns = production_classifier_namespace()
    classify = ns.get("classify_prompt_local_nlp")
    models = ns.get("MODELS")
    if not callable(classify) or not isinstance(models, dict):
        raise ClassifierLoadError("failed to load classify_prompt_local_nlp / MODELS")
    return classify, models


def tier_key(tier: int) -> str:
    return f"tier{int(tier)}"


def model_for_tier(models: dict[str, Any], tier: int) -> dict[str, Any]:
    key = tier_key(tier)
    if key not in models:
        raise KeyError(f"no production model for {key}")
    return models[key]


def classify_prompt(
    prompt: str,
    *,
    models: dict[str, Any] | None = None,
    classify_fn: Callable[[str], Any] | None = None,
) -> RouterDecision:
    if classify_fn is None or models is None:
        loaded_fn, loaded_models = load_production_router()
        classify_fn = classify_fn or loaded_fn
        models = models or loaded_models
    result = classify_fn(prompt)
    tier = int(result.recommended_tier)
    cfg = model_for_tier(models, tier)
    return RouterDecision(
        difficulty=str(result.difficulty),
        risk=str(result.risk),
        recommended_tier=tier,
        reason=str(result.reason),
        selected_model=str(cfg["name"]),
        provider=str(cfg["provider"]),
        tier_key=tier_key(tier),
    )
