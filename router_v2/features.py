"""Feature extraction for the learned router.

Everything here is computed locally on CPU. No LLM call is made to route a
query, which preserves EcoLogic's core constraint: the router must not itself
consume the resource it is trying to save.
"""

import re

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.base import BaseEstimator, TransformerMixin

QUESTION_WORDS = ("what", "who", "when", "where", "why", "how", "which", "whose")
CODE_TOKENS = ("def ", "import ", "return", "class ", "lambda", "assert",
               "print(", "()", "[]", "{}", "self.", "->")

HAND_FEATURE_NAMES = [
    "n_words", "n_chars", "log_n_words", "mean_word_len",
    "has_code_fence", "n_code_tokens", "has_def", "has_import", "has_assert",
    "paren_density", "colon_density", "bracket_density", "operator_density",
    "n_newlines", "has_question_mark", "n_question_words", "starts_with_question_word",
    "digit_ratio", "has_digit", "n_numbers", "has_currency", "has_percent",
    "upper_ratio", "has_mcq_markers", "n_mcq_markers",
]


def hand_features(text: str) -> list[float]:
    t = text or ""
    low = t.lower()
    words = t.split()
    nw = len(words)
    nc = len(t)
    denom = max(nc, 1)
    numbers = re.findall(r"\d+(?:\.\d+)?", t)
    mcq = re.findall(r"^\s*[ABCD][\.\)]\s", t, re.MULTILINE)
    return [
        nw,
        nc,
        float(np.log1p(nw)),
        (sum(len(w) for w in words) / nw) if nw else 0.0,
        float("```" in t),
        float(sum(tok in low for tok in CODE_TOKENS)),
        float("def " in low),
        float("import " in low),
        float("assert" in low),
        t.count("(") / denom,
        t.count(":") / denom,
        (t.count("[") + t.count("{")) / denom,
        sum(t.count(c) for c in "+-*/=<>") / denom,
        float(t.count("\n")),
        float("?" in t),
        float(sum(low.count(w) for w in QUESTION_WORDS)),
        float(low.strip().startswith(QUESTION_WORDS)),
        sum(c.isdigit() for c in t) / denom,
        float(any(c.isdigit() for c in t)),
        float(len(numbers)),
        float("$" in t),
        float("%" in t),
        (sum(c.isupper() for c in t) / denom),
        float(len(mcq) >= 2),
        float(len(mcq)),
    ]


class HandFeatures(BaseEstimator, TransformerMixin):
    """Sparse-compatible hand-engineered feature block."""

    def fit(self, X, y=None):
        raw = np.asarray([hand_features(t) for t in X], dtype=float)
        self.mean_ = raw.mean(axis=0)
        self.scale_ = np.where(raw.std(axis=0) > 0, raw.std(axis=0), 1.0)
        return self

    def transform(self, X):
        raw = np.asarray([hand_features(t) for t in X], dtype=float)
        return csr_matrix((raw - self.mean_) / self.scale_)


class EmbeddingFeatures(BaseEstimator, TransformerMixin):
    """all-MiniLM-L6-v2 sentence embeddings, computed locally on CPU."""

    _model = None

    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", batch_size=64):
        self.model_name = model_name
        self.batch_size = batch_size

    def _get(self):
        if EmbeddingFeatures._model is None:
            from sentence_transformers import SentenceTransformer
            EmbeddingFeatures._model = SentenceTransformer(self.model_name, device="cpu")
        return EmbeddingFeatures._model

    def fit(self, X, y=None):
        self._get()
        return self

    def transform(self, X):
        m = self._get()
        v = m.encode(list(X), batch_size=self.batch_size, show_progress_bar=False,
                     normalize_embeddings=True)
        return csr_matrix(np.asarray(v, dtype=float))


class Combine(BaseEstimator, TransformerMixin):
    """Horizontally stack several fitted transformers."""

    def __init__(self, parts):
        self.parts = parts

    def fit(self, X, y=None):
        for _, p in self.parts:
            p.fit(X, y)
        return self

    def transform(self, X):
        return hstack([p.transform(X) for _, p in self.parts]).tocsr()
