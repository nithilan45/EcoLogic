"""Stage 7 featurizer wrapper.

Lives in its own importable module rather than inside `s7_fit.py` so that the
pickled router references `s7_features.CachedEmbeddingFeatures` instead of
`__main__.CachedEmbeddingFeatures`, and can therefore be loaded by
`s7_calibrate.py`, `s7_final.py` and `s7_variance.py`.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "router_v2"))
from features import Combine, EmbeddingFeatures, HandFeatures  # noqa: E402


class CachedEmbeddingFeatures(EmbeddingFeatures):
    """Memoise MiniLM vectors by text.

    Output is identical to EmbeddingFeatures; this only avoids re-encoding the
    same thousands of prompts once per CV fold per hyperparameter, which at the
    Stage 7 pool size is the dominant cost of the R2 grid.
    """

    _cache: dict[str, np.ndarray] = {}

    def transform(self, X):
        texts = list(X)
        if not texts:
            return csr_matrix((0, self._get().get_sentence_embedding_dimension()))
        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if missing:
            m = self._get()
            v = m.encode(missing, batch_size=self.batch_size,
                         show_progress_bar=False, normalize_embeddings=True)
            for t, vec in zip(missing, np.asarray(v, dtype=float)):
                self._cache[t] = vec
        return csr_matrix(np.vstack([self._cache[t] for t in texts]))


def build_r2_cached():
    return Combine([("emb", CachedEmbeddingFeatures()), ("hand", HandFeatures())])
