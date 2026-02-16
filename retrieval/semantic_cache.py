from collections import OrderedDict
from typing import Any, Optional

import numpy as np
from sentence_transformers import SentenceTransformer


class SemanticCache:
    """Embedding-based semantic cache for repeated or near-duplicate queries."""

    def __init__(
        self,
        model_name: str,
        similarity_threshold: float = 0.95,
        max_size: int = 500,
    ):
        self.model = SentenceTransformer(model_name)
        self.similarity_threshold = similarity_threshold
        self.max_size = max_size

        self._entries: OrderedDict[str, Any] = OrderedDict()
        self._embeddings: list[np.ndarray] = []
        self._keys: list[str] = []

    def get(self, query: str) -> Optional[Any]:
        """Return a cached value when semantic similarity exceeds threshold."""
        if not self._keys:
            return None

        query_emb = self.model.encode([query], convert_to_numpy=True)[0]
        embedding_matrix = np.array(self._embeddings)
        norms = np.linalg.norm(embedding_matrix, axis=1) * np.linalg.norm(query_emb)
        norms = np.where(norms == 0, 1e-10, norms)
        similarities = np.dot(embedding_matrix, query_emb) / norms

        max_idx = int(np.argmax(similarities))
        if similarities[max_idx] >= self.similarity_threshold:
            key = self._keys[max_idx]
            # LRU refresh
            value = self._entries.pop(key)
            self._entries[key] = value
            return value

        return None

    def put(self, query: str, value: Any):
        """Store value for query and evict oldest records when over capacity."""
        emb = self.model.encode([query], convert_to_numpy=True)[0]

        if query in self._entries:
            self._entries.pop(query)
            idx = self._keys.index(query)
            self._keys.pop(idx)
            self._embeddings.pop(idx)

        self._entries[query] = value
        self._keys.append(query)
        self._embeddings.append(emb)

        while len(self._entries) > self.max_size:
            oldest_key, _ = self._entries.popitem(last=False)
            old_idx = self._keys.index(oldest_key)
            self._keys.pop(old_idx)
            self._embeddings.pop(old_idx)
