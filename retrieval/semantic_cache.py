from collections import OrderedDict
from typing import Any, Optional

import numpy as np


class SemanticCache:
    """In-memory semantic cache using cosine similarity over embeddings."""

    def __init__(self, threshold: float = 0.95, max_entries: int = 1000):
        self.threshold = threshold
        self.max_entries = max_entries
        self._store: "OrderedDict[str, dict[str, Any]]" = OrderedDict()

    def get(self, query: str, query_embedding: np.ndarray) -> Optional[Any]:
        """Return cached value for closest query if above similarity threshold."""
        if not self._store:
            return None

        if not isinstance(query_embedding, np.ndarray):
            query_embedding = np.array(query_embedding, dtype=np.float32)

        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            return None

        best_key = None
        best_similarity = -1.0

        for key, payload in self._store.items():
            cached_embedding = payload["embedding"]
            denominator = query_norm * np.linalg.norm(cached_embedding)
            if denominator == 0:
                continue

            similarity = float(np.dot(query_embedding, cached_embedding) / denominator)
            if similarity > best_similarity:
                best_similarity = similarity
                best_key = key

        if best_key is not None and best_similarity >= self.threshold:
            # mark as recently used
            self._store.move_to_end(best_key)
            return self._store[best_key]["value"]

        return None

    def put(self, query: str, query_embedding: np.ndarray, value: Any):
        """Insert query/value pair and evict oldest entries if needed."""
        if not isinstance(query_embedding, np.ndarray):
            query_embedding = np.array(query_embedding, dtype=np.float32)

        self._store[query] = {
            "embedding": query_embedding,
            "value": value,
        }
        self._store.move_to_end(query)

        while len(self._store) > self.max_entries:
            self._store.popitem(last=False)
