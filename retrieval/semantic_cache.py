from typing import Any, List, Optional

import numpy as np


class SemanticCache:
    """In-memory semantic cache with cosine similarity lookup."""

    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold
        self.embeddings: List[np.ndarray] = []
        self.values: List[Any] = []

    def get(self, query_embedding: np.ndarray) -> Optional[Any]:
        """Return cached value if similarity threshold is met."""
        if not self.embeddings:
            return None

        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            return None

        embedding_matrix = np.vstack(self.embeddings)
        norms = np.linalg.norm(embedding_matrix, axis=1)
        nonzero_mask = norms > 0
        if not np.any(nonzero_mask):
            return None

        similarities = np.full(len(self.embeddings), -1.0)
        similarities[nonzero_mask] = (
            embedding_matrix[nonzero_mask] @ query_embedding / (norms[nonzero_mask] * query_norm)
        )

        max_idx = int(np.argmax(similarities))
        if similarities[max_idx] >= self.threshold:
            return self.values[max_idx]

        return None

    def put(self, query_embedding: np.ndarray, value: Any):
        """Store value keyed by query embedding."""
        self.embeddings.append(query_embedding)
        self.values.append(value)
