import math
from dataclasses import dataclass
from typing import Any, List, Optional

from sentence_transformers import SentenceTransformer


@dataclass
class SemanticCacheEntry:
    query: str
    embedding: List[float]
    value: Any


class SemanticCache:
    """In-memory semantic cache for query/result reuse."""

    def __init__(self, model_name: str, similarity_threshold: float = 0.95):
        self.model = SentenceTransformer(model_name)
        self.similarity_threshold = similarity_threshold
        self.entries: List[SemanticCacheEntry] = []

    def _l2_norm(self, values: List[float]) -> float:
        return math.sqrt(sum(value * value for value in values))

    def _encode(self, text: str) -> List[float]:
        embedding = self.model.encode([text], convert_to_numpy=False)[0]
        embedding_list = [float(v) for v in embedding]
        norm = self._l2_norm(embedding_list)
        if norm == 0:
            return embedding_list
        return [value / norm for value in embedding_list]

    def _cosine_similarity(self, lhs: List[float], rhs: List[float]) -> float:
        return sum(l * r for l, r in zip(lhs, rhs))

    def get(self, query: str, threshold: Optional[float] = None) -> Optional[Any]:
        if not self.entries:
            return None

        query_embedding = self._encode(query)
        similarities = [self._cosine_similarity(query_embedding, entry.embedding) for entry in self.entries]

        best_index = max(range(len(similarities)), key=lambda idx: similarities[idx])
        best_similarity = similarities[best_index]
        active_threshold = threshold if threshold is not None else self.similarity_threshold

        if best_similarity >= active_threshold:
            return self.entries[best_index].value
        return None

    def put(self, query: str, value: Any):
        self.entries.append(
            SemanticCacheEntry(
                query=query,
                embedding=self._encode(query),
                value=value,
            )
        )
