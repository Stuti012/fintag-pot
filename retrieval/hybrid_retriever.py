from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import yaml
from rank_bm25 import BM25Okapi
import hashlib

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - optional dependency
    SentenceTransformer = None


from indexing.schema import Document, RetrievalResult
from retrieval.semantic_cache import SemanticCache

try:
    import chromadb
    from chromadb.config import Settings
except Exception:  # pragma: no cover - optional dependency
    chromadb = None
    Settings = None

try:
    from sentence_transformers import CrossEncoder
except ImportError:  # pragma: no cover - optional dependency
    CrossEncoder = None


class _NumpyVectorStore:
    """Small persistent vector store used when ChromaDB is unavailable."""

    def __init__(self, persist_path: Path):
        self.persist_path = persist_path
        self.ids: List[str] = []
        self.embeddings: np.ndarray = np.empty((0, 0), dtype=np.float32)
        self.documents: List[str] = []
        self.metadatas: List[Dict[str, str]] = []
        self._load()

    def _load(self):
        if not self.persist_path.exists():
            return
        with open(self.persist_path, "rb") as f:
            data = pickle.load(f)
        self.ids = data.get("ids", [])
        self.documents = data.get("documents", [])
        self.metadatas = data.get("metadatas", [])
        embeddings = data.get("embeddings", [])
        self.embeddings = np.asarray(embeddings, dtype=np.float32)

    def _save(self):
        data = {
            "ids": self.ids,
            "embeddings": self.embeddings.tolist(),
            "documents": self.documents,
            "metadatas": self.metadatas,
        }
        with open(self.persist_path, "wb") as f:
            pickle.dump(data, f)

    def add(self, ids: List[str], embeddings: List[List[float]], documents: List[str], metadatas: List[Dict[str, str]]):
        new_embeddings = np.asarray(embeddings, dtype=np.float32)
        if self.embeddings.size == 0:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])

        self.ids.extend(ids)
        self.documents.extend(documents)
        self.metadatas.extend(metadatas)
        self._save()

    def query(self, query_embeddings: List[List[float]], n_results: int):
        if self.embeddings.size == 0:
            return {"ids": [[]], "distances": [[]]}

        query = np.asarray(query_embeddings[0], dtype=np.float32)
        query_norm = np.linalg.norm(query)
        emb_norm = np.linalg.norm(self.embeddings, axis=1)
        denom = np.maximum(emb_norm * query_norm, 1e-12)
        cosine_sim = np.dot(self.embeddings, query) / denom
        cosine_dist = 1.0 - cosine_sim

        k = min(max(n_results, 1), len(self.ids))
        top_idx = np.argsort(cosine_dist)[:k]

        return {
            "ids": [[self.ids[i] for i in top_idx]],
            "distances": [[float(cosine_dist[i]) for i in top_idx]],
        }

    def clear(self):
        self.ids = []
        self.documents = []
        self.metadatas = []
        self.embeddings = np.empty((0, 0), dtype=np.float32)
        if self.persist_path.exists():
            self.persist_path.unlink()


class _HashingSentenceTransformerFallback:
    """Deterministic, lightweight embedding fallback."""

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _encode_single(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
        return vector

    def encode(self, texts: List[str], show_progress_bar: bool = False, convert_to_numpy: bool = True):
        _ = show_progress_bar
        embeddings = np.vstack([self._encode_single(text) for text in texts])
        return embeddings if convert_to_numpy else embeddings.tolist()


class HybridRetriever:
    """Hybrid retrieval using BM25 + vector search with optional reranking/cache."""

    def __init__(self, collection_name: str, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        retrieval_cfg = self.config.get("retrieval", {})
        rerank_cfg = retrieval_cfg.get("reranking", {})
        semantic_cfg = retrieval_cfg.get("semantic_cache", {})

        self.collection_name = collection_name
        self.top_k = retrieval_cfg.get("top_k", 5)
        self.bm25_weight = retrieval_cfg.get("bm25_weight", 0.5)
        self.vector_weight = retrieval_cfg.get("vector_weight", 0.5)

        self.enable_reranking = rerank_cfg.get("enabled", False)
        self.rerank_weight = rerank_cfg.get("blend_weight", 0.3)
        self.rerank_candidates_multiplier = rerank_cfg.get("candidate_multiplier", 3)

        self.enable_semantic_cache = semantic_cfg.get("enabled", False)
        self.semantic_cache_threshold = semantic_cfg.get("threshold", 0.95)

        print(f"Loading embedding model: {self.config['embedding']['model']}")
        if SentenceTransformer is None:
            print("Warning: sentence-transformers not installed; using hashing embedding fallback")
            self.embedding_model = _HashingSentenceTransformerFallback(
                dim=self.config["embedding"].get("fallback_dim", 256)
            )
        else:
            self.embedding_model = SentenceTransformer(self.config["embedding"]["model"])

        self.cross_encoder = None
        if self.enable_reranking:
            rerank_model_name = rerank_cfg.get("model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
            if CrossEncoder is None:
                print("Warning: CrossEncoder unavailable, reranking disabled")
                self.enable_reranking = False
            else:
                print(f"Loading reranker model: {rerank_model_name}")
                self.cross_encoder = CrossEncoder(rerank_model_name)

        self.semantic_cache = None
        if self.enable_semantic_cache:
            print(f"Initializing semantic cache (threshold={self.semantic_cache_threshold})")
            self.semantic_cache = SemanticCache(
                model_name=self.config["embedding"]["model"],
                similarity_threshold=self.semantic_cache_threshold,
            )

        persist_dir = Path(self.config["indexing"]["chroma_persist_dir"])
        persist_dir.mkdir(parents=True, exist_ok=True)
        self.bm25_index_path = persist_dir / f"{collection_name}_bm25.pkl"
        self.vector_index_path = persist_dir / f"{collection_name}_vectors.pkl"

        self.using_chroma = chromadb is not None
        if self.using_chroma:
            self.chroma_client = chromadb.PersistentClient(
                path=str(persist_dir),
                settings=Settings(anonymized_telemetry=False),
            )
            try:
                self.collection = self.chroma_client.get_collection(name=collection_name)
                print(f"Loaded existing collection: {collection_name}")
            except Exception:
                self.collection = self.chroma_client.create_collection(
                    name=collection_name,
                    metadata={"hnsw:space": "cosine"},
                )
                print(f"Created new collection: {collection_name}")
        else:
            print("ChromaDB not installed. Using local NumPy vector store fallback.")
            self.chroma_client = None
            self.collection = _NumpyVectorStore(self.vector_index_path)

        self.bm25_index = None
        self.bm25_docs: List[Document] = []
        self._load_bm25_index()

    def index_documents(self, documents: List[Document]):
        if not documents:
            return

        print(f"Indexing {len(documents)} documents...")
        ids, contents, embeddings, metadatas = [], [], [], []

        for doc in documents:
            doc_full_id = f"{doc.doc_id}_{doc.chunk_id}"
            ids.append(doc_full_id)
            contents.append(doc.content)
            metadata = {
                "doc_id": doc.doc_id,
                "chunk_id": doc.chunk_id,
                "doc_type": str(doc.doc_type),
                "source_type": str(doc.source_type),
            }
            for key, value in doc.metadata.items():
                metadata[key] = value if isinstance(value, (str, int, float, bool)) else str(value)
            metadatas.append(metadata)

        batch_size = self.config["embedding"].get("batch_size", 32)
        for i in range(0, len(contents), batch_size):
            batch = contents[i : i + batch_size]
            batch_embeddings = self.embedding_model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
            embeddings.extend(batch_embeddings.tolist())

        self.collection.add(ids=ids, embeddings=embeddings, documents=contents, metadatas=metadatas)

        self.bm25_docs.extend(documents)
        self.bm25_index = BM25Okapi([doc.content.lower().split() for doc in self.bm25_docs])
        self._save_bm25_index()
        print(f"Indexed {len(documents)} documents successfully")

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[RetrievalResult]:
        top_k = top_k or self.top_k

        if self.semantic_cache is not None:
            cached_results = self.semantic_cache.get(query)
            if cached_results is not None:
                return cached_results[:top_k]

        if not self.bm25_docs:
            return []

        retrieval_pool_size = top_k * self.rerank_candidates_multiplier if self.enable_reranking else top_k
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)[0].tolist()

        vector_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(retrieval_pool_size * 2, len(self.bm25_docs)),
        )

        bm25_scores: Dict[str, float] = {}
        if self.bm25_index is not None:
            tokenized_query = query.lower().split()
            scores = self.bm25_index.get_scores(tokenized_query)
            for i, score in enumerate(scores):
                doc = self.bm25_docs[i]
                bm25_scores[f"{doc.doc_id}_{doc.chunk_id}"] = float(score)

        combined_scores: Dict[str, float] = {}
        for i, doc_id in enumerate(vector_results["ids"][0]):
            distance = vector_results["distances"][0][i]
            vector_score = 1.0 / (1.0 + distance)
            combined_scores[doc_id] = self.vector_weight * vector_score

        if bm25_scores:
            max_bm25 = max(bm25_scores.values())
            for doc_id, score in bm25_scores.items():
                normalized = score / max_bm25 if max_bm25 > 0 else 0.0
                combined_scores[doc_id] = combined_scores.get(doc_id, 0.0) + self.bm25_weight * normalized

        ranked_ids = sorted(combined_scores, key=lambda d: combined_scores[d], reverse=True)

        candidates: List[RetrievalResult] = []
        for rank, doc_id in enumerate(ranked_ids[:retrieval_pool_size], start=1):
            doc = self._find_document_by_id(doc_id)
            if doc is None:
                continue
            candidates.append(RetrievalResult(document=doc, score=combined_scores[doc_id], rank=rank))

        if self.enable_reranking and self.cross_encoder is not None and candidates:
            candidates = self._rerank_results(query=query, candidates=candidates, top_k=top_k)

        final_results = candidates[:top_k]
        for i, result in enumerate(final_results, start=1):
            result.rank = i

        if self.semantic_cache is not None:
            self.semantic_cache.put(query, final_results)

        return final_results

    def _rerank_results(self, query: str, candidates: List[RetrievalResult], top_k: int) -> List[RetrievalResult]:
        pairs = [[query, c.document.content] for c in candidates]
        rerank_scores = self.cross_encoder.predict(pairs)

        rerank_array = np.asarray(rerank_scores, dtype=np.float32)
        min_score = float(np.min(rerank_array)) if rerank_array.size else 0.0
        max_score = float(np.max(rerank_array)) if rerank_array.size else 1.0
        denom = max(max_score - min_score, 1e-12)

        for i, candidate in enumerate(candidates):
            norm_rerank = (float(rerank_array[i]) - min_score) / denom
            candidate.score = ((1.0 - self.rerank_weight) * candidate.score) + (self.rerank_weight * norm_rerank)

        candidates.sort(key=lambda x: x.score, reverse=True)
        reranked = candidates[:top_k]
        for rank, result in enumerate(reranked, start=1):
            result.rank = rank
        return reranked

    def _find_document_by_id(self, full_doc_id: str) -> Optional[Document]:
        for doc in self.bm25_docs:
            if f"{doc.doc_id}_{doc.chunk_id}" == full_doc_id:
                return doc
        return None

    def _save_bm25_index(self):
        with open(self.bm25_index_path, "wb") as f:
            pickle.dump({"index": self.bm25_index, "docs": self.bm25_docs}, f)

    def _load_bm25_index(self):
        if not self.bm25_index_path.exists():
            return
        try:
            with open(self.bm25_index_path, "rb") as f:
                data = pickle.load(f)
            self.bm25_index = data.get("index")
            self.bm25_docs = data.get("docs", [])
            print(f"Loaded BM25 index with {len(self.bm25_docs)} documents")
        except Exception as exc:
            print(f"Failed to load BM25 index: {exc}")

    def clear_collection(self):
        if self.using_chroma:
            try:
                self.chroma_client.delete_collection(name=self.collection_name)
                print(f"Deleted collection: {self.collection_name}")
            except Exception:
                pass
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        else:
            self.collection.clear()

        self.bm25_index = None
        self.bm25_docs = []
        if self.bm25_index_path.exists():
            self.bm25_index_path.unlink()


if __name__ == "__main__":
    from indexing.schema import DocumentType, SourceType

    retriever = HybridRetriever("test_collection")
    retriever.clear_collection()

    docs = [
        Document(
            doc_id="doc1",
            chunk_id="chunk0",
            content="Apple reported revenue of $394.3 billion in fiscal 2022.",
            doc_type=DocumentType.TEXT,
            source_type=SourceType.FINQA,
            metadata={},
        ),
        Document(
            doc_id="doc2",
            chunk_id="chunk0",
            content="Microsoft's revenue grew by 18% year over year.",
            doc_type=DocumentType.TEXT,
            source_type=SourceType.FINQA,
            metadata={},
        ),
    ]

    retriever.index_documents(docs)
    results = retriever.retrieve("What was Apple's revenue?", top_k=2)

    print(f"Found {len(results)} results:")
    for result in results:
        print(f"Score: {result.score:.3f}, Content: {result.document.content}")
