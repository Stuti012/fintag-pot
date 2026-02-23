from typing import List, Optional
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from indexing.schema import Document, RetrievalResult
import yaml
from pathlib import Path
import pickle

from retrieval.semantic_cache import SemanticCache

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    chromadb = None
    Settings = None

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None


class HybridRetriever:
    """Hybrid retrieval using BM25 + Vector search (with BM25 fallback)."""

    def __init__(self, collection_name: str, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.collection_name = collection_name
        self.top_k = self.config['retrieval']['top_k']
        self.bm25_weight = self.config['retrieval']['bm25_weight']
        self.vector_weight = self.config['retrieval']['vector_weight']
        self.enable_reranking = self.config['retrieval'].get('enable_reranking', False)
        self.rerank_weight = self.config['retrieval'].get('rerank_weight', 0.3)
        self.rerank_candidates_multiplier = self.config['retrieval'].get('rerank_candidates_multiplier', 3)
        self.enable_semantic_cache = self.config['retrieval'].get('enable_semantic_cache', False)

        print(f"Loading embedding model: {self.config['embedding']['model']}")
        self.embedding_model = SentenceTransformer(self.config['embedding']['model'])

        self.cross_encoder = None
        if self.enable_reranking:
            rerank_model_name = self.config['retrieval'].get(
                'rerank_model',
                'cross-encoder/ms-marco-MiniLM-L-6-v2'
            )
            if CrossEncoder is None:
                print("Warning: CrossEncoder unavailable, reranking disabled")
                self.enable_reranking = False
            else:
                print(f"Loading reranker model: {rerank_model_name}")
                self.cross_encoder = CrossEncoder(rerank_model_name)

        self.semantic_cache = None
        if self.enable_semantic_cache:
            cache_threshold = self.config['retrieval'].get('semantic_cache_threshold', 0.95)
            print(f"Initializing semantic cache (threshold={cache_threshold})")
            self.semantic_cache = SemanticCache(
                model_name=self.config['embedding']['model'],
                similarity_threshold=cache_threshold,
            )

        persist_dir = Path(self.config['indexing']['chroma_persist_dir'])
        persist_dir.mkdir(parents=True, exist_ok=True)

        self.chroma_available = chromadb is not None
        self.chroma_client = None
        self.collection = None
        if self.chroma_available:
            self.chroma_client = chromadb.PersistentClient(
                path=str(persist_dir),
                settings=Settings(anonymized_telemetry=False)
            )

            try:
                self.collection = self.chroma_client.get_collection(name=collection_name)
                print(f"Loaded existing collection: {collection_name}")
            except Exception:
                self.collection = self.chroma_client.create_collection(
                    name=collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
                print(f"Created new collection: {collection_name}")
        else:
            print("Warning: chromadb is not installed. Using BM25-only retrieval mode.")
            self.vector_weight = 0.0
            self.bm25_weight = 1.0

        self.bm25_index = None
        self.bm25_docs = []
        self.bm25_index_path = persist_dir / f"{collection_name}_bm25.pkl"
        self._load_bm25_index()

    def index_documents(self, documents: List[Document]):
        """Index documents for hybrid retrieval."""
        if not documents:
            return

        print(f"Indexing {len(documents)} documents...")

        ids = []
        contents = []
        embeddings = []
        metadatas = []

        for doc in documents:
            doc_full_id = f"{doc.doc_id}_{doc.chunk_id}"
            ids.append(doc_full_id)
            contents.append(doc.content)

            metadata = {
                'doc_id': doc.doc_id,
                'chunk_id': doc.chunk_id,
                'doc_type': doc.doc_type.value if hasattr(doc.doc_type, 'value') else str(doc.doc_type),
                'source_type': doc.source_type.value if hasattr(doc.source_type, 'value') else str(doc.source_type),
            }
            for key, value in doc.metadata.items():
                metadata[key] = value if isinstance(value, (str, int, float, bool)) else str(value)
            metadatas.append(metadata)

        if self.chroma_available and self.collection is not None:
            batch_size = self.config['embedding']['batch_size']
            for i in range(0, len(contents), batch_size):
                batch = contents[i:i + batch_size]
                batch_embeddings = self.embedding_model.encode(
                    batch,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
                embeddings.extend(batch_embeddings.tolist())

            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=contents,
                metadatas=metadatas,
            )

        self.bm25_docs.extend(documents)
        self.bm25_index = BM25Okapi([doc.content.lower().split() for doc in self.bm25_docs])
        self._save_bm25_index()

        print(f"Indexed {len(documents)} documents successfully")

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[RetrievalResult]:
        """Hybrid retrieval using BM25 + Vector search."""
        if top_k is None:
            top_k = self.top_k

        if self.semantic_cache is not None:
            cached_results = self.semantic_cache.get(query)
            if cached_results is not None:
                return cached_results[:top_k]

        retrieval_pool_size = top_k * self.rerank_candidates_multiplier if self.enable_reranking else top_k

        bm25_scores = {}
        if self.bm25_index is not None:
            tokenized_query = query.lower().split()
            scores = self.bm25_index.get_scores(tokenized_query)
            for i, score in enumerate(scores):
                doc = self.bm25_docs[i]
                bm25_scores[f"{doc.doc_id}_{doc.chunk_id}"] = float(score)

        combined_scores = {}

        if self.chroma_available and self.collection is not None and self.bm25_docs:
            query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)[0].tolist()
            vector_results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(retrieval_pool_size * 2, len(self.bm25_docs)),
            )

            for i, doc_id in enumerate(vector_results['ids'][0]):
                distance = vector_results['distances'][0][i]
                vector_score = 1 / (1 + distance)
                combined_scores[doc_id] = self.vector_weight * vector_score

        if bm25_scores:
            max_bm25 = max(bm25_scores.values()) if bm25_scores else 1
            for doc_id, score in bm25_scores.items():
                normalized_score = score / max_bm25 if max_bm25 > 0 else 0
                combined_scores[doc_id] = combined_scores.get(doc_id, 0) + self.bm25_weight * normalized_score

        sorted_doc_ids = sorted(combined_scores.keys(), key=lambda x: combined_scores[x], reverse=True)

        results = []
        for rank, doc_id in enumerate(sorted_doc_ids[:retrieval_pool_size]):
            doc = self._find_document_by_id(doc_id)
            if doc:
                results.append(RetrievalResult(document=doc, score=combined_scores[doc_id], rank=rank + 1))

        if self.enable_reranking and self.cross_encoder and len(results) > top_k:
            pairs = [[query, r.document.content] for r in results]
            rerank_scores = self.cross_encoder.predict(pairs)
            for i, rerank_score in enumerate(rerank_scores):
                results[i].score = (
                    (1.0 - self.rerank_weight) * results[i].score
                    + self.rerank_weight * float(rerank_score)
                )
            results.sort(key=lambda x: x.score, reverse=True)

        final_results = []
        for idx, result in enumerate(results[:top_k], start=1):
            result.rank = idx
            final_results.append(result)

        if self.semantic_cache is not None:
            self.semantic_cache.put(query, final_results)

        return final_results

    def _find_document_by_id(self, full_doc_id: str) -> Optional[Document]:
        for doc in self.bm25_docs:
            if f"{doc.doc_id}_{doc.chunk_id}" == full_doc_id:
                return doc
        return None

    def _save_bm25_index(self):
        data = {
            'index': self.bm25_index,
            'docs': self.bm25_docs,
        }
        with open(self.bm25_index_path, 'wb') as f:
            pickle.dump(data, f)

    def _load_bm25_index(self):
        if self.bm25_index_path.exists():
            try:
                with open(self.bm25_index_path, 'rb') as f:
                    data = pickle.load(f)
                    self.bm25_index = data['index']
                    self.bm25_docs = data['docs']
                print(f"Loaded BM25 index with {len(self.bm25_docs)} documents")
            except Exception as e:
                print(f"Failed to load BM25 index: {e}")

    def clear_collection(self):
        if self.chroma_available and self.chroma_client is not None:
            try:
                self.chroma_client.delete_collection(name=self.collection_name)
                print(f"Deleted collection: {self.collection_name}")
            except Exception:
                pass

            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )

        self.bm25_index = None
        self.bm25_docs = []
        if self.bm25_index_path.exists():
            self.bm25_index_path.unlink()


if __name__ == "__main__":
    from indexing.schema import Document, DocumentType, SourceType

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
