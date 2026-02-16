import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
import numpy as np
from sentence_transformers import SentenceTransformer

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None
from indexing.schema import Document, RetrievalResult
import yaml
from pathlib import Path
import pickle


class HybridRetriever:
    """Hybrid retrieval using BM25 + Vector search"""
    
    def __init__(self, collection_name: str, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.collection_name = collection_name
        self.top_k = self.config['retrieval']['top_k']
        self.bm25_weight = self.config['retrieval']['bm25_weight']
        self.vector_weight = self.config['retrieval']['vector_weight']
        self.rerank_enabled = self.config['retrieval'].get('enable_reranking', False)
        self.rerank_weight = self.config['retrieval'].get('rerank_weight', 0.3)
        self.rerank_model_name = self.config['retrieval'].get('reranker_model', "cross-encoder/ms-marco-MiniLM-L-6-v2")
        
        # Initialize embedding model
        print(f"Loading embedding model: {self.config['embedding']['model']}")
        self.embedding_model = SentenceTransformer(self.config['embedding']['model'])

        self.cross_encoder = None
        if self.rerank_enabled:
            if CrossEncoder is None:
                print("CrossEncoder unavailable; install sentence-transformers with cross-encoder support. Skipping reranking.")
            else:
                try:
                    print(f"Loading reranker model: {self.rerank_model_name}")
                    self.cross_encoder = CrossEncoder(self.rerank_model_name)
                except Exception as e:
                    print(f"Failed to load reranker model ({self.rerank_model_name}): {e}")
        
        # Initialize ChromaDB
        persist_dir = Path(self.config['indexing']['chroma_persist_dir'])
        persist_dir.mkdir(parents=True, exist_ok=True)
        
        self.chroma_client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        try:
            self.collection = self.chroma_client.get_collection(name=collection_name)
            print(f"Loaded existing collection: {collection_name}")
        except:
            self.collection = self.chroma_client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            print(f"Created new collection: {collection_name}")
        
        # BM25 index
        self.bm25_index = None
        self.bm25_docs = []
        self.bm25_index_path = persist_dir / f"{collection_name}_bm25.pkl"
        
        # Load BM25 index if exists
        self._load_bm25_index()
    
    def index_documents(self, documents: List[Document]):
        """Index documents for hybrid retrieval"""
        if not documents:
            return
        
        print(f"Indexing {len(documents)} documents...")
        
        # Prepare data for ChromaDB
        ids = []
        contents = []
        embeddings = []
        metadatas = []
        
        # Prepare data for BM25
        bm25_docs = []
        
        for doc in documents:
            doc_full_id = f"{doc.doc_id}_{doc.chunk_id}"
            ids.append(doc_full_id)
            contents.append(doc.content)
            
            # Metadata for ChromaDB
            metadata = {
                'doc_id': doc.doc_id,
                'chunk_id': doc.chunk_id,
                'doc_type': doc.doc_type.value if hasattr(doc.doc_type, 'value') else str(doc.doc_type),
                'source_type': doc.source_type.value if hasattr(doc.source_type, 'value') else str(doc.source_type),
            }
            # Add custom metadata
            for key, value in doc.metadata.items():
                # ChromaDB requires simple types
                if isinstance(value, (str, int, float, bool)):
                    metadata[key] = value
                else:
                    metadata[key] = str(value)
            
            metadatas.append(metadata)
            
            # BM25 tokenization
            bm25_docs.append(doc.content.lower().split())
        
        # Generate embeddings in batches
        batch_size = self.config['embedding']['batch_size']
        for i in range(0, len(contents), batch_size):
            batch = contents[i:i+batch_size]
            batch_embeddings = self.embedding_model.encode(
                batch,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            embeddings.extend(batch_embeddings.tolist())
        
        # Add to ChromaDB
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=contents,
            metadatas=metadatas
        )
        
        # Build BM25 index
        self.bm25_docs.extend(documents)
        self.bm25_index = BM25Okapi([doc.content.lower().split() for doc in self.bm25_docs])
        
        # Save BM25 index
        self._save_bm25_index()
        
        print(f"Indexed {len(documents)} documents successfully")
    
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[RetrievalResult]:
        """Hybrid retrieval using BM25 + Vector search"""
        if top_k is None:
            top_k = self.top_k
        
        # Vector search via ChromaDB
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)[0].tolist()
        
        vector_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k * 2, len(self.bm25_docs))  # Get more for fusion
        )
        
        # BM25 search
        bm25_scores = {}
        if self.bm25_index is not None:
            tokenized_query = query.lower().split()
            scores = self.bm25_index.get_scores(tokenized_query)
            
            for i, score in enumerate(scores):
                doc = self.bm25_docs[i]
                doc_id = f"{doc.doc_id}_{doc.chunk_id}"
                bm25_scores[doc_id] = score
        
        # Combine scores
        combined_scores = {}
        
        # Process vector results
        for i, doc_id in enumerate(vector_results['ids'][0]):
            distance = vector_results['distances'][0][i]
            # Convert distance to similarity score (lower distance = higher similarity)
            vector_score = 1 / (1 + distance)
            combined_scores[doc_id] = self.vector_weight * vector_score
        
        # Add BM25 scores
        # Normalize BM25 scores
        if bm25_scores:
            max_bm25 = max(bm25_scores.values()) if bm25_scores else 1
            for doc_id, score in bm25_scores.items():
                normalized_score = score / max_bm25 if max_bm25 > 0 else 0
                if doc_id in combined_scores:
                    combined_scores[doc_id] += self.bm25_weight * normalized_score
                else:
                    combined_scores[doc_id] = self.bm25_weight * normalized_score
        
        # Sort by combined score
        sorted_doc_ids = sorted(combined_scores.keys(), key=lambda x: combined_scores[x], reverse=True)
        
        # Convert to RetrievalResult objects
        results = []
        candidate_doc_ids = sorted_doc_ids[: max(top_k * 3, top_k)]
        for rank, doc_id in enumerate(candidate_doc_ids):
            # Find document
            doc = self._find_document_by_id(doc_id)
            if doc:
                result = RetrievalResult(
                    document=doc,
                    score=combined_scores[doc_id],
                    rank=rank + 1
                )
                results.append(result)
        
        if self.cross_encoder is not None and results:
            results = self._rerank_results(query, results, top_k)
        else:
            results = results[:top_k]
            for rank, result in enumerate(results, start=1):
                result.rank = rank

        return results
    
    def _rerank_results(self, query: str, candidates: List[RetrievalResult], top_k: int) -> List[RetrievalResult]:
        """Rerank candidates with a cross-encoder and blend scores."""
        pairs = [[query, c.document.content] for c in candidates]
        rerank_scores = self.cross_encoder.predict(pairs)

        if len(rerank_scores) > 0:
            min_score = float(np.min(rerank_scores))
            max_score = float(np.max(rerank_scores))
            denom = (max_score - min_score) if max_score != min_score else 1.0
        else:
            min_score = 0.0
            denom = 1.0

        for i, candidate in enumerate(candidates):
            normalized_rerank = (float(rerank_scores[i]) - min_score) / denom
            candidate.score = ((1 - self.rerank_weight) * candidate.score) + (self.rerank_weight * normalized_rerank)

        candidates.sort(key=lambda x: x.score, reverse=True)
        reranked = candidates[:top_k]
        for rank, result in enumerate(reranked, start=1):
            result.rank = rank
        return reranked

    def _find_document_by_id(self, full_doc_id: str) -> Optional[Document]:
        """Find document by full ID"""
        for doc in self.bm25_docs:
            if f"{doc.doc_id}_{doc.chunk_id}" == full_doc_id:
                return doc
        return None
    
    def _save_bm25_index(self):
        """Save BM25 index to disk"""
        data = {
            'index': self.bm25_index,
            'docs': self.bm25_docs
        }
        with open(self.bm25_index_path, 'wb') as f:
            pickle.dump(data, f)
    
    def _load_bm25_index(self):
        """Load BM25 index from disk"""
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
        """Clear the collection"""
        try:
            self.chroma_client.delete_collection(name=self.collection_name)
            print(f"Deleted collection: {self.collection_name}")
        except:
            pass
        
        # Recreate
        self.collection = self.chroma_client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Clear BM25
        self.bm25_index = None
        self.bm25_docs = []
        if self.bm25_index_path.exists():
            self.bm25_index_path.unlink()


if __name__ == "__main__":
    from indexing.schema import Document, DocumentType, SourceType
    
    # Test
    retriever = HybridRetriever("test_collection")
    retriever.clear_collection()
    
    # Create test documents
    docs = [
        Document(
            doc_id="doc1",
            chunk_id="chunk0",
            content="Apple reported revenue of $394.3 billion in fiscal 2022.",
            doc_type=DocumentType.TEXT,
            source_type=SourceType.FINQA,
            metadata={}
        ),
        Document(
            doc_id="doc2",
            chunk_id="chunk0",
            content="Microsoft's revenue grew by 18% year over year.",
            doc_type=DocumentType.TEXT,
            source_type=SourceType.FINQA,
            metadata={}
        ),
    ]
    
    retriever.index_documents(docs)
    
    results = retriever.retrieve("What was Apple's revenue?", top_k=2)
    
    print(f"Found {len(results)} results:")
    for r in results:
        print(f"Score: {r.score:.3f}, Content: {r.document.content}")
