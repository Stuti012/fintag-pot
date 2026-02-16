from typing import List, Dict, Any
from indexing.schema import FinQAExample, Document, DocumentType, SourceType, RetrievalResult
from retrieval.hybrid_retriever import HybridRetriever
import yaml


class FinQARetriever:
    """Retriever specifically for FinQA dataset"""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.retriever = HybridRetriever(
            collection_name=self.config['indexing']['collection_name_finqa'],
            config_path=config_path
        )
    
    def index_examples(self, examples: List[FinQAExample]):
        """Index FinQA examples for retrieval"""
        all_documents = []
        
        for example in examples:
            docs = self._convert_example_to_documents(example)
            all_documents.extend(docs)
        
        print(f"Indexing {len(all_documents)} documents from {len(examples)} examples...")
        self.retriever.index_documents(all_documents)
    
    def retrieve_for_question(self, 
                            question: str,
                            question_id: str,
                            top_k: int = None) -> List[RetrievalResult]:
        """Retrieve evidence for a FinQA question
        
        Args:
            question: The question text
            question_id: ID of the question to retrieve from
            top_k: Number of results to return
        """
        if top_k is None:
            top_k = self.config['retrieval']['top_k']
        
        # Retrieve using hybrid search
        results = self.retriever.retrieve(query=question, top_k=top_k * 2)  # Get more for filtering
        
        # Filter to only documents from this question
        filtered_results = []
        for result in results:
            if result.document.metadata.get('question_id') == question_id:
                filtered_results.append(result)
                if len(filtered_results) >= top_k:
                    break
        
        # Re-rank by keeping diverse types
        return self._rerank_by_type(filtered_results, top_k)
    
    def _convert_example_to_documents(self, example: FinQAExample) -> List[Document]:
        """Convert FinQA example to indexable documents"""
        documents = []
        
        # Index pre-text
        for i, text in enumerate(example.pre_text):
            if text.strip():
                doc = Document(
                    doc_id=f"{example.id}_pre_{i}",
                    chunk_id="chunk_0",
                    content=text,
                    doc_type=DocumentType.TEXT,
                    source_type=SourceType.FINQA,
                    metadata={
                        'question_id': example.id,
                        'section': 'pre_text',
                        'index': i,
                        'question': example.question
                    }
                )
                documents.append(doc)
        
        # Index post-text
        for i, text in enumerate(example.post_text):
            if text.strip():
                doc = Document(
                    doc_id=f"{example.id}_post_{i}",
                    chunk_id="chunk_0",
                    content=text,
                    doc_type=DocumentType.TEXT,
                    source_type=SourceType.FINQA,
                    metadata={
                        'question_id': example.id,
                        'section': 'post_text',
                        'index': i,
                        'question': example.question
                    }
                )
                documents.append(doc)
        
        # Index table with multiple representations
        if example.table:
            # Raw table as markdown
            table_md = self._table_to_markdown(example.table)
            doc_table = Document(
                doc_id=f"{example.id}_table",
                chunk_id="chunk_0",
                content=table_md,
                doc_type=DocumentType.TABLE,
                source_type=SourceType.FINQA,
                metadata={
                    'question_id': example.id,
                    'section': 'table',
                    'table_data': example.table,
                    'question': example.question
                }
            )
            documents.append(doc_table)
            
            # Table summary for better retrieval
            table_summary = self._generate_table_summary(example.table)
            doc_summary = Document(
                doc_id=f"{example.id}_table_summary",
                chunk_id="chunk_0",
                content=table_summary,
                doc_type=DocumentType.TABLE_SUMMARY,
                source_type=SourceType.FINQA,
                metadata={
                    'question_id': example.id,
                    'section': 'table_summary',
                    'question': example.question,
                    'original_table_id': f"{example.id}_table"
                }
            )
            documents.append(doc_summary)
        
        return documents
    
    def _table_to_markdown(self, table: List[List[Any]]) -> str:
        """Convert table to markdown"""
        if not table:
            return ""
        
        lines = []
        for i, row in enumerate(table):
            row_str = " | ".join(str(cell) for cell in row)
            lines.append(f"| {row_str} |")
            
            if i == 0:
                sep = " | ".join(["---"] * len(row))
                lines.append(f"| {sep} |")
        
        return "\n".join(lines)
    
    def _generate_table_summary(self, table: List[List[Any]]) -> str:
        """Generate a text summary of table for better retrieval"""
        if not table or len(table) < 2:
            return ""
        
        summary_parts = []
        
        # Headers
        headers = [str(h) for h in table[0]]
        summary_parts.append(f"Table with columns: {', '.join(headers)}")
        
        # Extract numeric patterns
        for col_idx, header in enumerate(headers):
            values = []
            for row in table[1:]:
                if col_idx < len(row):
                    try:
                        val = float(str(row[col_idx]).replace(',', '').replace('$', '').replace('%', ''))
                        values.append(val)
                    except (ValueError, AttributeError):
                        pass
            
            if values:
                summary_parts.append(f"{header}: values include {', '.join([str(v) for v in values[:5]])}")
        
        return ". ".join(summary_parts)
    
    def _rerank_by_type(self, results: List[RetrievalResult], top_k: int) -> List[RetrievalResult]:
        """Re-rank to ensure diverse document types"""
        # Separate by type
        text_docs = [r for r in results if r.document.doc_type == DocumentType.TEXT]
        table_docs = [r for r in results if r.document.doc_type == DocumentType.TABLE]
        summary_docs = [r for r in results if r.document.doc_type == DocumentType.TABLE_SUMMARY]
        
        # Combine with priority: table > text > summary
        reranked = []
        
        # Add top tables
        reranked.extend(table_docs[:2])
        
        # Add top text
        reranked.extend(text_docs[:top_k - 2])
        
        # Fill remaining with summaries
        remaining = top_k - len(reranked)
        if remaining > 0:
            reranked.extend(summary_docs[:remaining])
        
        # Update ranks
        for i, result in enumerate(reranked):
            result.rank = i + 1
        
        return reranked[:top_k]


if __name__ == "__main__":
    from finqa.load_finqa import FinQALoader
    
    # Test
    loader = FinQALoader()
    examples = loader.load_split("validation")[:10]  # Test with small subset
    
    retriever = FinQARetriever()
    retriever.index_examples(examples)
    
    # Test retrieval
    ex = examples[0]
    results = retriever.retrieve_for_question(ex.question, ex.id, top_k=5)
    
    print(f"Question: {ex.question}")
    print(f"\nRetrieved {len(results)} documents:")
    for i, result in enumerate(results):
        print(f"{i+1}. Type: {result.document.doc_type}, Score: {result.score:.3f}")
        print(f"   Content preview: {result.document.content[:100]}...")
