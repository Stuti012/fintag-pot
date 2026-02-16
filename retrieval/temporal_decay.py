from typing import List, Optional
from datetime import datetime
import math
from indexing.schema import RetrievalResult, EDGARDocument
from retrieval.hybrid_retriever import HybridRetriever
import yaml


class TemporalDecayRetriever(HybridRetriever):
    """Retriever with temporal decay for EDGAR documents"""
    
    def __init__(self, collection_name: str, config_path: str = "config.yaml"):
        super().__init__(collection_name, config_path)
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.lambda_decay = config['temporal_decay']['lambda']
        self.enabled = config['temporal_decay']['enabled']
    
    def retrieve_with_temporal_decay(self,
                                    query: str,
                                    top_k: Optional[int] = None,
                                    reference_date: Optional[datetime] = None) -> List[RetrievalResult]:
        """Retrieve with temporal decay applied
        
        Score formula: S_final = S_semantic × e^(-λ(t_now - t_doc))
        """
        # Get base retrieval results
        results = self.retrieve(query, top_k)
        
        if not self.enabled:
            return results
        
        # Apply temporal decay
        if reference_date is None:
            reference_date = datetime.now()
        
        adjusted_results = []
        
        for result in results:
            doc = result.document
            
            # Extract filing date from metadata
            filing_date = None
            if isinstance(doc, EDGARDocument):
                filing_date = doc.filing_date
            elif 'filing_date' in doc.metadata:
                filing_date_str = doc.metadata['filing_date']
                if isinstance(filing_date_str, str):
                    try:
                        filing_date = datetime.fromisoformat(filing_date_str)
                    except:
                        try:
                            filing_date = datetime.strptime(filing_date_str, "%Y-%m-%d")
                        except:
                            pass
                elif isinstance(filing_date_str, datetime):
                    filing_date = filing_date_str
            
            if filing_date:
                # Calculate time difference in days
                time_diff = (reference_date - filing_date).days
                
                # Apply exponential decay
                decay_factor = math.exp(-self.lambda_decay * time_diff / 365)  # Normalize by years
                
                # Adjust score
                adjusted_score = result.score * decay_factor
                
                result.score = adjusted_score
                result.document.metadata['temporal_decay_factor'] = decay_factor
                result.document.metadata['days_old'] = time_diff
            
            adjusted_results.append(result)
        
        # Re-sort by adjusted scores
        adjusted_results.sort(key=lambda x: x.score, reverse=True)
        
        # Update ranks
        for i, result in enumerate(adjusted_results):
            result.rank = i + 1
        
        return adjusted_results
    
    def retrieve_recent_filings(self,
                               query: str,
                               ticker: Optional[str] = None,
                               filing_type: Optional[str] = None,
                               top_k: Optional[int] = None,
                               max_days_old: Optional[int] = 365) -> List[RetrievalResult]:
        """Retrieve recent filings with filters"""
        # Get results with temporal decay
        results = self.retrieve_with_temporal_decay(query, top_k=top_k * 2 if top_k else None)
        
        # Filter by ticker
        if ticker:
            results = [r for r in results if r.document.metadata.get('ticker', '').upper() == ticker.upper()]
        
        # Filter by filing type
        if filing_type:
            results = [r for r in results if r.document.metadata.get('filing_type') == filing_type]
        
        # Filter by age
        if max_days_old:
            results = [r for r in results if r.document.metadata.get('days_old', float('inf')) <= max_days_old]
        
        # Limit to top_k
        if top_k:
            results = results[:top_k]
        
        return results
    
    def get_most_recent_filing(self,
                              ticker: str,
                              filing_type: str = "10-K") -> Optional[RetrievalResult]:
        """Get the most recent filing for a ticker"""
        results = self.retrieve_recent_filings(
            query=f"{ticker} {filing_type}",
            ticker=ticker,
            filing_type=filing_type,
            top_k=1
        )
        
        return results[0] if results else None
    
    def compare_across_periods(self,
                             query: str,
                             ticker: str,
                             num_periods: int = 4) -> List[RetrievalResult]:
        """Retrieve information across multiple time periods"""
        # Get more results initially
        results = self.retrieve_recent_filings(
            query=query,
            ticker=ticker,
            top_k=num_periods * 3
        )
        
        # Group by filing date and select most relevant from each period
        period_results = {}
        
        for result in results:
            filing_date = result.document.metadata.get('filing_date')
            if filing_date:
                if isinstance(filing_date, str):
                    try:
                        filing_date = datetime.fromisoformat(filing_date)
                    except:
                        filing_date = datetime.strptime(filing_date, "%Y-%m-%d")
                
                period_key = filing_date.strftime("%Y-%m")
                
                if period_key not in period_results:
                    period_results[period_key] = result
                elif result.score > period_results[period_key].score:
                    period_results[period_key] = result
        
        # Get top N periods by recency
        sorted_periods = sorted(period_results.items(), key=lambda x: x[0], reverse=True)
        final_results = [result for _, result in sorted_periods[:num_periods]]
        
        return final_results


if __name__ == "__main__":
    from indexing.schema import EDGARDocument, SourceType
    
    # Test
    retriever = TemporalDecayRetriever("test_temporal_collection")
    retriever.clear_collection()
    
    # Create test documents with different dates
    docs = []
    for i in range(3):
        filing_date = datetime(2024 - i, 12, 31)
        doc = EDGARDocument(
            doc_id=f"doc_{i}",
            chunk_id="chunk_0",
            content=f"Apple Inc. reported strong financial results for fiscal year {2024-i}. Revenue increased significantly.",
            doc_type="text",
            source_type=SourceType.EDGAR,
            ticker="AAPL",
            cik="0000320193",
            filing_type="10-K",
            filing_date=filing_date,
            accession_number=f"0000320193-{2024-i}-00001",
            metadata={
                'filing_date': filing_date.isoformat(),
                'ticker': 'AAPL',
                'filing_type': '10-K'
            }
        )
        docs.append(doc)
    
    retriever.index_documents(docs)
    
    # Test retrieval with temporal decay
    results = retriever.retrieve_with_temporal_decay("Apple revenue", top_k=3)
    
    print("Results with temporal decay:")
    for r in results:
        print(f"Score: {r.score:.4f}, Days old: {r.document.metadata.get('days_old')}, "
              f"Decay: {r.document.metadata.get('temporal_decay_factor', 0):.4f}")
