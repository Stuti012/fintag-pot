from pathlib import Path
from typing import List, Dict
from datetime import datetime
import yaml
from tqdm import tqdm

from indexing.schema import EDGARDocument, Document, DocumentType, SourceType, TableDocument
from edgar.html_to_md import HTMLToMarkdown
from edgar.section_splitter import SECSectionSplitter
from edgar.table_parser import TableParser
from retrieval.temporal_decay import TemporalDecayRetriever
from llm.llm_client import LLMClient
from llm.prompts import get_table_summary_prompt


class IndexBuilder:
    """Build searchable index from EDGAR filings"""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.html_converter = HTMLToMarkdown()
        self.section_splitter = SECSectionSplitter()
        self.table_parser = TableParser()
        self.llm_client = LLMClient(config_path)
        
        # Initialize retriever
        self.retriever = TemporalDecayRetriever(
            collection_name=self.config['indexing']['collection_name_edgar'],
            config_path=config_path
        )
        
        self.data_path = Path(self.config['edgar']['data_path'])
        self.table_summary_enabled = self.config['indexing']['table_summary_enabled']
    
    def build_index_from_filings(self, 
                                filing_infos: List[Dict],
                                clear_existing: bool = False):
        """Build index from downloaded filings"""
        if clear_existing:
            print("Clearing existing index...")
            self.retriever.clear_collection()
        
        all_documents = []
        
        for filing_info in tqdm(filing_infos, desc="Processing filings"):
            docs = self.process_filing(filing_info)
            all_documents.extend(docs)
        
        print(f"\nIndexing {len(all_documents)} documents...")
        self.retriever.index_documents(all_documents)
        
        print("Index built successfully!")
    
    def process_filing(self, filing_info: Dict) -> List[Document]:
        """Process a single filing into indexable documents"""
        filepath = Path(filing_info['filepath'])
        
        if not filepath.exists():
            print(f"File not found: {filepath}")
            return []
        
        # Read HTML content
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            html_content = f.read()
        
        # Convert to markdown
        markdown_content, tables = self.html_converter.convert_with_table_extraction(html_content)
        
        # Split into sections
        filing_type = filing_info.get('filing_type', '10-K')
        sections = self.section_splitter.split_filing(markdown_content, filing_type)
        
        # Get important sections only
        sections = self.section_splitter.get_important_sections(sections)
        
        # Chunk sections
        chunk_size = self.config['retrieval']['chunk_size']
        chunk_overlap = self.config['retrieval']['chunk_overlap']
        chunks = self.section_splitter.split_into_chunks(sections, chunk_size, chunk_overlap)
        
        # Parse filing date
        filing_date_str = filing_info.get('filing_date', '')
        try:
            filing_date = datetime.strptime(filing_date_str, '%Y-%m-%d')
        except:
            filing_date = datetime.now()
        
        # Create documents from chunks
        documents = []
        
        for chunk in chunks:
            doc = EDGARDocument(
                doc_id=f"{filing_info['accession_number']}_{chunk['section']}",
                chunk_id=f"chunk_{chunk['chunk_id']}",
                content=chunk['content'],
                doc_type=DocumentType.TEXT,
                source_type=SourceType.EDGAR,
                ticker=filing_info['ticker'],
                cik=filing_info.get('cik', ''),
                filing_type=filing_info['filing_type'],
                filing_date=filing_date,
                section=chunk['section'],
                accession_number=filing_info['accession_number'],
                metadata={
                    'filing_date': filing_date.isoformat(),
                    'ticker': filing_info['ticker'],
                    'filing_type': filing_info['filing_type'],
                    'section': chunk['section'],
                    'word_count': chunk['word_count']
                }
            )
            documents.append(doc)
        
        # Add table documents
        for table_id, table_data in tables.items():
            table_doc = self._create_table_document(
                table_data,
                filing_info,
                filing_date
            )
            documents.append(table_doc)
            
            # Add table summary if enabled
            if self.table_summary_enabled:
                summary_doc = self._create_table_summary_document(
                    table_data,
                    filing_info,
                    filing_date
                )
                if summary_doc:
                    documents.append(summary_doc)
        
        return documents
    
    def _create_table_document(self, 
                              table_data: Dict,
                              filing_info: Dict,
                              filing_date: datetime) -> TableDocument:
        """Create document from table"""
        table_id = table_data['table_id']
        
        doc = TableDocument(
            doc_id=f"{filing_info['accession_number']}_table_{table_id}",
            chunk_id="chunk_0",
            content=table_data['markdown'],
            doc_type=DocumentType.TABLE,
            source_type=SourceType.EDGAR,
            table_data=table_data['rows'],
            headers=table_data['headers'],
            summary=table_data.get('caption', ''),
            metadata={
                'filing_date': filing_date.isoformat(),
                'ticker': filing_info['ticker'],
                'filing_type': filing_info['filing_type'],
                'table_id': table_id,
                'num_rows': table_data['num_rows'],
                'num_cols': table_data['num_cols'],
                'caption': table_data.get('caption', '')
            }
        )
        
        return doc
    
    def _create_table_summary_document(self,
                                      table_data: Dict,
                                      filing_info: Dict,
                                      filing_date: datetime) -> Document:
        """Create summary document for better retrieval"""
        table_id = table_data['table_id']
        
        # Generate summary using LLM if table is complex
        if table_data['num_rows'] > 5:
            try:
                prompt = get_table_summary_prompt(table_data['markdown'])
                summary = self.llm_client.generate(prompt, temperature=0.0)
            except:
                summary = table_data.get('summary', 'Financial table')
        else:
            summary = table_data.get('summary', 'Financial table')
        
        doc = Document(
            doc_id=f"{filing_info['accession_number']}_table_{table_id}_summary",
            chunk_id="chunk_0",
            content=summary,
            doc_type=DocumentType.TABLE_SUMMARY,
            source_type=SourceType.EDGAR,
            metadata={
                'filing_date': filing_date.isoformat(),
                'ticker': filing_info['ticker'],
                'filing_type': filing_info['filing_type'],
                'table_id': table_id,
                'original_table_doc_id': f"{filing_info['accession_number']}_table_{table_id}"
            }
        )
        
        return doc
    
    def rebuild_index(self, ticker: str = None):
        """Rebuild index from local filings"""
        from edgar.edgar_download import EDGARDownloader
        
        downloader = EDGARDownloader()
        filings = downloader.get_local_filings(ticker)
        
        if not filings:
            print("No local filings found")
            return
        
        print(f"Found {len(filings)} local filings")
        self.build_index_from_filings(filings, clear_existing=True)


if __name__ == "__main__":
    # Test
    builder = IndexBuilder()
    
    # This would normally use actual downloaded filings
    # For testing, create mock filing info
    mock_filing = {
        'ticker': 'AAPL',
        'cik': '0000320193',
        'filing_type': '10-K',
        'accession_number': '0000320193-2023-000077',
        'filing_date': '2023-11-03',
        'filepath': './data/edgar/AAPL/10-K/0000320193-2023-000077_2023-11-03.html'
    }
    
    # Check if file exists before processing
    if Path(mock_filing['filepath']).exists():
        docs = builder.process_filing(mock_filing)
        print(f"Created {len(docs)} documents from filing")
        
        for doc in docs[:5]:
            print(f"\n{doc.doc_type}: {doc.doc_id}")
            print(f"Content preview: {doc.content[:100]}...")
    else:
        print("No test filing available. Download filings first using edgar_download.py")
