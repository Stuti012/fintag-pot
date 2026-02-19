import argparse
import yaml
from pathlib import Path

from finqa.load_finqa import FinQALoader
from finqa.finqa_retriever import FinQARetriever
from finqa.program_generator import ProgramGenerator
from finqa.program_executor import ProgramExecutor
from finqa.finqa_eval import FinQAEvaluator

from edgar.edgar_download import EDGARDownloader
from indexing.build_index import IndexBuilder
from retrieval.temporal_decay import TemporalDecayRetriever
from retrieval.semantic_cache import SemanticCache

from llm.llm_client import LLMClient
from llm.prompts import get_edgar_answer_with_tables_prompt
from verification.numeric_verifier import NumericVerifier

from evaluation.compare_baseline_vs_proposed import BaselineVsProposedEvaluator


class FinQARAGSystem:
    """Main Financial QA system with Program-of-Thought reasoning"""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.config_path = config_path
        
        # Initialize components
        self.llm_client = LLMClient(config_path)
        self.verifier = NumericVerifier(config_path)
        
        # FinQA components
        self.finqa_loader = FinQALoader()
        self.finqa_retriever = FinQARetriever(config_path)
        self.program_generator = ProgramGenerator(config_path)
        self.program_executor = ProgramExecutor()
        
        # EDGAR components
        self.edgar_downloader = EDGARDownloader(config_path)
        self.index_builder = IndexBuilder(config_path)
        self.edgar_retriever = TemporalDecayRetriever(
            collection_name=self.config['indexing']['collection_name_edgar'],
            config_path=config_path
        )

        cache_cfg = self.config.get('semantic_cache', {})
        cache_threshold = cache_cfg.get('similarity_threshold', 0.95)
        cache_size = cache_cfg.get('max_size', 500)
        embedding_model = self.config['embedding']['model']

        self.finqa_cache = SemanticCache(
            model_name=embedding_model,
            similarity_threshold=cache_threshold,
            max_size=cache_size,
        )
        self.edgar_cache = SemanticCache(
            model_name=embedding_model,
            similarity_threshold=cache_threshold,
            max_size=cache_size,
        )
    
    def query_finqa(self, question: str, question_id: str) -> dict:
        """Query FinQA system with Program-of-Thought
        
        Args:
            question: Question text
            question_id: ID of the question (for retrieving from same document)
        
        Returns:
            Answer with program and evidence
        """
        print(f"\nQuestion: {question}")
        cache_key = f"{question_id}::{question}"
        cached_result = self.finqa_cache.get(cache_key)
        if cached_result:
            print("Semantic cache hit (FinQA)")
            return cached_result

        print("Retrieving evidence...")
        
        # Retrieve evidence
        retrieved = self.finqa_retriever.retrieve_for_question(
            question, question_id, top_k=self.config['retrieval']['top_k']
        )
        
        print(f"Retrieved {len(retrieved)} evidence pieces")
        
        # Extract available numbers
        available_numbers = self.program_generator.extract_numbers_from_evidence(retrieved)
        print(f"Found {len(available_numbers)} numbers in evidence")
        
        # Generate program with optional self-consistency
        print("Generating program...")
        use_self_consistency = self.config['program_execution'].get('enable_self_consistency', False)

        if use_self_consistency:
            (
                program_text,
                reasoning,
                success,
                error,
                voted_answer,
                sample_runs,
            ) = self.program_generator.generate_with_self_consistency(
                question,
                retrieved,
                available_numbers,
            )
        else:
            voted_answer = None
            sample_runs = []
            program_text, reasoning, success, error = self.program_generator.generate_with_repair(
                question, retrieved, available_numbers
            )
        
        if not success:
            return {
                'success': False,
                'error': error,
                'question': question
            }
        
        # Execute program
        print("Executing program...")
        final_answer, program_obj, exec_error = self.program_executor.execute(program_text)

        if voted_answer is not None:
            final_answer = voted_answer
        
        if exec_error:
            return {
                'success': False,
                'error': exec_error,
                'program': program_text,
                'question': question
            }
        
        # Format result
        result = {
            'success': True,
            'question': question,
            'answer': final_answer,
            'program': program_text,
            'reasoning': reasoning,
            'intermediate_values': program_obj.variables,
            'evidence': [
                {
                    'content': r.document.content,
                    'score': r.score,
                    'type': r.document.doc_type
                }
                for r in retrieved
            ],
            'self_consistency': {
                'enabled': use_self_consistency,
                'samples': sample_runs,
            }
        }
        
        print(f"\nAnswer: {final_answer}")
        print(f"Program:\n{program_text}")

        self.finqa_cache.put(cache_key, result)
        return result
    
    def query_edgar(self, query: str, ticker: str = None) -> dict:
        """Query EDGAR filings
        
        Args:
            query: Natural language query
            ticker: Optional ticker to filter by
        
        Returns:
            Answer with citations
        """
        print(f"\nQuery: {query}")
        if ticker:
            print(f"Ticker: {ticker}")

        cache_key = f"{ticker or 'ALL'}::{query}"
        cached_result = self.edgar_cache.get(cache_key)
        if cached_result:
            print("Semantic cache hit (EDGAR)")
            return cached_result

        print("Retrieving from EDGAR filings...")
        
        # Retrieve with temporal decay
        results = self.edgar_retriever.retrieve_recent_filings(
            query=query,
            ticker=ticker,
            top_k=self.config['retrieval']['top_k']
        )
        
        if not results:
            return {
                'success': False,
                'error': 'No relevant filings found',
                'query': query
            }
        
        print(f"Retrieved {len(results)} chunks")
        
        # Separate text and table evidence
        text_evidence = []
        table_evidence = []
        
        for result in results:
            if result.document.doc_type == 'table':
                table_evidence.append(result)
            else:
                text_evidence.append(result)
        
        # Format evidence
        text_str = "\n\n".join([
            f"[{r.document.metadata.get('ticker', '')} - "
            f"{r.document.metadata.get('filing_type', '')} - "
            f"{r.document.metadata.get('filing_date', '')}]\n"
            f"{r.document.content}"
            for r in text_evidence
        ])
        
        table_str = "\n\n".join([
            f"[Table from {r.document.metadata.get('filing_type', '')} "
            f"{r.document.metadata.get('filing_date', '')}]\n"
            f"{r.document.content}"
            for r in table_evidence
        ])
        
        # Generate answer
        print("Generating answer...")
        
        if table_evidence:
            prompt = get_edgar_answer_with_tables_prompt(query, text_str, table_str)
        else:
            from llm.prompts import get_edgar_answer_prompt
            prompt = get_edgar_answer_prompt(query, text_str)
        
        answer = self.llm_client.generate(prompt, temperature=0.0)
        
        # Verify numbers
        verification = self.verifier.verify_answer(answer, results)
        
        # Format result
        result = {
            'success': True,
            'query': query,
            'answer': answer,
            'citations': [
                {
                    'ticker': r.document.metadata.get('ticker', ''),
                    'filing_type': r.document.metadata.get('filing_type', ''),
                    'filing_date': r.document.metadata.get('filing_date', ''),
                    'section': r.document.metadata.get('section', ''),
                    'score': r.score
                }
                for r in results
            ],
            'verification': verification
        }
        
        print(f"\nAnswer:\n{answer}")
        
        if not verification['verified']:
            print(f"\nWarning: {verification['message']}")
        
        self.edgar_cache.put(cache_key, result)
        return result


def main():
    parser = argparse.ArgumentParser(description="Financial QA System with PoT")
    parser.add_argument('--mode', choices=['finqa', 'edgar', 'eval', 'compare'], 
                       required=True, help='Operating mode')
    parser.add_argument('--query', type=str, help='Query text')
    parser.add_argument('--question-id', type=str, help='FinQA question ID')
    parser.add_argument('--ticker', type=str, help='Stock ticker for EDGAR mode')
    parser.add_argument('--download', action='store_true', 
                       help='Download EDGAR filings')
    parser.add_argument('--build-index', action='store_true',
                       help='Build search index')
    parser.add_argument('--eval-split', type=str, default='validation',
                       help='FinQA split to evaluate')
    parser.add_argument('--num-examples', type=int, default=100,
                       help='Number of examples to evaluate')
    
    args = parser.parse_args()
    
    system = FinQARAGSystem()
    
    if args.mode == 'finqa':
        # FinQA mode
        if not args.query or not args.question_id:
            print("Error: --query and --question-id required for FinQA mode")
            return
        
        # First, need to load and index the example
        example = system.finqa_loader.get_example_by_id(args.question_id)
        system.finqa_retriever.index_examples([example])
        
        # Query
        result = system.query_finqa(args.query, args.question_id)
        
        if result['success']:
            print(f"\n{'='*60}")
            print(f"Answer: {result['answer']}")
            print(f"\nProgram:\n{result['program']}")
        else:
            print(f"\nError: {result['error']}")
    
    elif args.mode == 'edgar':
        # EDGAR mode
        if args.download:
            # Download filings
            if not args.ticker:
                print("Error: --ticker required for download")
                return
            
            print(f"Downloading filings for {args.ticker}...")
            filings = system.edgar_downloader.download_recent_filings(args.ticker)
            print(f"Downloaded {len(filings)} filings")
        
        if args.build_index:
            # Build index
            print("Building EDGAR index...")
            system.index_builder.rebuild_index(args.ticker)
        
        if args.query:
            # Query EDGAR
            result = system.query_edgar(args.query, args.ticker)
            
            if result['success']:
                print(f"\n{'='*60}")
                print(f"Answer:\n{result['answer']}")
                print(f"\nCitations:")
                for cit in result['citations'][:3]:
                    print(f"  - {cit['ticker']} {cit['filing_type']} "
                          f"({cit['filing_date']})")
            else:
                print(f"\nError: {result['error']}")
    
    elif args.mode == 'eval':
        # Evaluation mode
        evaluator = FinQAEvaluator()
        
        # Index examples
        print(f"Loading {args.eval_split} split...")
        examples = evaluator.loader.load_split(args.eval_split)[:args.num_examples]
        
        print("Indexing examples...")
        evaluator.retriever.index_examples(examples)
        
        # Run evaluation
        metrics = evaluator.evaluate_split(args.eval_split, args.num_examples)
        evaluator.print_metrics(metrics)
    
    elif args.mode == 'compare':
        # Comparison mode
        comparator = BaselineVsProposedEvaluator()
        results = comparator.run_comparison(
            split=args.eval_split,
            num_examples=args.num_examples
        )


if __name__ == "__main__":
    main()
