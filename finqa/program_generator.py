from collections import Counter
from typing import List, Tuple, Optional, Dict, Any
from indexing.schema import FinQAExample, RetrievalResult
from llm.llm_client import LLMClient
from llm.prompts import get_program_generation_prompt, get_program_repair_prompt
from finqa.program_executor import ProgramExecutor
import yaml


class ProgramGenerator:
    """Generate programs using Program-of-Thought reasoning"""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.llm_client = LLMClient(config_path)
        self.executor = ProgramExecutor(
            timeout_seconds=self.config['program_execution']['timeout_seconds']
        )
        self.max_retry = self.config['program_execution']['max_retry_attempts']
        self.self_consistency_config = self.config.get('program_generation', {}).get('self_consistency', {})
        self.self_consistency_samples = self.config['program_execution'].get('self_consistency_samples', 5)
        self.self_consistency_temperature = self.config['program_execution'].get('self_consistency_temperature', 0.7)
    
    def generate_program(self,
                        question: str,
                        retrieved_evidence: List[RetrievalResult],
                        available_numbers: Optional[List[float]] = None,
                        temperature: float = 0.0) -> Tuple[str, str, Optional[str]]:
        """Generate program for question with evidence
        
        Returns:
            (program_text, reasoning, error_message)
        """
        # Format evidence
        evidence_text = self._format_evidence(retrieved_evidence)
        
        # Generate program
        prompt = get_program_generation_prompt(question, evidence_text)
        
        try:
            llm_output = self.llm_client.generate(
                prompt=prompt,
                temperature=temperature
            )
            
            reasoning, program = self.executor.extract_reasoning_and_program(llm_output)
            
            # Verify program uses only available numbers
            if available_numbers:
                is_valid, error_msg = self.executor.verify_program(program, available_numbers)
                if not is_valid:
                    return program, reasoning, error_msg
            
            return program, reasoning, None
            
        except Exception as e:
            return "", "", f"Program generation failed: {str(e)}"
    
    def generate_with_repair(self,
                           question: str,
                           retrieved_evidence: List[RetrievalResult],
                           available_numbers: List[float]) -> Tuple[str, str, bool, Optional[str]]:
        """Generate program with repair loop
        
        Returns:
            (program, reasoning, success, error_message)
        """
        evidence_text = self._format_evidence(retrieved_evidence)
        
        for attempt in range(self.max_retry):
            # Generate program
            program, reasoning, gen_error = self.generate_program(
                question, retrieved_evidence, available_numbers
            )
            
            if gen_error:
                # Try to repair
                if attempt < self.max_retry - 1:
                    program = self._repair_program(
                        question, evidence_text, program, gen_error, available_numbers
                    )
                    continue
                else:
                    return program, reasoning, False, gen_error
            
            # Try to execute
            final_answer, program_obj, exec_error = self.executor.execute(program)
            
            if exec_error:
                # Try to repair
                if attempt < self.max_retry - 1:
                    program = self._repair_program(
                        question, evidence_text, program, exec_error, available_numbers
                    )
                    continue
                else:
                    return program, reasoning, False, exec_error
            
            # Success
            return program, reasoning, True, None
        
        return "", "", False, "Max retry attempts reached"

    def generate_with_self_consistency(
        self,
        question: str,
        retrieved_evidence: List[RetrievalResult],
        available_numbers: List[float],
        n: Optional[int] = None,
    ) -> Tuple[str, str, bool, Optional[str], Optional[str], List[Dict[str, Any]]]:
        """Generate multiple programs and select by majority-vote answer.

        Returns:
            (best_program, best_reasoning, success, error, voted_answer, sampled_runs)
        """
        num_samples = n or self.self_consistency_samples
        runs: List[Dict[str, Any]] = []

        for _ in range(num_samples):
            try:
                program, reasoning, gen_error = self.generate_program(
                    question=question,
                    retrieved_evidence=retrieved_evidence,
                    available_numbers=available_numbers,
                    temperature=self.self_consistency_temperature,
                )
                if gen_error:
                    runs.append(
                        {
                            'program': program,
                            'reasoning': reasoning,
                            'answer': None,
                            'success': False,
                            'error': gen_error,
                        }
                    )
                    continue

                final_answer, _, exec_error = self.executor.execute(program)
                if exec_error:
                    runs.append(
                        {
                            'program': program,
                            'reasoning': reasoning,
                            'answer': None,
                            'success': False,
                            'error': exec_error,
                        }
                    )
                    continue

                runs.append(
                    {
                        'program': program,
                        'reasoning': reasoning,
                        'answer': str(final_answer),
                        'success': True,
                        'error': None,
                    }
                )
            except Exception as exc:
                runs.append(
                    {
                        'program': '',
                        'reasoning': '',
                        'answer': None,
                        'success': False,
                        'error': str(exc),
                    }
                )

        successful_runs = [run for run in runs if run['success'] and run['answer'] is not None]
        if not successful_runs:
            return '', '', False, 'All self-consistency samples failed', None, runs

        voted_answer = Counter(run['answer'] for run in successful_runs).most_common(1)[0][0]

        selected_run = next(run for run in successful_runs if run['answer'] == voted_answer)
        return (
            selected_run['program'],
            selected_run['reasoning'],
            True,
            None,
            voted_answer,
            runs,
        )
    
    def generate_with_self_consistency(self,
                                       question: str,
                                       retrieved_evidence: List[RetrievalResult],
                                       available_numbers: List[float],
                                       n_samples: Optional[int] = None) -> Tuple[str, str, bool, Optional[str], Optional[float]]:
        """Generate multiple programs and return majority-vote result."""
        sample_count = n_samples or self.self_consistency_samples

        successful_runs: List[Tuple[str, str, float]] = []
        errors: List[str] = []

        for _ in range(sample_count):
            program, reasoning, error = self.generate_program(
                question,
                retrieved_evidence,
                available_numbers=available_numbers,
                temperature=0.7,
            )

            if error:
                errors.append(error)
                continue

            result, _, exec_error = self.executor.execute(program)
            if exec_error or result is None:
                errors.append(exec_error or "Empty execution result")
                continue

            try:
                numeric_result = float(result)
            except (TypeError, ValueError):
                errors.append(f"Non-numeric result: {result}")
                continue

            successful_runs.append((program, reasoning, numeric_result))

        if not successful_runs:
            message = errors[-1] if errors else "Self-consistency failed to produce any valid program"
            return "", "", False, message, None

        result_counter = Counter(round(run[2], 6) for run in successful_runs)
        majority_value, _ = result_counter.most_common(1)[0]

        for program, reasoning, value in successful_runs:
            if round(value, 6) == majority_value:
                return program, reasoning, True, None, value

        fallback_program, fallback_reasoning, fallback_value = successful_runs[0]
        return fallback_program, fallback_reasoning, True, None, fallback_value

    def _repair_program(self,
                       question: str,
                       evidence: str,
                       failed_program: str,
                       error: str,
                       available_numbers: List[float]) -> str:
        """Attempt to repair a failed program"""
        repair_prompt = get_program_repair_prompt(
            question, evidence, failed_program, error
        )
        
        # Add available numbers to help
        numbers_str = ", ".join([str(n) for n in available_numbers[:20]])  # Limit for token count
        repair_prompt += f"\n\nAVAILABLE NUMBERS: {numbers_str}"
        
        try:
            llm_output = self.llm_client.generate(
                prompt=repair_prompt,
                temperature=0.1  # Slightly higher for variety
            )
            
            _, repaired_program = self.executor.extract_reasoning_and_program(llm_output)
            return repaired_program
            
        except Exception as e:
            print(f"Program repair failed: {e}")
            return failed_program
    
    def _format_evidence(self, retrieved_evidence: List[RetrievalResult]) -> str:
        """Format retrieved evidence for prompt"""
        evidence_parts = []
        
        for i, result in enumerate(retrieved_evidence):
            doc = result.document
            evidence_parts.append(f"[Evidence {i+1}] (score: {result.score:.3f})")
            evidence_parts.append(f"Type: {doc.doc_type}")
            evidence_parts.append(f"Content: {doc.content}")
            evidence_parts.append("")
        
        return "\n".join(evidence_parts)
    
    def extract_numbers_from_evidence(self, retrieved_evidence: List[RetrievalResult]) -> List[float]:
        """Extract all numbers from retrieved evidence"""
        import re
        numbers = []
        
        for result in retrieved_evidence:
            content = result.document.content
            
            # Remove commas from numbers
            content = content.replace(',', '')
            
            # Find all numbers (including decimals and negatives)
            found = re.findall(r'-?\d+\.?\d*', content)
            
            for num_str in found:
                try:
                    num = float(num_str)
                    # Filter out years and very large numbers
                    if num < 1e10:
                        numbers.append(num)
                except ValueError:
                    continue
        
        # Remove duplicates while preserving order
        seen = set()
        unique_numbers = []
        for num in numbers:
            if num not in seen:
                seen.add(num)
                unique_numbers.append(num)
        
        return unique_numbers


if __name__ == "__main__":
    # Test
    generator = ProgramGenerator()
    
    # Mock evidence
    from indexing.schema import Document, DocumentType, SourceType
    
    mock_doc = Document(
        doc_id="test_1",
        chunk_id="chunk_0",
        content="Revenue in 2020 was $100 million. Revenue in 2021 was $150 million.",
        doc_type=DocumentType.TEXT,
        source_type=SourceType.FINQA,
        metadata={}
    )
    
    mock_result = RetrievalResult(
        document=mock_doc,
        score=0.95,
        rank=1
    )
    
    question = "What was the percentage increase in revenue from 2020 to 2021?"
    
    program, reasoning, error = generator.generate_program(
        question,
        [mock_result],
        available_numbers=[100.0, 150.0, 2020.0, 2021.0]
    )
    
    print("Program:", program)
    print("Reasoning:", reasoning)
    if error:
        print("Error:", error)
