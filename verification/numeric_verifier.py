import re
from typing import List, Dict, Tuple, Optional
from indexing.schema import RetrievalResult
from llm.llm_client import LLMClient
from llm.prompts import get_numeric_verification_prompt
import yaml


class NumericVerifier:
    """Verify numeric claims against evidence"""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.tolerance = self.config['verification']['tolerance']
        self.enabled = self.config['verification']['enable_numeric_verification']
        self.llm_client = LLMClient(config_path)
    
    def verify_answer(self,
                     answer: str,
                     retrieved_evidence: List[RetrievalResult]) -> Dict[str, any]:
        """Verify numeric claims in answer against evidence
        
        Returns:
            Dictionary with verification results
        """
        if not self.enabled:
            return {
                'verified': True,
                'message': 'Verification disabled'
            }
        
        # Extract numbers from answer
        answer_numbers = self.extract_numbers(answer)
        
        if not answer_numbers:
            return {
                'verified': True,
                'answer_numbers': [],
                'message': 'No numbers in answer'
            }
        
        # Extract numbers from evidence
        evidence_numbers = set()
        for result in retrieved_evidence:
            nums = self.extract_numbers(result.document.content)
            evidence_numbers.update(nums)
        
        # Verify each number
        verification_results = {}
        unverified_numbers = []
        
        for num in answer_numbers:
            verified = self._find_matching_number(num, evidence_numbers)
            verification_results[num] = verified
            
            if not verified:
                unverified_numbers.append(num)
        
        # Overall verification
        all_verified = len(unverified_numbers) == 0
        
        return {
            'verified': all_verified,
            'answer_numbers': answer_numbers,
            'evidence_numbers': list(evidence_numbers),
            'verification_details': verification_results,
            'unverified_numbers': unverified_numbers,
            'message': 'All numbers verified' if all_verified else f'{len(unverified_numbers)} numbers not found in evidence'
        }
    
    def verify_with_llm(self,
                       answer: str,
                       retrieved_evidence: List[RetrievalResult]) -> Dict[str, any]:
        """Use LLM to verify numeric claims"""
        # Format evidence
        evidence_text = "\n\n".join([
            f"[Evidence {i+1}]\n{result.document.content}"
            for i, result in enumerate(retrieved_evidence)
        ])
        
        # Get LLM verification
        prompt = get_numeric_verification_prompt(answer, evidence_text)
        
        try:
            verification_response = self.llm_client.generate(
                prompt=prompt,
                temperature=0.0
            )
            
            # Parse response
            return {
                'verified': 'PASS' in verification_response,
                'llm_verification': verification_response,
                'method': 'llm'
            }
        except Exception as e:
            return {
                'verified': False,
                'error': str(e),
                'method': 'llm'
            }
    
    def extract_numbers(self, text: str) -> List[float]:
        """Extract all numbers from text"""
        # Remove commas from numbers
        text = text.replace(',', '')
        
        # Find all numbers (including decimals, percentages, negatives)
        pattern = r'-?\$?\d+\.?\d*%?'
        matches = re.findall(pattern, text)
        
        numbers = []
        for match in matches:
            # Clean up
            clean = match.replace('$', '').replace('%', '').strip()
            
            try:
                num = float(clean)
                # Filter out years (common false positive)
                if not (1900 <= num <= 2100):
                    numbers.append(num)
            except ValueError:
                continue
        
        return numbers
    
    def _find_matching_number(self, target: float, candidates: set) -> bool:
        """Check if target number exists in candidates (with tolerance)"""
        for candidate in candidates:
            # Exact match
            if abs(target - candidate) < self.tolerance:
                return True
            
            # Relative tolerance for large numbers
            if candidate != 0:
                relative_diff = abs(target - candidate) / abs(candidate)
                if relative_diff < self.tolerance:
                    return True
        
        return False
    
    def verify_program_numbers(self,
                              program: str,
                              available_numbers: List[float]) -> Tuple[bool, List[float]]:
        """Verify all numbers in a program are available
        
        Returns:
            (all_verified, missing_numbers)
        """
        program_numbers = self.extract_numbers(program)
        
        missing = []
        for num in program_numbers:
            if not self._find_matching_number(num, set(available_numbers)):
                missing.append(num)
        
        return len(missing) == 0, missing
    
    def extract_numeric_claims(self, text: str) -> List[Dict[str, any]]:
        """Extract numeric claims with context
        
        Returns:
            List of claims with surrounding context
        """
        claims = []
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            numbers = self.extract_numbers(sentence)
            
            if numbers:
                claims.append({
                    'sentence': sentence.strip(),
                    'numbers': numbers,
                    'context': sentence.strip()
                })
        
        return claims
    
    def calculate_hallucination_rate(self,
                                    answers: List[str],
                                    evidences: List[List[RetrievalResult]]) -> float:
        """Calculate rate of hallucinated numbers across multiple answers"""
        total_numbers = 0
        hallucinated_numbers = 0
        
        for answer, evidence in zip(answers, evidences):
            verification = self.verify_answer(answer, evidence)
            
            total_numbers += len(verification['answer_numbers'])
            hallucinated_numbers += len(verification['unverified_numbers'])
        
        if total_numbers == 0:
            return 0.0
        
        return hallucinated_numbers / total_numbers


if __name__ == "__main__":
    # Test
    verifier = NumericVerifier()
    
    # Test number extraction
    test_text = "Revenue was $100.5 million in 2023, up from $85.3 million in 2022."
    numbers = verifier.extract_numbers(test_text)
    print(f"Extracted numbers: {numbers}")
    
    # Test verification
    from indexing.schema import Document, DocumentType, SourceType
    
    evidence_doc = Document(
        doc_id="test",
        chunk_id="chunk_0",
        content="The company reported revenue of $100.5 million and profit of $20.3 million.",
        doc_type=DocumentType.TEXT,
        source_type=SourceType.EDGAR,
        metadata={}
    )
    
    evidence = [RetrievalResult(document=evidence_doc, score=1.0, rank=1)]
    
    answer = "Revenue was $100.5 million"
    verification = verifier.verify_answer(answer, evidence)
    
    print(f"\nVerification result: {verification['verified']}")
    print(f"Message: {verification['message']}")
    
    # Test with hallucinated number
    bad_answer = "Revenue was $500 million"
    bad_verification = verifier.verify_answer(bad_answer, evidence)
    
    print(f"\nBad answer verification: {bad_verification['verified']}")
    print(f"Unverified numbers: {bad_verification['unverified_numbers']}")
