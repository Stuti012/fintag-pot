from typing import List, Dict, Tuple
import yaml
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np

from finqa.load_finqa import FinQALoader
from finqa.finqa_retriever import FinQARetriever
from finqa.program_generator import ProgramGenerator
from finqa.program_executor import ProgramExecutor
from indexing.schema import FinQAPrediction
from verification.numeric_verifier import NumericVerifier


class FinQAEvaluator:
    """Evaluate system on FinQA dataset"""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.loader = FinQALoader()
        self.retriever = FinQARetriever(config_path)
        self.generator = ProgramGenerator(config_path)
        self.executor = ProgramExecutor()
        self.verifier = NumericVerifier(config_path)
        
        self.tolerance = self.config['finqa']['numerical_tolerance']
        self.output_dir = Path(self.config['evaluation']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def evaluate_split(self, 
                      split: str = "validation",
                      num_examples: int = None) -> Dict[str, any]:
        """Evaluate on a dataset split
        
        Args:
            split: Dataset split to evaluate
            num_examples: Limit number of examples (None for all)
        
        Returns:
            Evaluation metrics
        """
        # Load examples
        examples = self.loader.load_split(split)
        
        if num_examples:
            examples = examples[:num_examples]
        
        print(f"Evaluating on {len(examples)} examples from {split}...")
        
        # Run evaluation
        predictions = []
        
        for example in tqdm(examples, desc="Evaluating"):
            prediction = self.evaluate_example(example)
            predictions.append(prediction)
        
        # Compute metrics
        metrics = self.compute_metrics(examples, predictions)
        
        # Save results
        self.save_results(split, predictions, metrics)
        
        return metrics
    
    def evaluate_example(self, example) -> FinQAPrediction:
        """Evaluate a single example"""
        # Retrieve evidence
        try:
            retrieved = self.retriever.retrieve_for_question(
                example.question,
                example.id,
                top_k=self.config['retrieval']['top_k']
            )
        except Exception as e:
            print(f"Retrieval error for {example.id}: {e}")
            retrieved = []
        
        # Extract available numbers
        available_numbers = self.generator.extract_numbers_from_evidence(retrieved)
        
        # Generate and execute program
        try:
            program_text, reasoning, success, error = self.generator.generate_with_repair(
                example.question,
                retrieved,
                available_numbers
            )
            
            if success:
                final_answer, program_obj, exec_error = self.executor.execute(program_text)
            else:
                final_answer = None
                program_obj = None
                exec_error = error
        
        except Exception as e:
            print(f"Generation error for {example.id}: {e}")
            final_answer = None
            program_obj = None
            exec_error = str(e)
            success = False
        
        # Create prediction
        prediction = FinQAPrediction(
            question_id=example.id,
            question=example.question,
            predicted_answer=final_answer if final_answer is not None else -999.0,
            gold_answer=example.answer,
            program=program_obj if program_obj else self.executor.Program(steps=[], variables={}),
            retrieved_evidence=retrieved,
            execution_success=success and final_answer is not None,
            error_message=exec_error,
            intermediate_values=program_obj.variables if program_obj else {}
        )
        
        return prediction
    
    def compute_metrics(self, 
                       examples: List,
                       predictions: List[FinQAPrediction]) -> Dict[str, float]:
        """Compute evaluation metrics"""
        exact_matches = 0
        numerical_correct = 0
        execution_success = 0
        evidence_precision_scores = []
        hallucination_count = 0
        total_numbers = 0
        
        for example, pred in zip(examples, predictions):
            # Parse gold answer
            try:
                gold_value = float(example.answer.replace(',', '').replace('%', '').strip())
            except:
                # Try to extract number from answer string
                import re
                numbers = re.findall(r'-?\d+\.?\d*', example.answer)
                if numbers:
                    gold_value = float(numbers[0])
                else:
                    continue
            
            predicted_value = pred.predicted_answer
            
            # Exact match
            if abs(predicted_value - gold_value) < 1e-6:
                exact_matches += 1
            
            # Numerical accuracy with tolerance
            if abs(predicted_value - gold_value) < self.tolerance:
                numerical_correct += 1
            
            # Execution success
            if pred.execution_success:
                execution_success += 1
            
            # Evidence precision: check if evidence contained required numbers
            if pred.retrieved_evidence:
                evidence_nums = set()
                for result in pred.retrieved_evidence:
                    nums = self.verifier.extract_numbers(result.document.content)
                    evidence_nums.update(nums)
                
                # Check if gold answer is in evidence
                found_gold = False
                for num in evidence_nums:
                    if abs(num - gold_value) < self.tolerance:
                        found_gold = True
                        break
                
                evidence_precision_scores.append(1.0 if found_gold else 0.0)
            
            # Hallucination detection
            program_nums = self.verifier.extract_numbers(str(pred.program.variables))
            total_numbers += len(program_nums)
            
            available_nums = set()
            for result in pred.retrieved_evidence:
                nums = self.verifier.extract_numbers(result.document.content)
                available_nums.update(nums)
            
            for num in program_nums:
                found = False
                for avail in available_nums:
                    if abs(num - avail) < self.tolerance:
                        found = True
                        break
                if not found:
                    hallucination_count += 1
        
        n = len(predictions)
        
        metrics = {
            'exact_match': exact_matches / n if n > 0 else 0,
            'numerical_accuracy': numerical_correct / n if n > 0 else 0,
            'execution_success_rate': execution_success / n if n > 0 else 0,
            'evidence_precision': np.mean(evidence_precision_scores) if evidence_precision_scores else 0,
            'hallucination_rate': hallucination_count / total_numbers if total_numbers > 0 else 0,
            'total_examples': n,
            'exact_matches': exact_matches,
            'numerical_correct': numerical_correct,
            'execution_success': execution_success,
            'hallucination_count': hallucination_count,
            'total_numbers': total_numbers
        }
        
        return metrics
    
    def save_results(self, 
                    split: str,
                    predictions: List[FinQAPrediction],
                    metrics: Dict[str, float]):
        """Save evaluation results"""
        # Save predictions
        pred_file = self.output_dir / f"{split}_predictions.json"
        
        pred_data = []
        for pred in predictions:
            pred_data.append({
                'question_id': pred.question_id,
                'question': pred.question,
                'predicted_answer': pred.predicted_answer,
                'gold_answer': pred.gold_answer,
                'execution_success': pred.execution_success,
                'error_message': pred.error_message,
                'num_retrieved': len(pred.retrieved_evidence)
            })
        
        with open(pred_file, 'w') as f:
            json.dump(pred_data, f, indent=2)
        
        # Save metrics
        metrics_file = self.output_dir / f"{split}_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\nResults saved to {self.output_dir}")
        print(f"Predictions: {pred_file}")
        print(f"Metrics: {metrics_file}")
    
    def print_metrics(self, metrics: Dict[str, float]):
        """Print metrics in readable format"""
        print("\n" + "="*60)
        print("FINQA EVALUATION RESULTS")
        print("="*60)
        print(f"Total Examples: {metrics['total_examples']}")
        print(f"\nAccuracy Metrics:")
        print(f"  Exact Match: {metrics['exact_match']:.2%}")
        print(f"  Numerical Accuracy (±{self.tolerance}): {metrics['numerical_accuracy']:.2%}")
        print(f"\nExecution Metrics:")
        print(f"  Program Execution Success: {metrics['execution_success_rate']:.2%}")
        print(f"  Successful Executions: {metrics['execution_success']}/{metrics['total_examples']}")
        print(f"\nRetrieval Metrics:")
        print(f"  Evidence Precision: {metrics['evidence_precision']:.2%}")
        print(f"\nVerification Metrics:")
        print(f"  Hallucination Rate: {metrics['hallucination_rate']:.2%}")
        print(f"  Hallucinated Numbers: {metrics['hallucination_count']}/{metrics['total_numbers']}")
        print("="*60)


if __name__ == "__main__":
    # Run evaluation
    evaluator = FinQAEvaluator()
    
    # First, need to index the validation set
    print("Loading and indexing validation set...")
    examples = evaluator.loader.load_split("validation")[:50]  # Test with subset
    evaluator.retriever.index_examples(examples)
    
    # Run evaluation
    print("\nRunning evaluation...")
    metrics = evaluator.evaluate_split("validation", num_examples=10)  # Small test
    
    # Print results
    evaluator.print_metrics(metrics)
