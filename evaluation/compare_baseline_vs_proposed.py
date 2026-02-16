import yaml
import json
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np

from finqa.finqa_eval import FinQAEvaluator


class BaselineVsProposedEvaluator:
    """Compare baseline system vs proposed system"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.output_dir = Path(self.config['evaluation']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def run_comparison(self, 
                      split: str = "validation",
                      num_examples: int = 100) -> Dict[str, Dict]:
        """Run both baseline and proposed systems
        
        Returns:
            Dictionary with results from both systems
        """
        results = {}
        
        # Run baseline
        print("\n" + "="*60)
        print("RUNNING BASELINE SYSTEM")
        print("="*60)
        baseline_metrics = self.run_baseline(split, num_examples)
        results['baseline'] = baseline_metrics
        
        # Run proposed
        print("\n" + "="*60)
        print("RUNNING PROPOSED SYSTEM")
        print("="*60)
        proposed_metrics = self.run_proposed(split, num_examples)
        results['proposed'] = proposed_metrics
        
        # Generate comparison
        self.generate_comparison_report(results)
        
        return results
    
    def run_baseline(self, split: str, num_examples: int) -> Dict:
        """Run baseline system
        
        Baseline features:
        - Text chunking only (no table-aware indexing)
        - Vector search only (no BM25)
        - Direct answer generation (no PoT)
        """
        # Modify config for baseline
        baseline_config = self.config.copy()
        baseline_config['retrieval']['enable_hybrid'] = False
        baseline_config['retrieval']['bm25_weight'] = 0.0
        baseline_config['retrieval']['vector_weight'] = 1.0
        baseline_config['indexing']['table_summary_enabled'] = False
        baseline_config['evaluation']['baseline_mode'] = True
        
        # Save temporary config
        baseline_config_path = "config_baseline_temp.yaml"
        with open(baseline_config_path, 'w') as f:
            yaml.dump(baseline_config, f)
        
        # Run evaluation
        evaluator = FinQAEvaluator(baseline_config_path)
        
        # Load and index (without table-aware features)
        examples = evaluator.loader.load_split(split)[:num_examples]
        
        # Index without table summaries
        evaluator.retriever.index_examples(examples)
        
        # Evaluate
        metrics = evaluator.evaluate_split(split, num_examples)
        
        # Rename output files
        for file in self.output_dir.glob(f"{split}_*"):
            new_name = file.parent / f"baseline_{file.name}"
            file.rename(new_name)
        
        # Cleanup
        Path(baseline_config_path).unlink()
        
        return metrics
    
    def run_proposed(self, split: str, num_examples: int) -> Dict:
        """Run proposed system
        
        Proposed features:
        - Table-aware hierarchical indexing
        - Hybrid retrieval (BM25 + vector)
        - Program-of-Thought reasoning
        - Temporal decay (for EDGAR)
        """
        # Use default config (proposed system)
        evaluator = FinQAEvaluator(self.config_path)
        
        # Load and index (with all features)
        examples = evaluator.loader.load_split(split)[:num_examples]
        evaluator.retriever.index_examples(examples)
        
        # Evaluate
        metrics = evaluator.evaluate_split(split, num_examples)
        
        # Rename output files
        for file in self.output_dir.glob(f"{split}_*"):
            if not file.name.startswith("baseline_"):
                new_name = file.parent / f"proposed_{file.name}"
                file.rename(new_name)
        
        return metrics
    
    def generate_comparison_report(self, results: Dict[str, Dict]):
        """Generate comparison report and visualizations"""
        baseline = results['baseline']
        proposed = results['proposed']
        
        # Print comparison
        print("\n" + "="*80)
        print("BASELINE VS PROPOSED COMPARISON")
        print("="*80)
        
        metrics_to_compare = [
            ('exact_match', 'Exact Match'),
            ('numerical_accuracy', 'Numerical Accuracy'),
            ('execution_success_rate', 'Execution Success Rate'),
            ('evidence_precision', 'Evidence Precision'),
            ('hallucination_rate', 'Hallucination Rate (lower is better)')
        ]
        
        print(f"\n{'Metric':<40} {'Baseline':<15} {'Proposed':<15} {'Δ':<15}")
        print("-" * 80)
        
        for metric_key, metric_name in metrics_to_compare:
            base_val = baseline[metric_key]
            prop_val = proposed[metric_key]
            
            # For hallucination rate, improvement is negative
            if 'hallucination' in metric_key:
                improvement = base_val - prop_val
                improvement_pct = (improvement / base_val * 100) if base_val > 0 else 0
            else:
                improvement = prop_val - base_val
                improvement_pct = (improvement / base_val * 100) if base_val > 0 else 0
            
            print(f"{metric_name:<40} {base_val:<15.2%} {prop_val:<15.2%} "
                  f"{improvement_pct:+.1f}%")
        
        print("="*80)
        
        # Save comparison
        comparison = {
            'baseline': baseline,
            'proposed': proposed,
            'improvements': {}
        }
        
        for metric_key, _ in metrics_to_compare:
            base_val = baseline[metric_key]
            prop_val = proposed[metric_key]
            
            if 'hallucination' in metric_key:
                improvement = base_val - prop_val
                improvement_pct = (improvement / base_val * 100) if base_val > 0 else 0
            else:
                improvement = prop_val - base_val
                improvement_pct = (improvement / base_val * 100) if base_val > 0 else 0
            
            comparison['improvements'][metric_key] = {
                'absolute': improvement,
                'percentage': improvement_pct
            }
        
        comparison_file = self.output_dir / "baseline_vs_proposed.json"
        with open(comparison_file, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        print(f"\nComparison saved to: {comparison_file}")
        
        # Generate plots
        self.plot_comparison(baseline, proposed, metrics_to_compare)
    
    def plot_comparison(self, 
                       baseline: Dict, 
                       proposed: Dict,
                       metrics: List):
        """Generate comparison plots"""
        try:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            # Plot 1: Bar chart of metrics
            metric_names = [name for _, name in metrics if 'hallucination' not in name]
            metric_keys = [key for key, name in metrics if 'hallucination' not in name]
            
            baseline_vals = [baseline[key] for key in metric_keys]
            proposed_vals = [proposed[key] for key in metric_keys]
            
            x = np.arange(len(metric_names))
            width = 0.35
            
            axes[0].bar(x - width/2, baseline_vals, width, label='Baseline', alpha=0.8)
            axes[0].bar(x + width/2, proposed_vals, width, label='Proposed', alpha=0.8)
            
            axes[0].set_xlabel('Metrics')
            axes[0].set_ylabel('Score')
            axes[0].set_title('Baseline vs Proposed System Performance')
            axes[0].set_xticks(x)
            axes[0].set_xticklabels([name.replace(' ', '\n') for name in metric_names], 
                                   rotation=0, ha='center', fontsize=9)
            axes[0].legend()
            axes[0].grid(axis='y', alpha=0.3)
            
            # Plot 2: Improvement percentages
            improvements = []
            for key in metric_keys:
                base_val = baseline[key]
                prop_val = proposed[key]
                improvement_pct = ((prop_val - base_val) / base_val * 100) if base_val > 0 else 0
                improvements.append(improvement_pct)
            
            colors = ['green' if imp > 0 else 'red' for imp in improvements]
            axes[1].barh(metric_names, improvements, color=colors, alpha=0.7)
            axes[1].axvline(x=0, color='black', linestyle='-', linewidth=0.8)
            axes[1].set_xlabel('Improvement (%)')
            axes[1].set_title('Relative Improvement of Proposed System')
            axes[1].grid(axis='x', alpha=0.3)
            
            plt.tight_layout()
            
            plot_file = self.output_dir / "baseline_vs_proposed.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"Plots saved to: {plot_file}")
            
            plt.close()
        
        except Exception as e:
            print(f"Could not generate plots: {e}")


if __name__ == "__main__":
    evaluator = BaselineVsProposedEvaluator()
    
    # Run comparison on small subset for testing
    results = evaluator.run_comparison(
        split="validation",
        num_examples=20  # Use small number for testing
    )
    
    print("\nComparison complete!")
