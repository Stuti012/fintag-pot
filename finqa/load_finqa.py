import json
import os
from typing import List, Dict, Any
from pathlib import Path
from indexing.schema import FinQAExample


class FinQALoader:
    def __init__(self, data_path: str = "./data/finqa"):
        self.data_path = Path(data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)
    
    def load_split(self, split: str = "validation") -> List[FinQAExample]:
        """Load FinQA dataset split
        
        Args:
            split: one of 'train', 'validation', 'test'
        
        Returns:
            List of FinQAExample objects
        """
        split_file = self.data_path / f"{split}.json"
        
        if not split_file.exists():
            raise FileNotFoundError(
                f"FinQA {split} file not found at {split_file}. "
                f"Please download from https://github.com/czyssrs/FinQA"
            )
        
        with open(split_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        examples = []
        for item in data:
            example = FinQAExample(
                id=item.get('id', item.get('uid', str(len(examples)))),
                question=item['qa']['question'],
                pre_text=item.get('pre_text', []),
                post_text=item.get('post_text', []),
                table=item.get('table', []),
                answer=item['qa']['answer'],
                program=item['qa'].get('program', None),
                gold_inds=item.get('gold_inds', None)
            )
            examples.append(example)
        
        return examples
    
    def load_all_splits(self) -> Dict[str, List[FinQAExample]]:
        """Load all available splits"""
        splits = {}
        for split in ['train', 'validation', 'test']:
            try:
                splits[split] = self.load_split(split)
                print(f"Loaded {len(splits[split])} examples from {split}")
            except FileNotFoundError:
                print(f"Split {split} not found, skipping")
        return splits
    
    def get_example_by_id(self, example_id: str, split: str = "validation") -> FinQAExample:
        """Get specific example by ID"""
        examples = self.load_split(split)
        for ex in examples:
            if ex.id == example_id:
                return ex
        raise ValueError(f"Example {example_id} not found in {split}")
    
    def convert_to_documents(self, example: FinQAExample) -> List[Dict[str, Any]]:
        """Convert FinQA example to indexable documents"""
        documents = []
        
        # Pre-text as separate documents
        for i, text in enumerate(example.pre_text):
            documents.append({
                'doc_id': f"{example.id}_pre_{i}",
                'chunk_id': f"chunk_0",
                'content': text,
                'doc_type': 'text',
                'source_type': 'finqa',
                'metadata': {
                    'question_id': example.id,
                    'section': 'pre_text',
                    'index': i
                }
            })
        
        # Post-text as separate documents
        for i, text in enumerate(example.post_text):
            documents.append({
                'doc_id': f"{example.id}_post_{i}",
                'chunk_id': f"chunk_0",
                'content': text,
                'doc_type': 'text',
                'source_type': 'finqa',
                'metadata': {
                    'question_id': example.id,
                    'section': 'post_text',
                    'index': i
                }
            })
        
        # Table as raw markdown
        if example.table:
            table_md = self._table_to_markdown(example.table)
            documents.append({
                'doc_id': f"{example.id}_table",
                'chunk_id': f"chunk_0",
                'content': table_md,
                'doc_type': 'table',
                'source_type': 'finqa',
                'metadata': {
                    'question_id': example.id,
                    'section': 'table',
                    'table_data': example.table
                }
            })
        
        return documents
    
    @staticmethod
    def _table_to_markdown(table: List[List[Any]]) -> str:
        """Convert table to markdown format"""
        if not table:
            return ""
        
        lines = []
        for row in table:
            row_str = " | ".join(str(cell) for cell in row)
            lines.append(f"| {row_str} |")
            
            # Add separator after header
            if len(lines) == 1:
                sep = " | ".join(["---"] * len(row))
                lines.append(f"| {sep} |")
        
        return "\n".join(lines)
    
    def extract_numbers_from_example(self, example: FinQAExample) -> List[float]:
        """Extract all numbers from an example for verification"""
        import re
        numbers = []
        
        # Extract from text
        for text in example.pre_text + example.post_text:
            found = re.findall(r'-?\d+\.?\d*', text)
            numbers.extend([float(n) for n in found])
        
        # Extract from table
        if example.table:
            for row in example.table:
                for cell in row:
                    try:
                        numbers.append(float(cell))
                    except (ValueError, TypeError):
                        # Try to extract numbers from strings
                        if isinstance(cell, str):
                            found = re.findall(r'-?\d+\.?\d*', cell.replace(',', ''))
                            numbers.extend([float(n) for n in found])
        
        return numbers


if __name__ == "__main__":
    loader = FinQALoader()
    examples = loader.load_split("validation")
    print(f"Loaded {len(examples)} examples")
    
    if examples:
        ex = examples[0]
        print(f"\nExample: {ex.id}")
        print(f"Question: {ex.question}")
        print(f"Answer: {ex.answer}")
        print(f"Pre-text items: {len(ex.pre_text)}")
        print(f"Post-text items: {len(ex.post_text)}")
        print(f"Table rows: {len(ex.table) if ex.table else 0}")
