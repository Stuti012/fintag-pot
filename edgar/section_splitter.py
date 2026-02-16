import re
from typing import Dict, List, Tuple
from pathlib import Path


class SECSectionSplitter:
    """Split SEC filings into sections (Items)"""
    
    # Common 10-K sections
    SECTION_10K = [
        (r'ITEM\s+1[^A0-9]', 'Item 1 - Business'),
        (r'ITEM\s+1A', 'Item 1A - Risk Factors'),
        (r'ITEM\s+1B', 'Item 1B - Unresolved Staff Comments'),
        (r'ITEM\s+2[^.]', 'Item 2 - Properties'),
        (r'ITEM\s+3[^.]', 'Item 3 - Legal Proceedings'),
        (r'ITEM\s+4[^.]', 'Item 4 - Mine Safety Disclosures'),
        (r'ITEM\s+5[^.]', 'Item 5 - Market for Registrant\'s Common Equity'),
        (r'ITEM\s+6[^.]', 'Item 6 - Selected Financial Data'),
        (r'ITEM\s+7A', 'Item 7A - Quantitative and Qualitative Disclosures'),
        (r'ITEM\s+7[^A]', 'Item 7 - Management\'s Discussion and Analysis'),
        (r'ITEM\s+8[^.]', 'Item 8 - Financial Statements'),
        (r'ITEM\s+9A', 'Item 9A - Controls and Procedures'),
        (r'ITEM\s+9B', 'Item 9B - Other Information'),
        (r'ITEM\s+9[^AB]', 'Item 9 - Changes in and Disagreements'),
        (r'ITEM\s+10[^.]', 'Item 10 - Directors and Executive Officers'),
        (r'ITEM\s+11[^.]', 'Item 11 - Executive Compensation'),
        (r'ITEM\s+12[^.]', 'Item 12 - Security Ownership'),
        (r'ITEM\s+13[^.]', 'Item 13 - Certain Relationships'),
        (r'ITEM\s+14[^.]', 'Item 14 - Principal Accountant Fees'),
        (r'ITEM\s+15[^.]', 'Item 15 - Exhibits and Financial Statement Schedules'),
    ]
    
    # Common 10-Q sections
    SECTION_10Q = [
        (r'ITEM\s+1[^A]', 'Item 1 - Financial Statements'),
        (r'ITEM\s+2[^.]', 'Item 2 - Management\'s Discussion and Analysis'),
        (r'ITEM\s+3[^.]', 'Item 3 - Quantitative and Qualitative Disclosures'),
        (r'ITEM\s+4[^.]', 'Item 4 - Controls and Procedures'),
        (r'ITEM\s+1A', 'Item 1A - Risk Factors'),
        (r'ITEM\s+5[^.]', 'Item 5 - Other Information'),
        (r'ITEM\s+6[^.]', 'Item 6 - Exhibits'),
    ]
    
    def __init__(self):
        pass
    
    def split_10k(self, content: str) -> Dict[str, str]:
        """Split 10-K filing into sections"""
        return self._split_by_items(content, self.SECTION_10K)
    
    def split_10q(self, content: str) -> Dict[str, str]:
        """Split 10-Q filing into sections"""
        return self._split_by_items(content, self.SECTION_10Q)
    
    def split_filing(self, content: str, filing_type: str) -> Dict[str, str]:
        """Split filing based on type"""
        if '10-K' in filing_type.upper():
            return self.split_10k(content)
        elif '10-Q' in filing_type.upper():
            return self.split_10q(content)
        else:
            # Return entire content as single section
            return {'full_document': content}
    
    def _split_by_items(self, 
                       content: str, 
                       item_patterns: List[Tuple[str, str]]) -> Dict[str, str]:
        """Split content by item patterns"""
        sections = {}
        
        # Find all item positions
        item_positions = []
        
        for pattern, name in item_patterns:
            matches = list(re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE))
            for match in matches:
                item_positions.append({
                    'start': match.start(),
                    'name': name,
                    'pattern': pattern
                })
        
        # Sort by position
        item_positions.sort(key=lambda x: x['start'])
        
        # Extract sections
        for i, item in enumerate(item_positions):
            start = item['start']
            
            # End is start of next item, or end of document
            if i + 1 < len(item_positions):
                end = item_positions[i + 1]['start']
            else:
                end = len(content)
            
            section_content = content[start:end].strip()
            
            # Only add if section has substantial content
            if len(section_content) > 100:
                sections[item['name']] = section_content
        
        # If no sections found, return full content
        if not sections:
            sections['full_document'] = content
        
        return sections
    
    def split_into_chunks(self,
                         sections: Dict[str, str],
                         chunk_size: int = 512,
                         chunk_overlap: int = 50) -> List[Dict[str, str]]:
        """Split sections into smaller chunks for indexing
        
        Returns:
            List of chunks with metadata
        """
        chunks = []
        
        for section_name, section_content in sections.items():
            # Split by paragraphs first
            paragraphs = section_content.split('\n\n')
            
            current_chunk = []
            current_length = 0
            chunk_idx = 0
            
            for para in paragraphs:
                para_words = para.split()
                para_length = len(para_words)
                
                if current_length + para_length > chunk_size and current_chunk:
                    # Save current chunk
                    chunk_text = ' '.join(current_chunk)
                    chunks.append({
                        'section': section_name,
                        'chunk_id': chunk_idx,
                        'content': chunk_text,
                        'word_count': current_length
                    })
                    
                    # Start new chunk with overlap
                    if chunk_overlap > 0:
                        overlap_words = current_chunk[-chunk_overlap:]
                        current_chunk = overlap_words + para_words
                        current_length = len(overlap_words) + para_length
                    else:
                        current_chunk = para_words
                        current_length = para_length
                    
                    chunk_idx += 1
                else:
                    current_chunk.extend(para_words)
                    current_length += para_length
            
            # Add remaining chunk
            if current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append({
                    'section': section_name,
                    'chunk_id': chunk_idx,
                    'content': chunk_text,
                    'word_count': current_length
                })
        
        return chunks
    
    def get_important_sections(self, sections: Dict[str, str]) -> Dict[str, str]:
        """Filter to most important sections for financial analysis"""
        important_items = [
            'Item 1 - Business',
            'Item 1A - Risk Factors',
            'Item 7 - Management\'s Discussion and Analysis',
            'Item 8 - Financial Statements',
            'Item 2 - Management\'s Discussion and Analysis',  # 10-Q
            'Item 1 - Financial Statements',  # 10-Q
        ]
        
        filtered = {}
        for section_name, content in sections.items():
            for important_item in important_items:
                if important_item in section_name:
                    filtered[section_name] = content
                    break
        
        return filtered if filtered else sections


if __name__ == "__main__":
    # Test
    splitter = SECSectionSplitter()
    
    # Test content
    test_content = """
    ITEM 1. BUSINESS
    
    Apple Inc. designs, manufactures and markets smartphones...
    
    ITEM 1A. RISK FACTORS
    
    We face intense competition in all aspects of our business...
    
    ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS
    
    The following discussion should be read in conjunction with...
    
    ITEM 8. FINANCIAL STATEMENTS
    
    Consolidated Statements of Operations...
    """
    
    sections = splitter.split_10k(test_content)
    
    print(f"Found {len(sections)} sections:")
    for section_name, content in sections.items():
        print(f"\n{section_name}")
        print(f"  Length: {len(content)} characters")
        print(f"  Preview: {content[:100]}...")
    
    # Test chunking
    chunks = splitter.split_into_chunks(sections, chunk_size=50, chunk_overlap=10)
    print(f"\nCreated {len(chunks)} chunks")
