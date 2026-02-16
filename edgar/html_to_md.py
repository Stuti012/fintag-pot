from bs4 import BeautifulSoup
from markdownify import markdownify as md
from pathlib import Path
from typing import Dict, List, Optional
import re


class HTMLToMarkdown:
    """Convert SEC HTML filings to Markdown while preserving tables"""
    
    def __init__(self):
        self.table_counter = 0
    
    def convert_file(self, html_filepath: str, output_filepath: Optional[str] = None) -> str:
        """Convert HTML file to Markdown
        
        Args:
            html_filepath: Path to HTML file
            output_filepath: Optional path to save markdown
        
        Returns:
            Markdown text
        """
        html_path = Path(html_filepath)
        
        with open(html_path, 'r', encoding='utf-8', errors='ignore') as f:
            html_content = f.read()
        
        markdown = self.convert(html_content)
        
        # Save if output path provided
        if output_filepath:
            output_path = Path(output_filepath)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(markdown)
        
        return markdown
    
    def convert(self, html_content: str) -> str:
        """Convert HTML content to Markdown"""
        self.table_counter = 0
        
        # Parse HTML
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for element in soup(['script', 'style', 'meta', 'link']):
            element.decompose()
        
        # Extract main document content
        # SEC filings often have the content in specific tags
        content = soup.find('document') or soup.find('body') or soup
        
        # Convert to markdown
        markdown = md(
            str(content),
            heading_style="ATX",
            bullets="-",
            strip=['script', 'style'],
            escape_asterisks=False,
            escape_underscores=False
        )
        
        # Clean up markdown
        markdown = self._clean_markdown(markdown)
        
        return markdown
    
    def _clean_markdown(self, markdown: str) -> str:
        """Clean up converted markdown"""
        # Remove excessive newlines (more than 2)
        markdown = re.sub(r'\n{3,}', '\n\n', markdown)
        
        # Remove leading/trailing whitespace from lines
        lines = [line.rstrip() for line in markdown.split('\n')]
        markdown = '\n'.join(lines)
        
        # Fix table formatting
        markdown = self._fix_table_formatting(markdown)
        
        # Remove HTML comments
        markdown = re.sub(r'<!--.*?-->', '', markdown, flags=re.DOTALL)
        
        # Clean up unicode characters
        markdown = markdown.replace('\xa0', ' ')  # Non-breaking space
        markdown = markdown.replace('\u200b', '')  # Zero-width space
        
        return markdown.strip()
    
    def _fix_table_formatting(self, markdown: str) -> str:
        """Fix markdown table formatting"""
        lines = markdown.split('\n')
        fixed_lines = []
        in_table = False
        
        for line in lines:
            # Detect table lines
            if '|' in line and line.strip().startswith('|'):
                in_table = True
                # Ensure proper spacing around pipes
                line = re.sub(r'\s*\|\s*', ' | ', line)
                line = line.strip()
            elif in_table and not line.strip():
                in_table = False
            
            fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)
    
    def extract_tables_separately(self, html_content: str) -> Dict[int, Dict[str, any]]:
        """Extract tables from HTML as separate objects
        
        Returns:
            Dictionary mapping table_id to table data
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        tables = {}
        
        for idx, table_tag in enumerate(soup.find_all('table')):
            # Extract table as markdown
            table_md = md(str(table_tag), strip=['script', 'style'])
            
            # Extract table as structured data
            headers = []
            rows = []
            
            # Get headers
            header_row = table_tag.find('tr')
            if header_row:
                headers = [th.get_text(strip=True) for th in header_row.find_all(['th', 'td'])]
            
            # Get data rows
            for row_tag in table_tag.find_all('tr')[1:]:  # Skip header
                cells = [td.get_text(strip=True) for td in row_tag.find_all(['td', 'th'])]
                if cells:  # Only add non-empty rows
                    rows.append(cells)
            
            # Get caption if available
            caption = table_tag.find('caption')
            caption_text = caption.get_text(strip=True) if caption else None
            
            tables[idx] = {
                'table_id': idx,
                'markdown': table_md,
                'headers': headers,
                'rows': rows,
                'caption': caption_text,
                'num_rows': len(rows),
                'num_cols': len(headers) if headers else (len(rows[0]) if rows else 0)
            }
        
        return tables
    
    def convert_with_table_extraction(self, 
                                     html_content: str) -> tuple[str, Dict[int, Dict]]:
        """Convert HTML and extract tables separately
        
        Returns:
            (markdown_content, extracted_tables)
        """
        markdown = self.convert(html_content)
        tables = self.extract_tables_separately(html_content)
        
        return markdown, tables
    
    def convert_batch(self, 
                     html_filepaths: List[str],
                     output_dir: Optional[str] = None) -> List[str]:
        """Convert multiple HTML files
        
        Returns:
            List of output markdown filepaths
        """
        output_paths = []
        
        for html_path in html_filepaths:
            html_path = Path(html_path)
            
            if output_dir:
                output_path = Path(output_dir) / f"{html_path.stem}.md"
            else:
                output_path = html_path.with_suffix('.md')
            
            self.convert_file(str(html_path), str(output_path))
            output_paths.append(str(output_path))
        
        return output_paths


if __name__ == "__main__":
    # Test
    converter = HTMLToMarkdown()
    
    # Test HTML
    test_html = """
    <html>
    <body>
        <h1>Financial Report</h1>
        <p>Revenue increased by 20% year over year.</p>
        <table>
            <tr>
                <th>Year</th>
                <th>Revenue</th>
                <th>Profit</th>
            </tr>
            <tr>
                <td>2022</td>
                <td>$100M</td>
                <td>$20M</td>
            </tr>
            <tr>
                <td>2023</td>
                <td>$120M</td>
                <td>$25M</td>
            </tr>
        </table>
    </body>
    </html>
    """
    
    markdown, tables = converter.convert_with_table_extraction(test_html)
    
    print("MARKDOWN:")
    print(markdown)
    print("\n" + "="*60)
    print(f"\nEXTRACTED {len(tables)} TABLES:")
    for table_id, table_data in tables.items():
        print(f"\nTable {table_id}:")
        print(f"  Headers: {table_data['headers']}")
        print(f"  Rows: {table_data['num_rows']}")
        print(f"  Markdown:\n{table_data['markdown']}")
