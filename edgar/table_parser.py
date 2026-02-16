from typing import List, Dict, Optional, Tuple
import re
from bs4 import BeautifulSoup


class TableParser:
    """Parse and extract financial tables from SEC filings"""
    
    def __init__(self):
        self.table_counter = 0
    
    def parse_html_tables(self, html_content: str) -> List[Dict]:
        """Extract tables from HTML content"""
        soup = BeautifulSoup(html_content, 'html.parser')
        tables = []
        
        for idx, table_tag in enumerate(soup.find_all('table')):
            table_data = self._parse_single_table(table_tag, idx)
            if table_data and table_data['rows']:  # Only add non-empty tables
                tables.append(table_data)
        
        return tables
    
    def parse_markdown_tables(self, markdown_content: str) -> List[Dict]:
        """Extract tables from Markdown content"""
        tables = []
        
        # Find all markdown tables
        table_pattern = r'\|.+\|[\r\n]+\|[-\s|]+\|[\r\n]+((?:\|.+\|[\r\n]*)+)'
        matches = re.finditer(table_pattern, markdown_content, re.MULTILINE)
        
        for idx, match in enumerate(matches):
            table_text = match.group(0)
            table_data = self._parse_markdown_table(table_text, idx)
            if table_data and table_data['rows']:
                tables.append(table_data)
        
        return tables
    
    def _parse_single_table(self, table_tag, table_id: int) -> Dict:
        """Parse a single HTML table"""
        # Get caption
        caption = table_tag.find('caption')
        caption_text = caption.get_text(strip=True) if caption else None
        
        # Extract rows
        rows_data = []
        headers = []
        
        for row_idx, row_tag in enumerate(table_tag.find_all('tr')):
            cells = []
            
            # Check if this is a header row
            header_cells = row_tag.find_all('th')
            if header_cells and row_idx == 0:
                headers = [self._clean_cell_text(th.get_text()) for th in header_cells]
                continue
            
            # Parse data cells
            data_cells = row_tag.find_all(['td', 'th'])
            for cell in data_cells:
                cell_text = self._clean_cell_text(cell.get_text())
                cells.append(cell_text)
            
            if cells:
                rows_data.append(cells)
        
        # If no explicit headers found, use first row
        if not headers and rows_data:
            headers = rows_data[0]
            rows_data = rows_data[1:]
        
        return {
            'table_id': table_id,
            'caption': caption_text,
            'headers': headers,
            'rows': rows_data,
            'num_rows': len(rows_data),
            'num_cols': len(headers) if headers else (len(rows_data[0]) if rows_data else 0),
            'markdown': self._to_markdown(headers, rows_data),
            'summary': self._generate_summary(caption_text, headers, rows_data)
        }
    
    def _parse_markdown_table(self, table_text: str, table_id: int) -> Dict:
        """Parse a markdown table"""
        lines = [line.strip() for line in table_text.split('\n') if line.strip()]
        
        if len(lines) < 2:
            return None
        
        # Extract headers
        headers = [cell.strip() for cell in lines[0].split('|')[1:-1]]
        
        # Skip separator line
        # Extract data rows
        rows_data = []
        for line in lines[2:]:
            cells = [cell.strip() for cell in line.split('|')[1:-1]]
            if cells and any(cells):  # Not empty row
                rows_data.append(cells)
        
        return {
            'table_id': table_id,
            'caption': None,
            'headers': headers,
            'rows': rows_data,
            'num_rows': len(rows_data),
            'num_cols': len(headers),
            'markdown': table_text,
            'summary': self._generate_summary(None, headers, rows_data)
        }
    
    def _clean_cell_text(self, text: str) -> str:
        """Clean cell text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Remove certain unicode characters
        text = text.replace('\xa0', ' ')
        text = text.replace('\u200b', '')
        
        return text
    
    def _to_markdown(self, headers: List[str], rows: List[List[str]]) -> str:
        """Convert table to markdown format"""
        if not headers and not rows:
            return ""
        
        lines = []
        
        # Headers
        if headers:
            header_line = "| " + " | ".join(headers) + " |"
            lines.append(header_line)
            
            # Separator
            separator = "| " + " | ".join(["---"] * len(headers)) + " |"
            lines.append(separator)
        
        # Rows
        for row in rows:
            # Pad row if needed
            while len(row) < len(headers):
                row.append("")
            
            row_line = "| " + " | ".join(row[:len(headers)]) + " |"
            lines.append(row_line)
        
        return "\n".join(lines)
    
    def _generate_summary(self, 
                         caption: Optional[str],
                         headers: List[str],
                         rows: List[List[str]]) -> str:
        """Generate a text summary of the table"""
        summary_parts = []
        
        if caption:
            summary_parts.append(f"Table: {caption}")
        
        if headers:
            summary_parts.append(f"Columns: {', '.join(headers)}")
        
        if rows:
            summary_parts.append(f"{len(rows)} rows of data")
            
            # Try to extract numeric values
            numeric_cols = self._identify_numeric_columns(headers, rows)
            if numeric_cols:
                summary_parts.append(f"Numeric columns: {', '.join(numeric_cols)}")
        
        return ". ".join(summary_parts) if summary_parts else "Financial table"
    
    def _identify_numeric_columns(self, headers: List[str], rows: List[List[str]]) -> List[str]:
        """Identify columns that contain numeric data"""
        numeric_cols = []
        
        if not headers or not rows:
            return numeric_cols
        
        for col_idx, header in enumerate(headers):
            # Check if column contains mostly numbers
            numeric_count = 0
            total_count = 0
            
            for row in rows:
                if col_idx < len(row):
                    cell = row[col_idx]
                    total_count += 1
                    
                    # Check if cell contains number
                    if self._contains_number(cell):
                        numeric_count += 1
            
            # If more than 50% numeric, consider it a numeric column
            if total_count > 0 and numeric_count / total_count > 0.5:
                numeric_cols.append(header)
        
        return numeric_cols
    
    def _contains_number(self, text: str) -> bool:
        """Check if text contains a number"""
        # Remove common non-numeric characters
        cleaned = re.sub(r'[$,\s%()]', '', text)
        
        try:
            float(cleaned)
            return True
        except ValueError:
            return False
    
    def extract_financial_data(self, table_data: Dict) -> Dict[str, List[float]]:
        """Extract financial metrics from table
        
        Returns:
            Dictionary mapping metric names to values
        """
        financial_data = {}
        
        headers = table_data.get('headers', [])
        rows = table_data.get('rows', [])
        
        if not headers or not rows:
            return financial_data
        
        # Identify the label column (usually first)
        label_col_idx = 0
        
        # Identify numeric columns
        for col_idx in range(1, len(headers)):
            col_name = headers[col_idx]
            values = []
            
            for row in rows:
                if col_idx < len(row):
                    cell = row[col_idx]
                    # Try to extract number
                    number = self._extract_number(cell)
                    if number is not None:
                        values.append(number)
            
            if values:
                financial_data[col_name] = values
        
        return financial_data
    
    def _extract_number(self, text: str) -> Optional[float]:
        """Extract numeric value from text"""
        # Remove common formatting
        cleaned = text.replace('$', '').replace(',', '').replace('(', '-').replace(')', '')
        cleaned = cleaned.replace('%', '').strip()
        
        try:
            return float(cleaned)
        except ValueError:
            # Try to find any number in the text
            match = re.search(r'-?\d+\.?\d*', cleaned)
            if match:
                try:
                    return float(match.group(0))
                except ValueError:
                    pass
        
        return None
    
    def filter_financial_tables(self, tables: List[Dict]) -> List[Dict]:
        """Filter to keep only tables with financial data"""
        financial_keywords = [
            'revenue', 'income', 'expense', 'assets', 'liabilities',
            'equity', 'cash', 'earnings', 'profit', 'loss',
            'balance', 'statement', 'financial', 'fiscal'
        ]
        
        filtered_tables = []
        
        for table in tables:
            # Check caption
            caption = table.get('caption', '') or ''
            
            # Check headers
            headers = ' '.join(table.get('headers', [])).lower()
            
            # Check first few cells
            first_cells = []
            for row in table.get('rows', [])[:3]:
                first_cells.extend(row)
            first_cells_text = ' '.join(first_cells).lower()
            
            # Combine all text
            combined_text = f"{caption} {headers} {first_cells_text}".lower()
            
            # Check for financial keywords
            if any(keyword in combined_text for keyword in financial_keywords):
                filtered_tables.append(table)
        
        return filtered_tables


if __name__ == "__main__":
    # Test
    parser = TableParser()
    
    test_html = """
    <table>
        <caption>Consolidated Balance Sheet</caption>
        <tr>
            <th>Item</th>
            <th>2023</th>
            <th>2022</th>
        </tr>
        <tr>
            <td>Revenue</td>
            <td>$394,328</td>
            <td>$365,817</td>
        </tr>
        <tr>
            <td>Net Income</td>
            <td>$99,803</td>
            <td>$94,680</td>
        </tr>
    </table>
    """
    
    tables = parser.parse_html_tables(test_html)
    
    print(f"Parsed {len(tables)} tables:")
    for table in tables:
        print(f"\nTable {table['table_id']}")
        print(f"  Caption: {table['caption']}")
        print(f"  Headers: {table['headers']}")
        print(f"  Rows: {table['num_rows']}")
        print(f"  Summary: {table['summary']}")
        print(f"  Markdown:\n{table['markdown']}")
        
        # Extract financial data
        fin_data = parser.extract_financial_data(table)
        print(f"  Financial data: {fin_data}")
