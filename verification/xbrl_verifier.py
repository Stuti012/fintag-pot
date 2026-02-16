from typing import Dict, List, Optional
import yaml


class XBRLVerifier:
    """Optional: Verify numbers against XBRL structured data"""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.enabled = self.config['verification'].get('enable_xbrl_verification', False)
        self.tolerance = 0.001  # 0.1% tolerance for rounding
    
    def verify_against_xbrl(self,
                           claimed_values: Dict[str, float],
                           xbrl_data: Dict[str, any]) -> Dict[str, any]:
        """Verify claimed values against XBRL data
        
        Args:
            claimed_values: Dict mapping metric names to values
            xbrl_data: XBRL data structure
        
        Returns:
            Verification results
        """
        if not self.enabled:
            return {
                'verified': True,
                'message': 'XBRL verification disabled'
            }
        
        results = {}
        discrepancies = []
        
        for metric, claimed_value in claimed_values.items():
            xbrl_value = self._find_xbrl_value(metric, xbrl_data)
            
            if xbrl_value is None:
                results[metric] = {
                    'verified': False,
                    'reason': 'Metric not found in XBRL',
                    'claimed': claimed_value
                }
                continue
            
            # Compare with tolerance
            if self._values_match(claimed_value, xbrl_value):
                results[metric] = {
                    'verified': True,
                    'claimed': claimed_value,
                    'xbrl': xbrl_value
                }
            else:
                results[metric] = {
                    'verified': False,
                    'reason': 'Value mismatch',
                    'claimed': claimed_value,
                    'xbrl': xbrl_value,
                    'diff': abs(claimed_value - xbrl_value),
                    'diff_pct': abs(claimed_value - xbrl_value) / xbrl_value * 100 if xbrl_value != 0 else float('inf')
                }
                discrepancies.append(metric)
        
        all_verified = len(discrepancies) == 0
        
        return {
            'verified': all_verified,
            'results': results,
            'discrepancies': discrepancies,
            'message': 'All values verified' if all_verified else f'{len(discrepancies)} discrepancies found'
        }
    
    def _find_xbrl_value(self, metric: str, xbrl_data: Dict) -> Optional[float]:
        """Find value for metric in XBRL data
        
        This is a simplified implementation. Real XBRL parsing is more complex.
        """
        # Common XBRL tag mappings
        tag_mappings = {
            'revenue': ['Revenues', 'NetRevenue', 'SalesRevenueNet'],
            'net_income': ['NetIncomeLoss', 'ProfitLoss'],
            'total_assets': ['Assets', 'AssetsCurrent'],
            'total_liabilities': ['Liabilities', 'LiabilitiesCurrent'],
            'equity': ['StockholdersEquity', 'ShareholdersEquity'],
            'cash': ['Cash', 'CashAndCashEquivalents'],
            'operating_income': ['OperatingIncomeLoss']
        }
        
        # Normalize metric name
        metric_normalized = metric.lower().replace(' ', '_')
        
        # Look for metric in XBRL data
        if metric_normalized in tag_mappings:
            for tag in tag_mappings[metric_normalized]:
                if tag in xbrl_data:
                    return self._extract_numeric_value(xbrl_data[tag])
        
        # Direct lookup
        if metric in xbrl_data:
            return self._extract_numeric_value(xbrl_data[metric])
        
        return None
    
    def _extract_numeric_value(self, xbrl_entry: any) -> Optional[float]:
        """Extract numeric value from XBRL entry"""
        if isinstance(xbrl_entry, (int, float)):
            return float(xbrl_entry)
        
        if isinstance(xbrl_entry, dict):
            # XBRL entries are often dicts with 'value' key
            if 'value' in xbrl_entry:
                return float(xbrl_entry['value'])
            if 'amount' in xbrl_entry:
                return float(xbrl_entry['amount'])
        
        return None
    
    def _values_match(self, value1: float, value2: float) -> bool:
        """Check if two values match within tolerance"""
        if value1 == value2:
            return True
        
        # Absolute difference
        abs_diff = abs(value1 - value2)
        if abs_diff < 0.01:  # Within 1 cent
            return True
        
        # Relative difference
        if value2 != 0:
            rel_diff = abs_diff / abs(value2)
            if rel_diff < self.tolerance:
                return True
        
        return False
    
    def parse_xbrl_file(self, xbrl_filepath: str) -> Dict:
        """Parse XBRL file
        
        Note: This is a placeholder. Real XBRL parsing requires libraries like:
        - arelle
        - python-xbrl
        """
        # Placeholder implementation
        print(f"XBRL parsing not fully implemented. Would parse: {xbrl_filepath}")
        
        # Return mock structure
        return {
            'Revenues': 100000000,
            'NetIncomeLoss': 20000000,
            'Assets': 500000000,
            'Liabilities': 300000000,
            'StockholdersEquity': 200000000
        }
    
    def extract_xbrl_from_filing(self, accession_number: str) -> Optional[Dict]:
        """Extract XBRL data from SEC filing
        
        XBRL files are available at:
        https://www.sec.gov/cgi-bin/viewer?action=view&cik={cik}&accession_number={accession}&xbrl_type=v
        """
        # Placeholder
        print(f"XBRL extraction not fully implemented for: {accession_number}")
        return None


if __name__ == "__main__":
    # Test
    verifier = XBRLVerifier()
    
    # Mock XBRL data
    mock_xbrl = {
        'Revenues': 100000000.0,
        'NetIncomeLoss': 20000000.0,
        'Assets': 500000000.0
    }
    
    # Test values
    claimed = {
        'revenue': 100000000.0,  # Exact match
        'net_income': 20000500.0,  # Close match (within tolerance)
        'total_assets': 600000000.0  # Mismatch
    }
    
    results = verifier.verify_against_xbrl(claimed, mock_xbrl)
    
    print("Verification results:")
    print(f"Overall verified: {results['verified']}")
    print(f"Message: {results['message']}")
    
    for metric, result in results['results'].items():
        print(f"\n{metric}:")
        print(f"  Verified: {result['verified']}")
        if not result['verified']:
            print(f"  Reason: {result.get('reason')}")
            if 'diff_pct' in result:
                print(f"  Difference: {result['diff_pct']:.2f}%")
