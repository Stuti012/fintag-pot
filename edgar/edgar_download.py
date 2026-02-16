import os
import requests
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import time
from tqdm import tqdm
import yaml
from dotenv import load_dotenv

load_dotenv()


class EDGARDownloader:
    """Download SEC filings using SEC EDGAR API"""
    
    BASE_URL = "https://www.sec.gov"
    SEARCH_URL = f"{BASE_URL}/cgi-bin/browse-edgar"
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_path = Path(self.config['edgar']['data_path'])
        self.data_path.mkdir(parents=True, exist_ok=True)
        
        # User agent is required by SEC
        self.user_agent = os.getenv('EDGAR_USER_AGENT', self.config['edgar']['user_agent'])
        self.headers = {'User-Agent': self.user_agent}
        
        self.filing_types = self.config['edgar']['filing_types']
        self.max_filings_per_ticker = self.config['edgar']['max_filings_per_ticker']
        
        # Rate limiting
        self.request_delay = 0.1  # SEC allows 10 requests per second
    
    def download_filing(self, 
                       ticker: str,
                       filing_type: str,
                       accession_number: str,
                       filing_date: str) -> Optional[Dict]:
        """Download a single filing"""
        # Build filing URL
        cik = self._get_cik_from_ticker(ticker)
        if not cik:
            print(f"Could not find CIK for ticker: {ticker}")
            return None
        
        # Format accession number for URL
        acc_no_clean = accession_number.replace('-', '')
        
        # Filing URL
        filing_url = f"{self.BASE_URL}/Archives/edgar/data/{cik}/{acc_no_clean}/{accession_number}.txt"
        
        # Try HTML version first
        html_url = f"{self.BASE_URL}/Archives/edgar/data/{cik}/{acc_no_clean}/{accession_number}-index.html"
        
        try:
            time.sleep(self.request_delay)
            response = requests.get(html_url, headers=self.headers, timeout=30)
            
            if response.status_code != 200:
                # Try text version
                response = requests.get(filing_url, headers=self.headers, timeout=30)
            
            if response.status_code == 200:
                # Save filing
                filing_dir = self.data_path / ticker / filing_type
                filing_dir.mkdir(parents=True, exist_ok=True)
                
                filename = f"{accession_number}_{filing_date}.html"
                filepath = filing_dir / filename
                
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                
                return {
                    'ticker': ticker,
                    'cik': cik,
                    'filing_type': filing_type,
                    'accession_number': accession_number,
                    'filing_date': filing_date,
                    'filepath': str(filepath),
                    'url': html_url if '/html' in html_url else filing_url
                }
            else:
                print(f"Failed to download {accession_number}: Status {response.status_code}")
                return None
                
        except Exception as e:
            print(f"Error downloading {accession_number}: {e}")
            return None
    
    def download_recent_filings(self,
                               ticker: str,
                               filing_types: Optional[List[str]] = None,
                               max_filings: Optional[int] = None) -> List[Dict]:
        """Download recent filings for a ticker"""
        if filing_types is None:
            filing_types = self.filing_types
        
        if max_filings is None:
            max_filings = self.max_filings_per_ticker
        
        downloaded_filings = []
        
        for filing_type in filing_types:
            print(f"Downloading {filing_type} filings for {ticker}...")
            
            filings = self._get_filing_list(ticker, filing_type, max_filings)
            
            for filing in tqdm(filings[:max_filings], desc=f"{ticker} {filing_type}"):
                result = self.download_filing(
                    ticker=ticker,
                    filing_type=filing_type,
                    accession_number=filing['accession_number'],
                    filing_date=filing['filing_date']
                )
                
                if result:
                    downloaded_filings.append(result)
        
        return downloaded_filings
    
    def download_multiple_tickers(self, 
                                 tickers: List[str],
                                 filing_types: Optional[List[str]] = None) -> Dict[str, List[Dict]]:
        """Download filings for multiple tickers"""
        results = {}
        
        for ticker in tickers:
            print(f"\n{'='*60}")
            print(f"Processing {ticker}")
            print(f"{'='*60}")
            
            filings = self.download_recent_filings(ticker, filing_types)
            results[ticker] = filings
            
            print(f"Downloaded {len(filings)} filings for {ticker}")
        
        return results
    
    def _get_filing_list(self, ticker: str, filing_type: str, count: int = 10) -> List[Dict]:
        """Get list of available filings"""
        cik = self._get_cik_from_ticker(ticker)
        if not cik:
            return []
        
        # Use SEC's RSS feed or company search
        url = f"{self.SEARCH_URL}?action=getcompany&CIK={cik}&type={filing_type}&count={count}&output=atom"
        
        try:
            time.sleep(self.request_delay)
            response = requests.get(url, headers=self.headers, timeout=30)
            
            if response.status_code == 200:
                # Parse RSS/Atom feed
                return self._parse_filing_list(response.text)
            else:
                print(f"Failed to get filing list: Status {response.status_code}")
                return []
                
        except Exception as e:
            print(f"Error getting filing list: {e}")
            return []
    
    def _parse_filing_list(self, xml_content: str) -> List[Dict]:
        """Parse filing list from XML/Atom feed"""
        from xml.etree import ElementTree as ET
        
        filings = []
        
        try:
            root = ET.fromstring(xml_content)
            
            # Find all entry elements
            for entry in root.findall('.//{http://www.w3.org/2005/Atom}entry'):
                # Extract accession number from id or link
                accession = None
                link = entry.find('.//{http://www.w3.org/2005/Atom}link')
                if link is not None:
                    href = link.get('href', '')
                    # Extract accession number from URL
                    parts = href.split('/')
                    for part in parts:
                        if '-' in part and len(part) > 10:
                            accession = part
                            break
                
                # Extract filing date
                updated = entry.find('.//{http://www.w3.org/2005/Atom}updated')
                filing_date = updated.text.split('T')[0] if updated is not None else None
                
                if accession and filing_date:
                    filings.append({
                        'accession_number': accession,
                        'filing_date': filing_date
                    })
        
        except Exception as e:
            print(f"Error parsing XML: {e}")
        
        return filings
    
    def _get_cik_from_ticker(self, ticker: str) -> Optional[str]:
        """Get CIK number from ticker symbol"""
        # SEC maintains a ticker to CIK mapping
        ticker_url = f"{self.BASE_URL}/cgi-bin/browse-edgar?action=getcompany&company={ticker}&count=1"
        
        try:
            time.sleep(self.request_delay)
            response = requests.get(ticker_url, headers=self.headers, timeout=30)
            
            if response.status_code == 200:
                # Extract CIK from response
                import re
                match = re.search(r'CIK=(\d+)', response.text)
                if match:
                    cik = match.group(1).lstrip('0')  # Remove leading zeros
                    return cik
        
        except Exception as e:
            print(f"Error getting CIK: {e}")
        
        return None
    
    def get_local_filings(self, ticker: Optional[str] = None) -> List[Dict]:
        """Get list of locally downloaded filings"""
        filings = []
        
        if ticker:
            ticker_dir = self.data_path / ticker
            if not ticker_dir.exists():
                return []
            
            search_dirs = [ticker_dir]
        else:
            search_dirs = [d for d in self.data_path.iterdir() if d.is_dir()]
        
        for ticker_dir in search_dirs:
            ticker_name = ticker_dir.name
            
            for filing_type_dir in ticker_dir.iterdir():
                if filing_type_dir.is_dir():
                    filing_type = filing_type_dir.name
                    
                    for filepath in filing_type_dir.glob("*.html"):
                        # Parse filename
                        filename = filepath.stem
                        parts = filename.split('_')
                        
                        filings.append({
                            'ticker': ticker_name,
                            'filing_type': filing_type,
                            'filepath': str(filepath),
                            'accession_number': parts[0] if parts else None,
                            'filing_date': parts[1] if len(parts) > 1 else None
                        })
        
        return filings


if __name__ == "__main__":
    downloader = EDGARDownloader()
    
    # Test with a single ticker
    test_ticker = "AAPL"
    
    print(f"Testing with {test_ticker}...")
    filings = downloader.download_recent_filings(test_ticker, max_filings=2)
    
    print(f"\nDownloaded {len(filings)} filings")
    for filing in filings:
        print(f"  {filing['filing_type']}: {filing['filing_date']}")
    
    # List local filings
    local_filings = downloader.get_local_filings(test_ticker)
    print(f"\nLocal filings: {len(local_filings)}")
