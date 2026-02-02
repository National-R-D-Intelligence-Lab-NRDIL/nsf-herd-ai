"""
NSF HERD Data Downloader
Downloads HERD survey data from NSF website
"""

import requests
from bs4 import BeautifulSoup
from pathlib import Path
from urllib.parse import urljoin
import zipfile
import io
import re

class HERDDownloader:
    def __init__(self, output_dir):
        self.base_url = 'https://ncses.nsf.gov/explore-data/microdata/higher-education-research-development'
        self.output_dir = Path(output_dir)
        self.headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    
    def run(self, start_year=2010):
        """Download all HERD data from start_year to present"""
        print(f"⬇️  Starting download to {self.output_dir}...")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Get the webpage
        try:
            response = requests.get(self.base_url, headers=self.headers)
            response.raise_for_status()
        except Exception as e:
            print(f"❌ Error accessing NSF: {e}")
            return
        
        # Step 2: Find ZIP download links
        soup = BeautifulSoup(response.text, 'html.parser')
        zip_links = []
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            # Match pattern: Higher_Education_R_and_D_2023.zip
            if 'higher_education_r_and_d_' in href.lower() and href.endswith('.zip') and '_short' not in href:
                year_match = re.search(r'_(\d{4})\.zip', href)
                if year_match:
                    year = int(year_match.group(1))
                    if year >= start_year:
                        full_url = urljoin('https://ncses.nsf.gov', href)
                        zip_links.append((year, full_url))
        
        print(f"🔎 Found {len(zip_links)} datasets from {start_year} onward")
        
        # Step 3: Download each year
        for year, url in sorted(zip_links):
            self._download_year(year, url)
    
    def _download_year(self, year, url):
        """Download and extract one year's data"""
        # Check if already downloaded
        existing = list(self.output_dir.glob(f"*{year}*.csv"))
        if existing:
            print(f"   ✓ {year} already exists, skipping")
            return
        
        print(f"   ⏳ Downloading {year}...", end=" ")
        
        try:
            # Download ZIP
            response = requests.get(url, headers=self.headers, stream=True)
            response.raise_for_status()
            
            # Extract CSVs directly from memory (no ZIP saved to disk)
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                csv_files = [f for f in z.namelist() if f.endswith('.csv')]
                
                for csv_file in csv_files:
                    # Save CSV with year in filename
                    csv_data = z.read(csv_file)
                    output_path = self.output_dir / f"herd_{year}_{Path(csv_file).name}"
                    output_path.write_bytes(csv_data)
                    print(f"✅ {output_path.name}")
        
        except Exception as e:
            print(f"❌ Failed: {e}")

if __name__ == "__main__":
    downloader = HERDDownloader("../../data/raw")
    downloader.run(start_year=2010)