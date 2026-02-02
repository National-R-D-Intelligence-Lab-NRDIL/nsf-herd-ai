"""
NSF HERD Data Loader
Loads transformed CSVs into SQLite database (Q1 funding sources only)
"""

import pandas as pd
import sqlite3
from pathlib import Path
import re

class HERDLoader:
    def __init__(self, transformed_dir, db_path):
        self.transformed_dir = Path(transformed_dir)
        self.db_path = Path(db_path)
    
    def load_all(self):
        """Load all transformed CSVs into SQLite database"""
        print("="*80)
        print("NSF HERD DATA LOADER (Q1 Funding Sources)")
        print("="*80)
        print()
        
        # Find transformed files
        csv_files = sorted(self.transformed_dir.glob("herd_*.csv"))
        
        if not csv_files:
            print(f"❌ No transformed files found in {self.transformed_dir}")
            print(f"   Run 2_transform.py first")
            return
        
        print(f"📁 Found {len(csv_files)} transformed files")
        print(f"💾 Database: {self.db_path}")
        print()
        
        # Load all CSVs
        all_data = []
        
        for csv_file in csv_files:
            year = self._extract_year(csv_file.name)
            print(f"   📄 Loading {csv_file.name}...", end=" ")
            
            try:
                df = pd.read_csv(csv_file, dtype={'inst_id': str})
                all_data.append(df)
                print(f"✅ {len(df)} rows")
            except Exception as e:
                print(f"❌ Error: {e}")
        
        if not all_data:
            print("\n❌ No data loaded")
            return
        
        # Combine all years
        print()
        print("🔄 Combining data from all years...")
        combined = pd.concat(all_data, ignore_index=True)
        
        print(f"   Total rows: {len(combined):,}")
        print(f"   Total columns: {len(combined.columns)}")
        print(f"   Years: {combined['year'].min()} - {combined['year'].max()}")
        print(f"   Institutions: {combined['inst_id'].nunique():,}")
        
        # Save to SQLite
        print()
        print("💾 Writing to SQLite database...")
        
        # Create database directory
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Connect
        conn = sqlite3.connect(self.db_path)
        
        # Drop existing table (fresh load)
        conn.execute("DROP TABLE IF EXISTS institutions")
        
        # Write data
        combined.to_sql('institutions', conn, index=False, if_exists='replace')
        
        # Create indexes
        print("   Creating indexes...")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_inst_id ON institutions(inst_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_year ON institutions(year)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_state ON institutions(state)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_inst_year ON institutions(inst_id, year)")
        
        conn.close()
        
        # Verify
        print()
        print("="*80)
        print("✅ DATABASE CREATED SUCCESSFULLY")
        print("="*80)
        print()
        
        self._verify_database()
    
    def _extract_year(self, filename):
        """Extract year from filename like herd_2023.csv"""
        match = re.search(r'(\d{4})', filename)
        return int(match.group(1)) if match else None
    
    def _verify_database(self):
        """Quick verification of created database"""
        conn = sqlite3.connect(self.db_path)
        
        # Table info
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(institutions)")
        columns = [row[1] for row in cursor.fetchall()]
        
        print("Database schema:")
        print(f"  Table: institutions")
        print(f"  Columns ({len(columns)}): {', '.join(columns)}")
        print()
        
        # Sample data
        print("Sample data (first 3 institutions):")
        df_sample = pd.read_sql("""
            SELECT inst_id, name, state, year, total_rd
            FROM institutions
            LIMIT 3
        """, conn)
        print(df_sample.to_string(index=False))
        print()
        
        # Summary stats
        print("Quick stats:")
        stats = pd.read_sql("""
            SELECT 
                COUNT(*) as total_rows,
                COUNT(DISTINCT inst_id) as institutions,
                MIN(year) as first_year,
                MAX(year) as last_year,
                printf('$%,d', CAST(AVG(total_rd) AS INTEGER)) as avg_total_rd,
                printf('$%,d', MAX(total_rd)) as max_total_rd
            FROM institutions
        """, conn)
        
        for col in stats.columns:
            print(f"  {col}: {stats[col][0]}")
        
        # Check UNT
        print()
        print("UNT verification:")
        unt = pd.read_sql("""
            SELECT year, total_rd, federal, institutional
            FROM institutions
            WHERE inst_id = '003594'
            ORDER BY year DESC
            LIMIT 1
        """, conn)
        
        if not unt.empty:
            print(f"  UNT {unt['year'][0]}: ${unt['total_rd'][0]:,}")
            print(f"  Federal: ${unt['federal'][0]:,}")
            print(f"  Institutional: ${unt['institutional'][0]:,}")
        else:
            print("  ⚠️ UNT data not found")
        
        conn.close()
        
        print()
        print(f"📁 Database location: {self.db_path}")
        print()

if __name__ == "__main__":
    loader = HERDLoader(
        transformed_dir="C:/Users/kalya/nsf-herd-mvp/data/transformed",
        db_path="C:/Users/kalya/nsf-herd-mvp/data/herd.db"
    )
    loader.load_all()