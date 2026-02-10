"""
NSF HERD Data Transformer (Auto-Discovery)
Converts LONG format to WIDE format with dynamic column discovery
"""

import pandas as pd
from pathlib import Path
import re
from collections import defaultdict

class HERDTransformer:
    def __init__(self, raw_dir, output_dir, questions_to_extract=None):
        self.raw_dir = Path(raw_dir)
        self.output_dir = Path(output_dir)
        self.questions_to_extract = questions_to_extract or ['01']
        
        # Track discoveries
        self.schema_discovered = defaultdict(lambda: {
            'questionnaire_numbers': set(),
            'row_values': set(),
            'years_present': []
        })
    
    def _standardize_row_value(self, row_val):
        """Convert row value to clean column name"""
        if pd.isna(row_val):
            return None
        
        clean = str(row_val).lower().strip()
        clean = re.sub(r'[^\w\s]', '', clean)  # Remove punctuation
        clean = re.sub(r'\s+', '_', clean)  # Spaces to underscores
        
        # Common standardizations
        replacements = {
            'federal_government': 'federal',
            'state_and_local_government': 'state_local',
            'nonprofit_organizations': 'nonprofit',
            'institution_funds': 'institutional',
            'all_other_sources': 'other_sources',
            'total': 'total_rd'
        }
        
        return replacements.get(clean, clean)
    
    def _clean_institution_names(self, df):
        """
        Standardize institution names by moving trailing ', The' to the front.
        
        NSF inconsistently appends ', The' to some institution names.
        We standardize by moving it to the front for visual consistency.
        
        Examples:
            "University of Alabama, The" -> "The University of Alabama"
            "Ohio State University, The" -> "The Ohio State University"
            "University of Texas at Austin" -> unchanged (no trailing ', The')
        
        Args:
            df: DataFrame with 'name' column
            
        Returns:
            DataFrame with cleaned 'name' column
        """
        def clean_name(name):
            if pd.isna(name):
                return name
            # If name ends with ", The" (case-insensitive), move it to front
            if re.search(r',\s*The\s*$', name, re.IGNORECASE):
                clean = re.sub(r',\s*The\s*$', '', name, flags=re.IGNORECASE).strip()
                return f"The {clean}"
            return name
        
        df['name'] = df['name'].apply(clean_name)
        return df
    
    def discover_and_transform_year(self, year):
        """Discover Q1 structure and transform one year"""
        csv_files = list(self.raw_dir.glob(f"*{year}*.csv"))
        
        if not csv_files:
            print(f"   ❌ No CSV for {year}")
            return False
        
        csv_file = csv_files[0]
        print(f"   📄 {year}: {csv_file.name}...", end=" ")
        
        try:
            # Read CSV
            df = pd.read_csv(
                csv_file,
                encoding='latin-1',
                dtype={'inst_id': str},
                low_memory=False
            )
            
            # Normalize column names
            df.columns = df.columns.str.strip()
            df['questionnaire_no'] = df['questionnaire_no'].astype(str)
            
            # Filter for Q1 funding sources only (01.a through 01.g)
            # Excludes 01.1 (institution fund details)
            df_q1 = df[df['questionnaire_no'].str.match(r'^01\.[a-g]$', na=False)].copy()
            
            if df_q1.empty:
                print(f"❌ No Q1 data")
                return False
            
            # Discover Q1 structure
            for _, row in df_q1.iterrows():
                quest_no = row['questionnaire_no']
                row_val = row['row']
                
                self.schema_discovered['01']['questionnaire_numbers'].add(quest_no)
                self.schema_discovered['01']['row_values'].add(row_val)
            
            self.schema_discovered['01']['years_present'].append(year)
            
            # Build mapping: questionnaire_no → column_name
            q1_mapping = {}
            for quest_no in df_q1['questionnaire_no'].unique():
                # Get the row value for this questionnaire_no
                sample = df_q1[df_q1['questionnaire_no'] == quest_no]
                if len(sample) > 0:
                    row_val = sample['row'].iloc[0]
                    col_name = self._standardize_row_value(row_val)
                    if col_name:
                        q1_mapping[quest_no] = col_name
            
            # Transform: group by institution
            transformed_rows = []
            
            for inst_id in df_q1['inst_id'].unique():
                inst_data = df_q1[df_q1['inst_id'] == inst_id]
                
                # Build row for this institution
                row = {
                    'inst_id': str(inst_id),
                    'name': inst_data['inst_name_long'].iloc[0],
                    'city': inst_data['inst_city'].iloc[0],
                    'state': inst_data['inst_state_code'].iloc[0],
                    'year': inst_data['year'].iloc[0]
                }
                
                # Add Q1 funding columns dynamically
                for quest_no, col_name in q1_mapping.items():
                    matching = inst_data[inst_data['questionnaire_no'] == quest_no]
                    if len(matching) > 0 and pd.notna(matching['data'].iloc[0]):
                        value_thousands = matching['data'].iloc[0]
                        row[col_name] = int(value_thousands * 1000)  # Convert to dollars
                    else:
                        row[col_name] = 0
                
                transformed_rows.append(row)
            
            # Create DataFrame
            df_transformed = pd.DataFrame(transformed_rows)
            
            # Clean institution names (move trailing ", The" to front)
            df_transformed = self._clean_institution_names(df_transformed)
            
            # Save
            self.output_dir.mkdir(parents=True, exist_ok=True)
            output_file = self.output_dir / f"herd_{year}.csv"
            # First, ensure inst_id is string
            df_transformed['inst_id'] = df_transformed['inst_id'].astype(str)
            # Then save
            df_transformed.to_csv(output_file, index=False)
            
            data_cols = len([c for c in df_transformed.columns if c not in ['inst_id', 'name', 'city', 'state', 'year']])
            print(f"✅ {len(df_transformed)} institutions, {data_cols} Q1 columns")
            
            return True
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def transform_all(self):
        """Transform all years and report discoveries"""
        print("="*80)
        print("NSF HERD DATA TRANSFORMATION (Auto-Discovery)")
        print("="*80)
        print()
        
        # Find years
        years = set()
        for csv_file in self.raw_dir.glob("*.csv"):
            match = re.search(r'(\d{4})', csv_file.name)
            if match:
                years.add(int(match.group(1)))
        
        if not years:
            print(f"❌ No CSV files in {self.raw_dir}")
            return
        
        print(f"🔎 Found {len(years)} years: {sorted(years)}")
        print()
        
        # Transform each
        success = 0
        for year in sorted(years):
            if self.discover_and_transform_year(year):
                success += 1
        
        print()
        print("="*80)
        print("TRANSFORMATION SUMMARY")
        print("="*80)
        print(f"✅ Transformed {success}/{len(years)} years")
        print()
        
        # Report discoveries
        print("SCHEMA DISCOVERY REPORT")
        print("-"*80)
        
        for question_prefix in sorted(self.schema_discovered.keys()):
            info = self.schema_discovered[question_prefix]
            years_found = sorted(info['years_present'])
            
            print(f"\nQuestion {question_prefix}:")
            print(f"  Years: {min(years_found)} - {max(years_found)} ({len(years_found)} years)")
            print(f"  Questionnaire numbers: {sorted(info['questionnaire_numbers'])}")
            print(f"  Row values discovered:")
            
            for row_val in sorted(info['row_values']):
                col_name = self._standardize_row_value(row_val)
                print(f"    {row_val:<40} → {col_name}")
        
        print()
        print("="*80)
        print(f"📁 Output: {self.output_dir}")
        print()

if __name__ == "__main__":
    transformer = HERDTransformer(
        raw_dir="C:/Users/kalya/nsf-herd-mvp/data/raw",
        output_dir="C:/Users/kalya/nsf-herd-mvp/data/transformed",
        questions_to_extract=['01']
    )
    transformer.transform_all()