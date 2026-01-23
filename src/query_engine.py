"""
NSF HERD Query Engine
Handles AI-powered SQL generation and execution
"""

from google import genai
import pandas as pd
import sqlite3
import re
import os
from pathlib import Path
import time


class HERDQueryEngine:
    """AI-powered query engine for NSF HERD data"""

    def __init__(self, api_key, db_path):
        self.client = genai.Client(api_key=api_key)
        self.db_path = db_path
        self.schema = """
Table: institutions
Columns:
- inst_id (TEXT): Institution identifier (e.g., '003594' for UNT)
- name (TEXT): Full name (e.g., 'University of North Texas, Denton')
- city, state (TEXT): Location
- year (INTEGER): 2010-2024
- total_rd (INTEGER): Total R&D expenditure ($)
- federal, state_local, business, nonprofit, institutional, other_sources (INTEGER): Funding sources ($)

CRITICAL - Institution name matching rules:
1. ALWAYS use LIKE with wildcards for flexible matching
2. Many names end with ', The' (e.g., 'University of Texas at Austin, The')
3. Names vary across years - same institution may have different name formats
4. Use multiple key terms with wildcards between them

Common abbreviations and their full forms:
- 'UT' = 'University of Texas'
- 'UNT' = 'University of North Texas'
- 'A&M' or 'A and M' = 'A&M' (use LIKE '%A_M%' to match both)
- 'SUNY' = 'State University of New York'
- 'CUNY' = 'City University of New York'
- 'Cal' or 'UC' = 'University of California'
- 'USC' = 'University of Southern California'
- 'MIT' = 'Massachusetts Institute of Technology'
- 'St.' or 'Saint' = use LIKE '%St%' to match both

Example name matching patterns:
- 'UT Austin' -> WHERE name LIKE '%University of Texas%Austin%'
- 'UT Dallas' -> WHERE name LIKE '%University of Texas%Dallas%'
- 'Texas A&M' (main) -> WHERE name LIKE '%Texas A_M%College Station%'
- 'Ohio State' -> WHERE name LIKE '%Ohio State%'
- 'UNT' or 'UNT Denton' -> WHERE name LIKE '%North Texas%Denton%'
- 'Penn State' -> WHERE name LIKE '%Pennsylvania State%'
- 'Michigan' (main) -> WHERE name LIKE '%University of Michigan%Ann Arbor%'
- 'SUNY Buffalo' -> WHERE name LIKE '%New York%Buffalo%'
- 'UC Berkeley' -> WHERE name LIKE '%California%Berkeley%'
- 'MIT' -> WHERE name LIKE '%Massachusetts Institute of Technology%'

For state filtering, use the state column:
- Texas universities: WHERE state = 'TX'
- California universities: WHERE state = 'CA'

Growth rate calculation:
To calculate growth rate between two years:
((end_value - start_value) / start_value) * 100 as growth_pct

Example - Growth rate query pattern:
SELECT 
    name,
    MAX(CASE WHEN year = 2020 THEN total_rd END) as start_value,
    MAX(CASE WHEN year = 2024 THEN total_rd END) as end_value,
    ROUND(((MAX(CASE WHEN year = 2024 THEN total_rd END) - MAX(CASE WHEN year = 2020 THEN total_rd END)) * 100.0 / 
           MAX(CASE WHEN year = 2020 THEN total_rd END)), 1) as growth_pct
FROM institutions
WHERE year IN (2020, 2024)
GROUP BY name
HAVING start_value > 0 AND end_value > 0
ORDER BY growth_pct DESC;

UNT PEER INSTITUTIONS (for benchmarking and comparison):
When user asks about "peers", "peer institutions", "benchmarking", or "how UNT compares":

Use inst_id for exact matching (more reliable than name matching):

UNT: inst_id = '003594'

Texas Peers (10 inst_ids):
'003658' -- UT Austin
'003632' -- Texas A&M (College Station)
'003656' -- UT Arlington
'009741' -- UT Dallas
'102077' -- UTRGV
'003661' -- UTEP
'010115' -- UTSA
'003652' -- University of Houston
'003644' -- Texas Tech
'003615' -- Texas State

National Peers (10 inst_ids):
'001081' -- Arizona State
'001574' -- Georgia State
'003954' -- University of Central Florida
'001825' -- Purdue
'001316' -- UC Riverside
'001776' -- University of Illinois Chicago
'003675' -- University of Utah
'330008' -- University of South Florida
'003509' -- University of Memphis
'002029' -- Tulane

Example - UNT vs Texas peers query:
SELECT name, total_rd
FROM institutions
WHERE year = 2024 AND inst_id IN ('003594', '003658', '003632', '003656', '009741', '102077', '003661', '010115', '003652', '003644', '003615')
ORDER BY total_rd DESC;

Example - UNT vs National peers query:
SELECT name, total_rd
FROM institutions
WHERE year = 2024 AND inst_id IN ('003594', '001081', '001574', '003954', '001825', '001316', '001776', '003675', '330008', '003509', '002029')
ORDER BY total_rd DESC;

Example - UNT vs ALL peers query:
SELECT name, total_rd
FROM institutions
WHERE year = 2024 AND inst_id IN ('003594', '003658', '003632', '003656', '009741', '102077', '003661', '010115', '003652', '003644', '003615', '001081', '001574', '003954', '001825', '001316', '001776', '003675', '330008', '003509', '002029')
ORDER BY total_rd DESC;

CAGR (Compound Annual Growth Rate) calculation:
CAGR measures average annual growth rate over a period.
Formula: ((end_value / start_value) ^ (1/years) - 1) * 100

Example - 5-year CAGR for total R&D:
SELECT 
    name,
    MAX(CASE WHEN year = 2019 THEN total_rd END) as rd_2019,
    MAX(CASE WHEN year = 2024 THEN total_rd END) as rd_2024,
    ROUND((POWER(MAX(CASE WHEN year = 2024 THEN total_rd END) * 1.0 / 
           NULLIF(MAX(CASE WHEN year = 2019 THEN total_rd END), 0), 1.0/5) - 1) * 100, 1) as cagr_5yr
FROM institutions
WHERE state = 'TX' AND year IN (2019, 2024)
GROUP BY name
HAVING MAX(CASE WHEN year = 2019 THEN total_rd END) > 0 AND MAX(CASE WHEN year = 2024 THEN total_rd END) > 0
ORDER BY cagr_5yr DESC;

Example - CAGR by funding source:
SELECT 
    name,
    ROUND((POWER(MAX(CASE WHEN year = 2024 THEN total_rd END) * 1.0 / 
           NULLIF(MAX(CASE WHEN year = 2019 THEN total_rd END), 0), 1.0/5) - 1) * 100, 1) as total_cagr,
    ROUND((POWER(MAX(CASE WHEN year = 2024 THEN federal END) * 1.0 / 
           NULLIF(MAX(CASE WHEN year = 2019 THEN federal END), 0), 1.0/5) - 1) * 100, 1) as federal_cagr,
    ROUND((POWER(MAX(CASE WHEN year = 2024 THEN institutional END) * 1.0 / 
           NULLIF(MAX(CASE WHEN year = 2019 THEN institutional END), 0), 1.0/5) - 1) * 100, 1) as institutional_cagr
FROM institutions
WHERE state = 'TX' AND year IN (2019, 2024)
GROUP BY name
HAVING MAX(CASE WHEN year = 2019 THEN total_rd END) > 0 AND MAX(CASE WHEN year = 2024 THEN total_rd END) > 0
ORDER BY total_cagr DESC;

CAGR benchmarks for Texas universities (2019-2024):
- Total R&D: Top performers 15-32%, median ~8%
- Federal: Top performers 20-27%, median ~10%
- Institutional: Top performers 21-35%, median ~8%
- Key insight: Institutions achieving 14%+ total CAGR typically have institutional investment CAGR above 20%
"""

    def _clean_sql(self, text):
        """Extract clean SQL from AI response"""
        text = re.sub(r'```[\w]*\n?', '', text)
        text = re.sub(r'```', '', text)

        lines = text.split('\n')
        sql_lines = []
        found_start = False

        for line in lines:
            line = line.strip()
            if not found_start:
                if re.match(r'^(SELECT|INSERT|UPDATE|DELETE|WITH)', line, re.IGNORECASE):
                    found_start = True
                    sql_lines.append(line)
            else:
                if ';' in line:
                    sql_lines.append(line.split(';')[0] + ';')
                    break
                elif line and not line.startswith(('Note:', 'This', 'The', '--')):
                    sql_lines.append(line)

        sql = '\n'.join(sql_lines).strip()
        if sql and not sql.endswith(';'):
            sql += ';'
        return sql

    def generate_sql(self, question):
        """Convert natural language to SQL with retry logic"""
        prompt = f"""Given this database schema:

{self.schema}

Convert to SQLite query: "{question}"

Rules:
- Use clear column names (avoid CAST, complex expressions in SELECT)
- Include institution names and identifying info
- Show raw values AND calculations (not just calculations)
- Sort ascending by default (oldest to newest, smallest to largest)
- Match institution names exactly as user specifies
- "UNT Denton" should match only 'University of North Texas, Denton'
- Don't use wildcards unless user says "all UNT campuses"
Return ONLY the SQL query."""

        # Retry up to 3 times for 503 errors
        for attempt in range(3):
            try:
                response = self.client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=prompt
                )
                return self._clean_sql(response.text)
            except Exception as e:
                if '503' in str(e) and attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                raise e

    def execute_sql(self, sql):
        """Execute SQL and return DataFrame"""
        conn = sqlite3.connect(self.db_path)
        result = pd.read_sql(sql, conn)
        conn.close()
        return result

    def ask(self, question):
        """Main interface: question -> SQL -> results -> summary"""
        sql = self.generate_sql(question)
        results = self.execute_sql(sql)
        summary = self.summarize_results(question, results)
        return sql, results, summary

    def summarize_results(self, question, results):
        """Generate strategic summary from results"""
        
        if results.empty:
            return "No data found."
        
        # Convert results to simple text
        results_text = results.to_string(index=False, max_rows=20)
        row_count = len(results)
        
        prompt = f"""You are a strategic research analyst. Based ONLY on this data, write a 2-3 sentence insight.

Question: {question}

Data ({row_count} rows):
{results_text}

Guidelines:
- State the KEY FINDING first (who is highest/lowest, what's the trend)
- Provide CONTEXT (rankings, comparisons, patterns)
- If relevant, note strategic implications (gaps, opportunities, benchmarks)
- Use specific numbers from the table
- Do NOT speculate beyond the data
- Do NOT use phrases like "based on the data" or "the table shows"
- Write in direct, executive tone

Example good summaries:
- "UNT Denton's 9.6% CAGR ranks 8th of 11 Texas peers. Texas State (20.7%) and UTSA (15.3%) demonstrate that 14%+ growth is achievable at similar institutional scale."
- "Federal funding grew 20% annually while institutional investment grew only 4.9%. High-growth peers like Texas State show 27% institutional CAGR, suggesting this is the primary gap."
- "UNT ranks 10th among Texas peers at $124M total R&D. The gap to 9th place (UT Arlington, $154M) is $30M."

Summary:"""
        
        # Retry up to 3 times for 503 errors
        for attempt in range(3):
            try:
                response = self.client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=prompt
                )
                break
            except Exception as e:
                if '503' in str(e) and attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                raise e
        
        # Clean up the response
        summary = response.text.strip()
        
        # Remove LaTeX-style math formatting that Gemini sometimes adds
        summary = summary.replace('$', '')
        summary = summary.replace('\\', '')
        
        # Clean up any double spaces
        summary = ' '.join(summary.split())
        
        return summary

    def generate_narrative(self, prompt):
        """Generate text using the model for narrative synthesis"""
        # Retry up to 3 times for 503 errors
        for attempt in range(3):
            try:
                response = self.client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=prompt
                )
                return response.text
            except Exception as e:
                if '503' in str(e) and attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                raise e