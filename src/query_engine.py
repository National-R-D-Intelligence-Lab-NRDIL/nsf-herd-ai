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
import yaml


class HERDQueryEngine:
    """AI-powered query engine for NSF HERD data"""

    def __init__(self, api_key, db_path):
        self.client = genai.Client(api_key=api_key)
        self.db_path = db_path
        
        # Load config
        config_file = os.getenv('CONFIG_FILE', 'configs/template.yml')
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f) 
        
        # Build peer ID strings from config
        inst_id = self.config['institution']['inst_id']
        inst_name = self.config['institution']['name']
        texas_ids = ", ".join([f"'{p['id']}'" for p in self.config['peers']['texas']])
        national_ids = ", ".join([f"'{p['id']}'" for p in self.config['peers']['national']])
        texas_with_inst = f"'{inst_id}', {texas_ids}"
        national_with_inst = f"'{inst_id}', {national_ids}"
        all_peer_ids = f"'{inst_id}', {texas_ids}, {national_ids}"
        
        self.schema = f"""
Table: institutions
Columns:
- inst_id (TEXT): Institution identifier
- name (TEXT): Full name
- city, state (TEXT): Location
- year (INTEGER): 2010-2024
- total_rd (INTEGER): Total R&D expenditure ($)
- federal, state_local, business, nonprofit, institutional, other_sources (INTEGER): Funding sources ($)

CRITICAL - Institution name matching rules:
1. ALWAYS use LIKE with wildcards for flexible matching
2. Many names end with ', The' (e.g., 'University of Texas at Austin, The')
3. Use inst_id for exact matching when available

Current Institution: inst_id = '{inst_id}' ({inst_name})

Texas Peers (with current institution):
{texas_with_inst}

National Peers (with current institution):
{national_with_inst}

All Peers:
{all_peer_ids}

Example - vs Texas peers:
SELECT name, total_rd FROM institutions 
WHERE year = 2024 AND inst_id IN ({texas_with_inst})
ORDER BY total_rd DESC;

Example - vs National peers:
SELECT name, total_rd FROM institutions 
WHERE year = 2024 AND inst_id IN ({national_with_inst})
ORDER BY total_rd DESC;

CAGR (Compound Annual Growth Rate) calculation:
Formula: ((end_value / start_value) ^ (1/years) - 1) * 100

Example - 5-year CAGR:
SELECT 
    name,
    MAX(CASE WHEN year = 2019 THEN total_rd END) as rd_2019,
    MAX(CASE WHEN year = 2024 THEN total_rd END) as rd_2024,
    ROUND((POWER(MAX(CASE WHEN year = 2024 THEN total_rd END) * 1.0 / 
           NULLIF(MAX(CASE WHEN year = 2019 THEN total_rd END), 0), 1.0/5) - 1) * 100, 1) as cagr_5yr
FROM institutions
WHERE inst_id IN ({texas_with_inst}) AND year IN (2019, 2024)
GROUP BY name
HAVING rd_2019 > 0 AND rd_2024 > 0
ORDER BY cagr_5yr DESC;
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
- Use clear column names
- Include institution names
- Show raw values AND calculations
- Return ONLY the SQL query."""

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
        
        results_text = results.to_string(index=False, max_rows=20)
        row_count = len(results)
        
        prompt = f"""You are a strategic research analyst. Based ONLY on this data, write a 2-3 sentence insight.

Question: {question}

Data ({row_count} rows):
{results_text}

Guidelines:
- State the KEY FINDING first
- Provide CONTEXT (rankings, comparisons)
- Use specific numbers
- Direct, executive tone

Summary:"""
        
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
        
        summary = response.text.strip()
        summary = summary.replace('$', '')
        summary = summary.replace('\\', '')
        summary = ' '.join(summary.split())
        
        return summary

    def generate_narrative(self, prompt):
        """Generate text using the model for narrative synthesis"""
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