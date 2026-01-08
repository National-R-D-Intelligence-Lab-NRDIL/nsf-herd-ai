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
        """Generate one-line summary from results only"""
        
        if results.empty:
            return "No data found."
        
        # Convert results to simple text
        results_text = results.to_string(index=False, max_rows=10)
        
        prompt = f"""Based ONLY on this data table, write ONE sentence summarizing the key finding.

Question: {question}

Data:
{results_text}

Rules:
- Use ONLY numbers from the table
- No speculation or external knowledge
- One sentence maximum
- Be specific with values

Summary:"""
        
        response = self.client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        
        return response.text.strip()