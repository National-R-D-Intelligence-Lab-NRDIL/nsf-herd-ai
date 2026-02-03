"""
NSF HERD Query Engine
Converts natural language questions into SQL queries against the HERD database,
executes them, and synthesizes the results into readable summaries.

Also provides the two direct queries the Snapshot feature needs:
rank trend over time, and the anchor view for a given institution.
"""

from google import genai
import pandas as pd
import sqlite3
import re
import time


# Database schema fed to the AI so it knows exactly what it's working with.
# Includes naming quirks and CAGR examples that trip up most models without guidance.
SCHEMA_PROMPT = """
Table: institutions
Columns:
- inst_id (TEXT): Institution identifier (e.g. '003594')
- name (TEXT): Full institution name as reported by NSF
- city (TEXT): City
- state (TEXT): Two-letter state code (e.g. 'TX')
- year (INTEGER): Fiscal year, range 2010-2024
- total_rd (INTEGER): Total R&D expenditure in dollars
- federal (INTEGER): Federal funding in dollars
- state_local (INTEGER): State and local government funding in dollars
- business (INTEGER): Business/industry funding in dollars
- nonprofit (INTEGER): Nonprofit organization funding in dollars
- institutional (INTEGER): Institution's own funding in dollars
- other_sources (INTEGER): All other funding sources in dollars

Name matching rules - these matter:
1. ALWAYS use LIKE with wildcards when matching by name. Names are inconsistent.
2. Many institutions have ', The' appended (e.g. 'University of Texas at Austin, The')
3. Some have city appended (e.g. 'University of North Texas, Denton')
4. When in doubt, use inst_id for exact matching instead of name strings.

CAGR (Compound Annual Growth Rate) calculation:
Formula: ((end_value / start_value) ^ (1/years) - 1) * 100

Example - 5-year CAGR across all institutions:
SELECT 
    name,
    MAX(CASE WHEN year = 2019 THEN total_rd END) as rd_2019,
    MAX(CASE WHEN year = 2024 THEN total_rd END) as rd_2024,
    ROUND((POWER(MAX(CASE WHEN year = 2024 THEN total_rd END) * 1.0 / 
           NULLIF(MAX(CASE WHEN year = 2019 THEN total_rd END), 0), 1.0/5) - 1) * 100, 1) as cagr_5yr
FROM institutions
WHERE year IN (2019, 2024)
GROUP BY name
HAVING rd_2019 > 0 AND rd_2024 > 0
ORDER BY cagr_5yr DESC;

Example - compare specific institutions:
SELECT name, year, total_rd, federal, institutional 
FROM institutions 
WHERE name LIKE '%Ohio State%' AND year BETWEEN 2020 AND 2024
ORDER BY year;

Example - top N by funding in a given year:
SELECT name, state, total_rd 
FROM institutions 
WHERE year = 2024 
ORDER BY total_rd DESC 
LIMIT 10;
"""

# Candidate anchor positions for the snapshot view.
# In the real DB (1,004 institutions) most of these will exist.
# The dynamic selection logic in get_anchor_view() filters out
# any that don't exist or are too close to the target.
ANCHOR_CANDIDATES = [1, 10, 25, 50, 100, 250, 500, 750]


class HERDQueryEngine:
    """Handles the full pipeline: question -> SQL -> results -> summary"""

    def __init__(self, api_key, db_path):
        self.client = genai.Client(api_key=api_key)
        self.db_path = db_path

    # ----------------------------------------------------------
    # Direct DB access — used by snapshot and institution list.
    # Keeps the connection lifecycle tight: open, run, close.
    # ----------------------------------------------------------
    def _query(self, sql, params=None):
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql(sql, conn, params=params)
        conn.close()
        return df

    # ----------------------------------------------------------
    # Institution list for the snapshot dropdown.
    # Pulls distinct names once — caller should cache this.
    # ----------------------------------------------------------
    def get_institution_list(self):
        # Get the most recent name for each inst_id to handle name changes over time
        # (e.g., "University of North Texas" became "University of North Texas, Denton" in 2011)
        return self._query("""
            WITH recent_names AS (
                SELECT DISTINCT inst_id, name
                FROM institutions
                WHERE year = (SELECT MAX(year) FROM institutions WHERE institutions.inst_id = inst_id)
            ),
            valid_institutions AS (
                SELECT inst_id
                FROM institutions
                WHERE year BETWEEN 2019 AND 2024
                GROUP BY inst_id
                HAVING COUNT(DISTINCT year) >= 5 
                   AND SUM(total_rd) > 0
            )
            SELECT rn.name
            FROM recent_names rn
            INNER JOIN valid_institutions vi ON rn.inst_id = vi.inst_id
            ORDER BY rn.name;
        """)['name'].tolist()

    # ----------------------------------------------------------
    # Snapshot: rank trend
    # Returns the institution's national rank for each year in
    # the selected window. RANK() is partitioned by year so each
    # year's ranking is independent.
    # ----------------------------------------------------------
    def get_rank_trend(self, institution_name, start_year=2019, end_year=2024):
        sql = """
        WITH ranked AS (
            SELECT
                name,
                year,
                total_rd,
                RANK() OVER (PARTITION BY year ORDER BY total_rd DESC) as national_rank
            FROM institutions
        )
        SELECT year, national_rank, total_rd
        FROM ranked
        WHERE name = ? AND year BETWEEN ? AND ?
        ORDER BY year;
        """
        return self._query(sql, params=(institution_name, start_year, end_year))

    # ----------------------------------------------------------
    # Snapshot: anchor view
    # Two-step process:
    #   1. Rank all institutions for the latest year in one pass
    #   2. Pick anchors dynamically based on where the target landed
    # Returns the target + surrounding benchmarks, sorted by rank.
    # ----------------------------------------------------------
    def get_anchor_view(self, institution_name, year=2024):
        # Step 1: full ranking for that year
        df_ranked = self._query("""
            SELECT name, total_rd,
                   RANK() OVER (ORDER BY total_rd DESC) as national_rank
            FROM institutions
            WHERE year = ?
            ORDER BY national_rank;
        """, params=(year,))

        total_institutions = len(df_ranked)

        # Find the target row — if missing, return empty
        target_rows = df_ranked[df_ranked['name'] == institution_name]
        if target_rows.empty:
            return pd.DataFrame(), 0, total_institutions

        target_rank = int(target_rows['national_rank'].values[0])

        # Step 2: pick anchors that are meaningfully spaced around the target.
        # Skip any candidate that's within 2 positions of the target — too close to be useful.
        anchors_above = [r for r in ANCHOR_CANDIDATES if r < target_rank and (target_rank - r) > 2]
        anchors_below = [r for r in ANCHOR_CANDIDATES if r > target_rank and (r - target_rank) > 2]

        selected = set()
        selected.add(1)                          # #1 is always shown
        selected.add(total_institutions)          # last place is always shown
        selected.update(anchors_above[-2:])       # 2 closest benchmarks above
        selected.update(anchors_below[:2])        # 2 closest benchmarks below
        selected.add(target_rank)                 # the target itself

        # Filter and tag
        anchor_df = df_ranked[df_ranked['national_rank'].isin(selected)].copy()
        anchor_df['is_target'] = anchor_df['name'] == institution_name
        anchor_df = anchor_df.sort_values('national_rank').reset_index(drop=True)

        return anchor_df, target_rank, total_institutions

    # ----------------------------------------------------------
    # AI-powered query pipeline (the original free-form Q&A path)
    # ----------------------------------------------------------
    def _clean_sql(self, text):
        """
        Pulls the actual SQL out of whatever the model returns.
        Models like to wrap SQL in markdown fences, add explanations after it,
        or include commentary lines — this strips all of that away.
        """
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
                # Skip lines that are clearly commentary, not SQL
                elif line and not line.startswith(('Note:', 'This', 'The', '--')):
                    sql_lines.append(line)

        sql = '\n'.join(sql_lines).strip()
        if sql and not sql.endswith(';'):
            sql += ';'
        return sql

    def generate_sql(self, question):
        """
        Sends the question + schema to Gemini, gets back a SQL query.
        Retries on 503s since the API can be flaky under load.
        """
        prompt = f"""Given this database schema:

{SCHEMA_PROMPT}

Convert this to a SQLite query: "{question}"

Rules:
- Use clear, descriptive column aliases
- Always include institution names in results
- Show both raw values and any calculated metrics
- Return ONLY the SQL query, nothing else."""

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
        """Runs the SQL against the local SQLite database, returns a DataFrame."""
        conn = sqlite3.connect(self.db_path)
        result = pd.read_sql(sql, conn)
        conn.close()
        return result

    def ask(self, question):
        """
        Main entry point for free-form questions. Takes plain English,
        runs it through the full pipeline, returns everything the UI needs.
        """
        sql = self.generate_sql(question)
        results = self.execute_sql(sql)
        summary = self.summarize_results(question, results)
        return sql, results, summary

    def summarize_results(self, question, results):
        """
        Takes the query results and asks Gemini to write a short, sharp summary.
        Keeps it to 2-3 sentences focused on the actual numbers.
        """
        if results.empty:
            return "No data found for this query."

        results_text = results.to_string(index=False, max_rows=20)
        row_count = len(results)

        prompt = f"""You are a research funding analyst. Based ONLY on this data, write a 2-3 sentence insight.

Question: {question}

Data ({row_count} rows):
{results_text}

Guidelines:
- Lead with the key finding
- Include specific numbers
- Add context (rankings, comparisons) where the data supports it
- Keep it direct, no filler words

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

        # Dollar signs and backslashes cause rendering issues in Streamlit
        summary = response.text.strip()
        summary = summary.replace('$', '').replace('\\', '')
        summary = ' '.join(summary.split())

        return summary