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
    def get_peer_comparison(self, institution_name, start_year=2019, end_year=2024):
        baseline_rd = self._query("""
            SELECT total_rd FROM institutions 
            WHERE name = ? AND year = ?
        """, params=(institution_name, start_year))
        
        if baseline_rd.empty:
            return pd.DataFrame(), {}
        
        baseline = int(baseline_rd['total_rd'].iloc[0])
        lower = baseline * 0.8
        upper = baseline * 1.2
        
        peers_sql = """
        WITH baseline AS (
            SELECT name, total_rd
            FROM institutions
            WHERE year = ? AND total_rd BETWEEN ? AND ?
              AND name != ?
            ORDER BY ABS(total_rd - ?)
            LIMIT 8
        )
        SELECT i.name, i.year, i.total_rd
        FROM institutions i
        INNER JOIN baseline b ON i.name = b.name
        WHERE i.year BETWEEN ? AND ?
        
        UNION ALL
        
        SELECT name, year, total_rd
        FROM institutions
        WHERE name = ? AND year BETWEEN ? AND ?
        ORDER BY name, year
        """
        
        df = self._query(peers_sql, params=(
            start_year, lower, upper, institution_name, baseline,
            start_year, end_year,
            institution_name, start_year, end_year
        ))
        
        growth_sql = """
        SELECT name,
               MAX(CASE WHEN year = ? THEN total_rd END) as start_rd,
               MAX(CASE WHEN year = ? THEN total_rd END) as end_rd,
               ROUND((POWER(MAX(CASE WHEN year = ? THEN total_rd END) * 1.0 / 
                      NULLIF(MAX(CASE WHEN year = ? THEN total_rd END), 0), 
                      1.0/?) - 1) * 100, 1) as cagr
        FROM institutions
        WHERE name IN ({})
        GROUP BY name
        """.format(','.join(['?' for _ in range(len(df['name'].unique()))]))
        
        names = df['name'].unique().tolist()
        growth_df = self._query(growth_sql, params=(
            end_year, start_year, end_year, start_year, 
            end_year - start_year, *names
        ))
        
        target_growth = growth_df[growth_df['name'] == institution_name]['cagr'].iloc[0]
        peer_growth = growth_df[growth_df['name'] != institution_name]
        peer_avg = round(peer_growth['cagr'].mean(), 1)
        rank_in_peers = (peer_growth['cagr'] > target_growth).sum() + 1
        
        stats = {
            'target_growth': target_growth,
            'peer_avg': peer_avg,
            'rank': rank_in_peers,
            'total_peers': len(peer_growth) + 1
        }
        
        return df, stats

    def get_funding_breakdown(self, institution_name, start_year=2019, end_year=2024):
        latest = self._query("""
            SELECT federal, state_local, business, nonprofit, institutional, other_sources, total_rd
            FROM institutions
            WHERE name = ? AND year = ?
        """, params=(institution_name, end_year))
        
        if latest.empty:
            return pd.DataFrame(), pd.DataFrame(), 0
        
        trend = self._query("""
            SELECT year, 
                   ROUND(federal * 100.0 / NULLIF(total_rd, 0), 1) as federal_pct
            FROM institutions
            WHERE name = ? AND year BETWEEN ? AND ?
            ORDER BY year
        """, params=(institution_name, start_year, end_year))
        
        national_median = self._query(f"""
            SELECT AVG(federal_pct) as median
            FROM (
                SELECT federal * 100.0 / NULLIF(total_rd, 0) as federal_pct
                FROM institutions
                WHERE year = {end_year}
            )
        """)['median'].iloc[0]
        
        return latest, trend, round(national_median, 1)

    def get_state_ranking(self, institution_name, year=2024, start_year=2019):
        state_code = self._query("""
            SELECT state FROM institutions WHERE name = ? LIMIT 1
        """, params=(institution_name,))
        
        if state_code.empty:
            return pd.DataFrame(), 0, 0, ""
        
        state = state_code['state'].iloc[0]
        
        state_df = self._query("""
            WITH state_inst AS (
                SELECT name, 
                       MAX(CASE WHEN year = ? THEN total_rd END) as rd_latest,
                       MAX(CASE WHEN year = ? THEN total_rd END) as rd_start,
                       RANK() OVER (ORDER BY MAX(CASE WHEN year = ? THEN total_rd END) DESC) as state_rank
                FROM institutions
                WHERE state = ?
                GROUP BY name
            )
            SELECT name, rd_latest as total_rd, state_rank,
                   ROUND((POWER(rd_latest * 1.0 / NULLIF(rd_start, 0), 1.0/?) - 1) * 100, 1) as cagr
            FROM state_inst
            WHERE rd_latest > 0
            ORDER BY state_rank
        """, params=(year, start_year, year, state, year - start_year))
        
        target_row = state_df[state_df['name'] == institution_name]
        if target_row.empty:
            return state_df, 0, 0, state
        
        rank = int(target_row['state_rank'].iloc[0])
        total_state_rd = state_df['total_rd'].sum()
        target_rd = int(target_row['total_rd'].iloc[0])
        market_share = round((target_rd / total_state_rd) * 100, 1)
        
        return state_df, rank, market_share, state

    def generate_strategic_insight(self, institution_name, start_year=2019, end_year=2024):
        rank_df = self.get_rank_trend(institution_name, start_year, end_year)
        if rank_df.empty:
            return "Insufficient data for analysis."
        
        current_rank = int(rank_df.iloc[-1]['national_rank'])
        start_rank = int(rank_df.iloc[0]['national_rank'])
        
        _, peer_stats = self.get_peer_comparison(institution_name, start_year, end_year)
        _, trend_df, national_median = self.get_funding_breakdown(institution_name, start_year, end_year)
        state_df, state_rank, _, state = self.get_state_ranking(institution_name, end_year, start_year)
        
        federal_pct = round(trend_df.iloc[-1]['federal_pct'], 1) if not trend_df.empty else 0
        
        prompt = f"""You are a research strategy advisor. Write ONE sentence (max 30 words) summarizing competitive position and primary risk/opportunity.

Data:
- Rank: #{current_rank} (was #{start_rank} in {start_year})
- Growth: {peer_stats.get('target_growth', 0)}% vs peer avg {peer_stats.get('peer_avg', 0)}%
- Federal: {federal_pct}% (national median: {national_median}%)
- State: #{state_rank} in {state}

Be direct. Focus on competitive position and biggest risk/opportunity."""
        
        response = self.client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        
        return response.text.strip()

    def get_executive_metrics(self, institution_name, start_year=2019, end_year=2024):
        rank_df = self.get_rank_trend(institution_name, start_year, end_year)
        if rank_df.empty:
            return None
        
        current = rank_df.iloc[-1]
        start = rank_df.iloc[0]
        
        _, peer_stats = self.get_peer_comparison(institution_name, start_year, end_year)
        
        return {
            'current_rank': int(current['national_rank']),
            'start_rank': int(start['national_rank']),
            'rank_change': int(start['national_rank']) - int(current['national_rank']),
            'current_rd': int(current['total_rd']),
            'start_rd': int(start['total_rd']),
            'rd_change': int(current['total_rd']) - int(start['total_rd']),
            'target_growth': peer_stats.get('target_growth', 0),
            'peer_avg': peer_stats.get('peer_avg', 0),
            'rank_in_peers': peer_stats.get('rank', 0),
            'total_peers': peer_stats.get('total_peers', 0)
        }

    def _create_pdf_charts(self, institution_name, start_year, end_year):
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
        import io
        import base64
        
        charts = {}
        
        rank_df = self.get_rank_trend(institution_name, start_year, end_year)
        if not rank_df.empty:
            fig, ax = plt.subplots(figsize=(8, 4))
            years = rank_df['year'].tolist()
            ranks = rank_df['national_rank'].tolist()
            
            colors = ['#93C5FD'] * len(years)
            colors[-1] = '#2563EB'
            
            ax.barh(range(len(years)), ranks, color=colors)
            ax.set_yticks(range(len(years)))
            ax.set_yticklabels(years)
            ax.set_xlabel('National Rank')
            ax.invert_xaxis()
            ax.grid(axis='x', alpha=0.3)
            
            for i, (y, r) in enumerate(zip(years, ranks)):
                ax.text(r - 5, i, f'#{r}', va='center', ha='right', fontsize=10)
            
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            charts['rank_trend'] = base64.b64encode(buf.read()).decode()
            plt.close()
        
        anchor_df, _, _ = self.get_anchor_view(institution_name, end_year)
        if not anchor_df.empty:
            fig, ax = plt.subplots(figsize=(8, 5))
            
            names = anchor_df['name'].tolist()
            rd = anchor_df['total_rd'].tolist()
            is_target = anchor_df['is_target'].tolist()
            
            colors = ['#2563EB' if t else '#9CA3AF' for t in is_target]
            display_names = [n[:35] + "..." if len(n) > 35 else n for n in names]
            
            ax.barh(range(len(names)), rd, color=colors)
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(display_names, fontsize=9)
            ax.set_xlabel('Total R&D ($)')
            ax.ticklabel_format(style='plain', axis='x')
            
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            charts['anchor_view'] = base64.b64encode(buf.read()).decode()
            plt.close()
        
        peers_df, _ = self.get_peer_comparison(institution_name, start_year, end_year)
        if not peers_df.empty:
            fig, ax = plt.subplots(figsize=(8, 4))
            
            for name in peers_df['name'].unique():
                inst_data = peers_df[peers_df['name'] == name]
                is_target = name == institution_name
                
                ax.plot(
                    inst_data['year'],
                    inst_data['total_rd'],
                    marker='o',
                    linewidth=3 if is_target else 1,
                    color='#2563EB' if is_target else '#9CA3AF',
                    linestyle='-' if is_target else ':',
                    label=name if is_target else None
                )
            
            ax.set_xlabel('Year')
            ax.set_ylabel('Total R&D ($)')
            ax.ticklabel_format(style='plain', axis='y')
            ax.grid(alpha=0.3)
            if institution_name in peers_df['name'].values:
                ax.legend()
            
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            charts['peer_comparison'] = base64.b64encode(buf.read()).decode()
            plt.close()
        
        breakdown_df, _, _ = self.get_funding_breakdown(institution_name, start_year, end_year)
        if not breakdown_df.empty:
            fig, ax = plt.subplots(figsize=(6, 6))
            
            row = breakdown_df.iloc[0]
            sources = {
                'Federal': int(row['federal']),
                'Institutional': int(row['institutional']),
                'State/Local': int(row['state_local']),
                'Business': int(row['business']),
                'Nonprofit': int(row['nonprofit']),
                'Other': int(row['other_sources'])
            }
            
            ax.pie(
                list(sources.values()),
                labels=list(sources.keys()),
                autopct='%1.1f%%',
                startangle=90
            )
            ax.set_title(f'{end_year} Funding Sources')
            
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            charts['funding_pie'] = base64.b64encode(buf.read()).decode()
            plt.close()
        
        _, trend_df, national_median = self.get_funding_breakdown(institution_name, start_year, end_year)
        if not trend_df.empty:
            fig, ax = plt.subplots(figsize=(8, 4))
            
            ax.plot(
                trend_df['year'],
                trend_df['federal_pct'],
                marker='o',
                linewidth=2,
                color='#2563EB',
                label=institution_name
            )
            ax.axhline(
                y=national_median,
                color='red',
                linestyle='--',
                label=f'National Median ({national_median}%)'
            )
            
            ax.set_xlabel('Year')
            ax.set_ylabel('Federal %')
            ax.set_ylim(0, 100)
            ax.grid(alpha=0.3)
            ax.legend()
            ax.set_title('Federal Dependency Over Time')
            
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            charts['federal_trend'] = base64.b64encode(buf.read()).decode()
            plt.close()
        
        return charts

    def generate_pdf_report(self, institution_name, start_year, end_year, charts_dict=None):
        from datetime import datetime
        
        metrics = self.get_executive_metrics(institution_name, start_year, end_year)
        if not metrics:
            return None
        
        chart_images = self._create_pdf_charts(institution_name, start_year, end_year)
        
        insight = self.generate_strategic_insight(institution_name, start_year, end_year)
        breakdown_df, trend_df, _ = self.get_funding_breakdown(institution_name, start_year, end_year)
        state_df, state_rank, market_share, state = self.get_state_ranking(institution_name, end_year, start_year)
        
        federal = int(breakdown_df.iloc[0]['federal']) if not breakdown_df.empty else 0
        total = int(breakdown_df.iloc[0]['total_rd']) if not breakdown_df.empty else 1
        federal_pct = round((federal / total) * 100, 1)
        
        state_table_rows = ""
        for _, row in state_df.head(10).iterrows():
            highlight = 'background: #EFF6FF;' if row['name'] == institution_name else ''
            state_table_rows += f"""
            <tr style="{highlight}">
                <td>#{int(row['state_rank'])}</td>
                <td>{row['name']}</td>
                <td>${row['total_rd']:,.0f}</td>
                <td>{row['cagr']}%</td>
            </tr>
            """
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                @page {{ margin: 0.75in; }}
                body {{ font-family: Arial, sans-serif; color: #1e293b; line-height: 1.5; }}
                h1 {{ color: #0f172a; font-size: 24px; margin-bottom: 5px; }}
                h2 {{ color: #475569; font-size: 16px; margin-top: 30px; margin-bottom: 15px; 
                      border-bottom: 2px solid #e2e8f0; padding-bottom: 8px; }}
                .header {{ border-bottom: 2px solid #e2e8f0; padding-bottom: 15px; margin-bottom: 30px; }}
                .subtitle {{ color: #64748b; font-size: 12px; }}
                .metrics {{ display: flex; gap: 20px; margin: 20px 0; }}
                .metric {{ flex: 1; background: #f8fafc; padding: 15px; border-radius: 6px; }}
                .metric-label {{ color: #64748b; font-size: 11px; text-transform: uppercase; }}
                .metric-value {{ color: #0f172a; font-size: 24px; font-weight: bold; margin: 5px 0; }}
                .metric-change {{ color: #10b981; font-size: 12px; }}
                .insight {{ background: #eff6ff; padding: 15px; border-left: 4px solid #2563EB; margin: 20px 0; }}
                .insight strong {{ color: #1e40af; }}
                table {{ width: 100%; border-collapse: collapse; margin: 15px 0; font-size: 13px; }}
                th {{ background: #f1f5f9; padding: 10px; text-align: left; font-weight: 600; }}
                td {{ padding: 10px; border-bottom: 1px solid #e2e8f0; }}
                img {{ max-width: 100%; height: auto; margin: 15px 0; }}
                .footer {{ margin-top: 30px; padding-top: 15px; border-top: 1px solid #e2e8f0; 
                          color: #64748b; font-size: 11px; text-align: center; }}
                .risk {{ padding: 12px; border-radius: 6px; margin: 15px 0; font-weight: 600; }}
                .risk.low {{ background: #d1fae5; color: #065f46; }}
                .risk.med {{ background: #fef3c7; color: #92400e; }}
                .risk.high {{ background: #fee2e2; color: #991b1b; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{institution_name}</h1>
                <div class="subtitle">Research Intelligence Report | {start_year}–{end_year} | Generated {datetime.now().strftime('%B %d, %Y')}</div>
            </div>

            <h2>Executive Summary</h2>
            <div class="metrics">
                <div class="metric">
                    <div class="metric-label">Current Rank</div>
                    <div class="metric-value">#{metrics['current_rank']}</div>
                    <div class="metric-change">{'↑' if metrics['rank_change'] > 0 else '↓'}{abs(metrics['rank_change'])} since {start_year}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Total R&D ({end_year})</div>
                    <div class="metric-value">${metrics['current_rd']:,.0f}</div>
                    <div class="metric-change">+${metrics['rd_change']:,.0f}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">5-Year Growth</div>
                    <div class="metric-value">{metrics['target_growth']}%</div>
                    <div class="metric-change">vs {metrics['peer_avg']}% peer avg</div>
                </div>
            </div>
            
            <div class="insight">
                <strong>Strategic Insight:</strong> {insight}
            </div>

            <h2>National Position</h2>
            <img src="data:image/png;base64,{chart_images.get('rank_trend', '')}" alt="Rank Trend">
            <img src="data:image/png;base64,{chart_images.get('anchor_view', '')}" alt="National Position">

            <h2>Peer Performance Comparison</h2>
            <div class="metrics">
                <div class="metric">
                    <div class="metric-label">Your Growth</div>
                    <div class="metric-value">{metrics['target_growth']}%</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Peer Average</div>
                    <div class="metric-value">{metrics['peer_avg']}%</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Rank Among Peers</div>
                    <div class="metric-value">#{metrics['rank_in_peers']} of {metrics['total_peers']}</div>
                </div>
            </div>
            <img src="data:image/png;base64,{chart_images.get('peer_comparison', '')}" alt="Peer Comparison">

            <h2>Funding Source Analysis</h2>
            <div style="display: flex; gap: 20px;">
                <div style="flex: 1;">
                    <img src="data:image/png;base64,{chart_images.get('funding_pie', '')}" alt="Funding Sources" style="max-width: 400px;">
                </div>
                <div style="flex: 1;">
                    <img src="data:image/png;base64,{chart_images.get('federal_trend', '')}" alt="Federal Trend" style="max-width: 400px;">
                </div>
            </div>
            
            <div class="risk {'low' if federal_pct < 60 else 'med' if federal_pct < 70 else 'high'}">
                {'✓ LOW RISK - Diversified funding base' if federal_pct < 60 else 
                 '⚠ MODERATE RISK - Significant federal dependence' if federal_pct < 70 else
                 '⚠ HIGH RISK - Heavy federal dependence'}
            </div>

            <h2>{state} Competitive Position</h2>
            <div class="metrics">
                <div class="metric">
                    <div class="metric-label">State Rank</div>
                    <div class="metric-value">#{state_rank}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">State Market Share</div>
                    <div class="metric-value">{market_share}%</div>
                </div>
            </div>
            
            <table>
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Institution</th>
                        <th>2024 R&D</th>
                        <th>5-Yr CAGR</th>
                    </tr>
                </thead>
                <tbody>
                    {state_table_rows}
                </tbody>
            </table>

            <div class="footer">
                Generated by NSF HERD Research Intelligence | Data from NSF HERD Survey (2010-2024)
            </div>
        </body>
        </html>
        """
        
        try:
            from weasyprint import HTML
            pdf_bytes = HTML(string=html).write_pdf()
            return pdf_bytes
        except Exception as e:
            print(f"PDF generation error: {e}")
            return None