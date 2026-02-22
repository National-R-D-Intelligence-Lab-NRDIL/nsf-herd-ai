"""
NSF HERD Query Engine
Converts natural language questions into SQL queries against the HERD database,
executes them, and synthesizes the results into readable summaries.

Also provides the two direct queries the Snapshot feature needs:
rank trend over time, and the anchor view for a given institution.

Phase 2 upgrades:
- Context-aware SQL generation (selected institution, time window, state)
- Institution name resolution (abbreviations → inst_id)
- Rich few-shot examples across all 3 tables
- SQL validation and auto-retry on empty results
- Conversation history for follow-up questions
"""

from google import genai
import pandas as pd
import numpy as np
import sqlite3
import re
import time


# ======================================================================
# SCHEMA PROMPT — the core reference the LLM uses to write SQL.
# Structured as: tables → relationships → rules → examples
# ======================================================================
SCHEMA_PROMPT = """
=== DATABASE SCHEMA ===

Table: institutions
  Primary key: (inst_id, year)
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

Table: field_expenditures
  Primary key: (inst_id, year, field_code)
  - inst_id (TEXT)
  - year (INTEGER): 2010-2024
  - field_code (TEXT): Short code for the field (see valid values below)
  - parent_field (TEXT): Parent field code this belongs to
  - is_parent (INTEGER): 1 = parent-level category, 0 = sub-field
  - field_name (TEXT): Full display name (e.g. 'Engineering, all')
  - federal (INTEGER): Federal R&D dollars in this field
  - nonfederal (INTEGER): Nonfederal R&D dollars
  - total (INTEGER): Total R&D dollars (federal + nonfederal)

Table: agency_funding
  Primary key: (inst_id, year, agency_code)
  - inst_id (TEXT)
  - year (INTEGER): 2010-2024
  - agency_code (TEXT): One of: DOD, DOE, HHS, NASA, NSF, USDA, 'Other agencies'
  - agency_name (TEXT): Display name (e.g. 'Dept of Defense', 'HHS (incl. NIH)')
  - amount (INTEGER): Federal dollars from this agency

=== RELATIONSHIPS ===
- All 3 tables join on (inst_id, year)
- SUM(field_expenditures.total WHERE is_parent=1) = institutions.total_rd
- SUM(agency_funding.amount) = institutions.federal

=== VALID FIELD CODES ===

Parent fields (is_parent = 1):
  cs, engineering, geosciences, life_sciences, math, physical_sciences,
  psychology, social_sciences, other_sciences, non_se

Sub-fields (is_parent = 0):
  Engineering: eng_aerospace, eng_biomedical, eng_chemical, eng_civil,
    eng_electrical, eng_industrial, eng_mechanical, eng_materials, eng_other
  Life Sciences: life_agricultural, life_biomedical, life_health,
    life_natural_resources, life_other
  Physical Sciences: phys_astronomy, phys_chemistry, phys_materials,
    phys_physics, phys_other
  Social Sciences: soc_anthropology, soc_economics, soc_political,
    soc_sociology, soc_other
  Non-S&E: nse_business, nse_communication, nse_education, nse_humanities,
    nse_law, nse_social_work, nse_arts, nse_other
  Geosciences: geo_atmospheric, geo_earth, geo_ocean, geo_other

Note: cs, math, psychology, other_sciences have NO sub-fields.
Note: Missing field rows = $0 (institutions only report active fields).

=== VALID AGENCY CODES ===
  DOD, DOE, HHS, NASA, NSF, USDA, 'Other agencies'
  Note: HHS includes NIH. The survey does not break out NIH separately.

=== COMMON ALIASES ===
Users may use casual names. Map them to the correct codes:

Field aliases:
  "biomedical engineering", "BME", "biomed eng" → field_code = 'eng_biomedical'
  "comp sci", "CS", "computer science" → field_code = 'cs'
  "econ", "economics" → field_code = 'soc_economics'
  "poli sci", "political science" → field_code = 'soc_political'
  "physics" → field_code = 'phys_physics'
  "chemistry", "chem" → field_code = 'phys_chemistry'
  "astronomy", "astro" → field_code = 'phys_astronomy'
  "bio", "biology" → field_code = 'life_sciences' (parent) or check sub-fields
  "ag", "agriculture" → field_code = 'life_agricultural'
  "health sciences" → field_code = 'life_health'
  "non-science", "non-S&E", "humanities" → field_code = 'non_se' (parent) or 'nse_humanities'
  "materials", "materials science" → field_code = 'phys_materials' (available from 2016 only)

Agency aliases:
  "NIH", "National Institutes of Health" → agency_code = 'HHS' (NIH is part of HHS)
  "Pentagon", "military", "defense" → agency_code = 'DOD'
  "energy" → agency_code = 'DOE'
  "agriculture" → agency_code = 'USDA'
  "space" → agency_code = 'NASA'

=== SUB-FIELDS ADDED IN 2016 ===
These 4 sub-fields have NO DATA before 2016. Never compute CAGR with start year before 2016:
  eng_industrial, life_natural_resources, phys_materials, soc_anthropology

=== CRITICAL NAME MATCHING RULES ===

Institution names are INCONSISTENT across years. 259 institutions changed
names over the 15-year survey period. Examples:
  - 'University of North Texas' became 'University of North Texas, Denton'
  - 'SUNY Buffalo' → 'State University of New York, University at Buffalo'
  - 'Georgia Institute of Technology' (not 'Georgia Tech')
  - 'California Institute of Technology' (not 'Caltech')
  - 'Massachusetts Institute of Technology' (LIKE '%MIT%' also matches Smith College!)

RULES:
1. When an inst_id is provided in the context, ALWAYS use inst_id for filtering,
   not name. This is the only reliable way to track institutions across years.
2. When matching by name, use LIKE with wildcards AND be specific enough to avoid
   false matches. Example: name LIKE '%Massachusetts Institute of Technology%'
   NOT: name LIKE '%MIT%'
3. For names in output, always pull name from the latest year:
   Use a CTE: SELECT inst_id, name FROM institutions WHERE year = (SELECT MAX(year) FROM institutions)

=== CAGR FORMULA (SQLite) ===
ROUND((POWER(end_value * 1.0 / NULLIF(start_value, 0), 1.0 / num_years) - 1) * 100, 1)
Always protect against division by zero with NULLIF.
"""

# ======================================================================
# FEW-SHOT EXAMPLES — cover all 3 tables and common question patterns
# ======================================================================
FEW_SHOT_EXAMPLES = """
=== EXAMPLE QUERIES ===

Q: "Top 10 universities by total R&D in 2024"
SQL:
SELECT name, total_rd, 
       RANK() OVER (ORDER BY total_rd DESC) as national_rank
FROM institutions
WHERE year = 2024
ORDER BY total_rd DESC
LIMIT 10;

Q: "Which Texas schools grew faster than UNT from 2019 to 2024?"
Context: inst_id='003594' is the selected institution, state='TX', start_year=2019, end_year=2024
SQL:
WITH latest_names AS (
    SELECT inst_id, name FROM institutions
    WHERE year = (SELECT MAX(year) FROM institutions)
),
texas_cagr AS (
    SELECT i.inst_id,
           MAX(CASE WHEN i.year = 2019 THEN i.total_rd END) as rd_start,
           MAX(CASE WHEN i.year = 2024 THEN i.total_rd END) as rd_end,
           ROUND((POWER(
               MAX(CASE WHEN i.year = 2024 THEN i.total_rd END) * 1.0 /
               NULLIF(MAX(CASE WHEN i.year = 2019 THEN i.total_rd END), 0),
               1.0 / 5) - 1) * 100, 1) as cagr_5yr
    FROM institutions i
    WHERE i.state = 'TX' AND i.year IN (2019, 2024)
    GROUP BY i.inst_id
    HAVING rd_start > 0 AND rd_end > 0
),
unt_cagr AS (
    SELECT cagr_5yr FROM texas_cagr WHERE inst_id = '003594'
)
SELECT ln.name, tc.rd_start, tc.rd_end, tc.cagr_5yr,
       (SELECT cagr_5yr FROM unt_cagr) as unt_cagr
FROM texas_cagr tc
JOIN latest_names ln ON tc.inst_id = ln.inst_id
WHERE tc.cagr_5yr > (SELECT cagr_5yr FROM unt_cagr)
  AND tc.inst_id != '003594'
ORDER BY tc.cagr_5yr DESC;

Q: "Top 10 by engineering R&D in 2024"
SQL:
SELECT i.name, fe.total as engineering_rd,
       ROUND(fe.total * 100.0 / NULLIF(i.total_rd, 0), 1) as pct_of_portfolio,
       RANK() OVER (ORDER BY fe.total DESC) as engineering_rank
FROM field_expenditures fe
JOIN institutions i ON fe.inst_id = i.inst_id AND fe.year = i.year
WHERE fe.year = 2024 AND fe.field_code = 'engineering'
ORDER BY fe.total DESC
LIMIT 10;

Q: "Compare MIT, Stanford, and Caltech R&D from 2019 to 2024"
SQL:
SELECT i.name, i.year, i.total_rd, i.federal
FROM institutions i
WHERE i.name IN (
    'Massachusetts Institute of Technology',
    'Stanford University',
    'California Institute of Technology'
)
AND i.year BETWEEN 2019 AND 2024
ORDER BY i.name, i.year;

Q: "Which universities get the most NSF funding in 2024?"
SQL:
SELECT i.name, af.amount as nsf_funding,
       ROUND(af.amount * 100.0 / NULLIF(i.federal, 0), 1) as pct_of_federal,
       RANK() OVER (ORDER BY af.amount DESC) as nsf_rank
FROM agency_funding af
JOIN institutions i ON af.inst_id = i.inst_id AND af.year = i.year
WHERE af.year = 2024 AND af.agency_code = 'NSF'
ORDER BY af.amount DESC
LIMIT 10;

Q: "How does UNT's life sciences compare to other Texas schools?"
Context: inst_id='003594', state='TX', year=2024
SQL:
WITH latest_names AS (
    SELECT inst_id, name FROM institutions
    WHERE year = (SELECT MAX(year) FROM institutions)
),
tx_life AS (
    SELECT fe.inst_id, fe.total as life_sciences_rd,
           ROUND(fe.total * 100.0 / NULLIF(i.total_rd, 0), 1) as pct_of_portfolio,
           RANK() OVER (ORDER BY fe.total DESC) as state_rank
    FROM field_expenditures fe
    JOIN institutions i ON fe.inst_id = i.inst_id AND fe.year = i.year
    WHERE fe.year = 2024 AND fe.field_code = 'life_sciences' AND i.state = 'TX'
)
SELECT ln.name, tl.life_sciences_rd, tl.pct_of_portfolio, tl.state_rank
FROM tx_life tl
JOIN latest_names ln ON tl.inst_id = ln.inst_id
ORDER BY tl.life_sciences_rd DESC
LIMIT 15;

Q: "Which agencies increased funding to UNT the most from 2019 to 2024?"
Context: inst_id='003594', start_year=2019, end_year=2024
SQL:
WITH agency_growth AS (
    SELECT af.agency_code, af.agency_name,
           MAX(CASE WHEN af.year = 2019 THEN af.amount END) as amount_2019,
           MAX(CASE WHEN af.year = 2024 THEN af.amount END) as amount_2024
    FROM agency_funding af
    WHERE af.inst_id = '003594' AND af.year IN (2019, 2024)
    GROUP BY af.agency_code, af.agency_name
)
SELECT agency_name,
       COALESCE(amount_2019, 0) as funding_2019,
       COALESCE(amount_2024, 0) as funding_2024,
       COALESCE(amount_2024, 0) - COALESCE(amount_2019, 0) as change,
       CASE WHEN COALESCE(amount_2019, 0) > 0
            THEN ROUND((COALESCE(amount_2024, 0) * 1.0 / amount_2019 - 1) * 100, 1)
            ELSE NULL END as pct_change
FROM agency_growth
ORDER BY change DESC;

Q: "What are the fastest growing sub-fields at UNT?"
Context: inst_id='003594', start_year=2019, end_year=2024
SQL:
WITH subfield_growth AS (
    SELECT fe.field_code, fe.field_name, fe.parent_field,
           MAX(CASE WHEN fe.year = 2019 THEN fe.total END) as rd_2019,
           MAX(CASE WHEN fe.year = 2024 THEN fe.total END) as rd_2024
    FROM field_expenditures fe
    WHERE fe.inst_id = '003594' AND fe.is_parent = 0 AND fe.year IN (2019, 2024)
    GROUP BY fe.field_code, fe.field_name, fe.parent_field
    HAVING rd_2019 > 0 AND rd_2024 > 0
)
SELECT field_name, rd_2019, rd_2024,
       rd_2024 - rd_2019 as absolute_change,
       ROUND((POWER(rd_2024 * 1.0 / rd_2019, 1.0 / 5) - 1) * 100, 1) as cagr_5yr
FROM subfield_growth
ORDER BY cagr_5yr DESC;

Q: "How concentrated is UNT's federal funding compared to other universities?"
Context: inst_id='003594', year=2024
SQL:
WITH inst_concentration AS (
    SELECT af.inst_id,
           MAX(af.amount) as top_agency_amount,
           SUM(af.amount) as total_federal,
           ROUND(MAX(af.amount) * 100.0 / NULLIF(SUM(af.amount), 0), 1) as top_agency_pct,
           COUNT(DISTINCT af.agency_code) as num_agencies
    FROM agency_funding af
    WHERE af.year = 2024
    GROUP BY af.inst_id
    HAVING total_federal > 0
)
SELECT ln.name, ic.top_agency_pct, ic.total_federal, ic.num_agencies,
       RANK() OVER (ORDER BY ic.top_agency_pct DESC) as concentration_rank
FROM inst_concentration ic
JOIN (SELECT inst_id, name FROM institutions WHERE year = 2024) ln
  ON ic.inst_id = ln.inst_id
WHERE ic.inst_id = '003594'
   OR ic.top_agency_pct >= (SELECT top_agency_pct FROM inst_concentration WHERE inst_id = '003594') - 5
   AND ic.top_agency_pct <= (SELECT top_agency_pct FROM inst_concentration WHERE inst_id = '003594') + 5
ORDER BY ic.top_agency_pct DESC
LIMIT 15;

Q: "Which states have the highest total R&D?"
SQL:
SELECT state,
       SUM(total_rd) as total_state_rd,
       COUNT(DISTINCT inst_id) as num_institutions,
       ROUND(SUM(total_rd) * 1.0 / COUNT(DISTINCT inst_id), 0) as avg_per_institution
FROM institutions
WHERE year = 2024
GROUP BY state
ORDER BY total_state_rd DESC
LIMIT 15;

Q: "Where do we rank in engineering?" (context: inst_id='003594', state='TX')
-- COMPETITIVE BAND: rank by the specific metric, show ~8 above and ~7 below.
-- Include the selected institution IN the results with is_selected = 1.
-- Do NOT compare against all institutions — just the competitive neighborhood.
SQL:
WITH latest_names AS (
    SELECT inst_id, name FROM institutions
    WHERE year = (SELECT MAX(year) FROM institutions)
),
eng_ranked AS (
    SELECT fe.inst_id, fe.total as engineering_rd,
           RANK() OVER (ORDER BY fe.total DESC) as eng_rank,
           COUNT(*) OVER () as total_institutions
    FROM field_expenditures fe
    WHERE fe.year = 2024 AND fe.field_code = 'engineering'
      AND fe.total > 0
),
target AS (SELECT eng_rank FROM eng_ranked WHERE inst_id = '003594')
SELECT ln.name, er.engineering_rd, er.eng_rank,
       er.total_institutions,
       CASE WHEN er.inst_id = '003594' THEN 1 ELSE 0 END as is_selected
FROM eng_ranked er
JOIN latest_names ln ON er.inst_id = ln.inst_id
WHERE er.eng_rank BETWEEN (SELECT eng_rank FROM target) - 8
                       AND (SELECT eng_rank FROM target) + 7
ORDER BY er.eng_rank ASC;

Q: "Which Texas schools get more NSF funding than us but have lower total R&D?"
-- Cross-cutting query: joins institutions + agency_funding with two directional filters.
-- This is the kind of question the Q&A tab is designed for — no single tab can answer it.
Context: inst_id='003594', state='TX'
SQL:
WITH latest_names AS (
    SELECT inst_id, name FROM institutions
    WHERE year = (SELECT MAX(year) FROM institutions)
),
unt_vals AS (
    SELECT i.total_rd,
           (SELECT amount FROM agency_funding
            WHERE inst_id = '003594' AND year = 2024 AND agency_code = 'NSF') as nsf_amount
    FROM institutions i
    WHERE i.inst_id = '003594' AND i.year = 2024
)
SELECT ln.name, i.total_rd, af.amount as nsf_funding,
       ROUND(af.amount * 100.0 / NULLIF(i.federal, 0), 1) as nsf_pct_of_federal,
       (SELECT total_rd FROM unt_vals) as unt_total_rd,
       (SELECT nsf_amount FROM unt_vals) as unt_nsf_funding,
       CASE WHEN i.inst_id = '003594' THEN 1 ELSE 0 END as is_selected
FROM institutions i
JOIN agency_funding af ON i.inst_id = af.inst_id AND i.year = af.year
JOIN latest_names ln ON i.inst_id = ln.inst_id
WHERE i.year = 2024 AND i.state = 'TX'
  AND af.agency_code = 'NSF'
  AND af.amount > (SELECT nsf_amount FROM unt_vals)
  AND i.total_rd < (SELECT total_rd FROM unt_vals)
ORDER BY af.amount DESC;
"""


# Candidate anchor positions for the snapshot view.
ANCHOR_CANDIDATES = [1, 10, 25, 50, 100, 250, 500, 750]

# ======================================================================
# Common institution abbreviations → search patterns
# Used in name resolution to find the right inst_id.
# ======================================================================
INSTITUTION_ALIASES = {
    'MIT': 'Massachusetts Institute of Technology',
    'Caltech': 'California Institute of Technology',
    'Georgia Tech': 'Georgia Institute of Technology',
    'GT': 'Georgia Institute of Technology',
    'UNT': 'University of North Texas',
    'UT Austin': 'University of Texas at Austin',
    'UCLA': 'University of California, Los Angeles',
    'UCSD': 'University of California, San Diego',
    'UCSF': 'University of California, San Francisco',
    'UC Berkeley': 'University of California, Berkeley',
    'UCB': 'University of California, Berkeley',
    'USC': 'University of Southern California',
    'UNC': 'University of North Carolina at Chapel Hill',
    'OSU': 'Ohio State University',
    'PSU': 'Pennsylvania State University',
    'UMich': 'University of Michigan',
    'UW': 'University of Washington',
    'NYU': 'New York University',
    'CMU': 'Carnegie Mellon',
    'TAMU': 'Texas A&M University',
    'Texas A&M': 'Texas A&M University',
    'JHU': 'Johns Hopkins University',
    'Hopkins': 'Johns Hopkins University',
    'UF': 'University of Florida',
    'UVA': 'University of Virginia',
    'Vandy': 'Vanderbilt University',
    'Wash U': 'Washington University in St. Louis',
    'WashU': 'Washington University in St. Louis',
    'Rice': 'Rice University',
    'Purdue': 'Purdue University',
    'Penn': 'University of Pennsylvania',
    'UPenn': 'University of Pennsylvania',
    'Penn State': 'Pennsylvania State University',
    'ASU': 'Arizona State University',
    'CU Boulder': 'University of Colorado Boulder',
    'Duke': 'Duke University',
    'Yale': 'Yale University',
    'Harvard': 'Harvard University',
    'Stanford': 'Stanford University',
    'Cornell': 'Cornell University',
    'Princeton': 'Princeton University',
    'Northwestern': 'Northwestern University',
    'Emory': 'Emory University',
}


class HERDQueryEngine:
    """Handles the full pipeline: question -> SQL -> results -> summary"""

    def __init__(self, api_key, db_path):
        self.client = genai.Client(api_key=api_key)
        self.db_path = db_path
        # Cache the inst_id → latest name lookup (built lazily)
        self._name_cache = None

    # ----------------------------------------------------------
    # Direct DB access
    # ----------------------------------------------------------
    def _query(self, sql, params=None):
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql(sql, conn, params=params)
        conn.close()
        return df

    def _get_name_cache(self):
        """Lazy-load a dict: inst_id → latest name, and name → inst_id."""
        if self._name_cache is None:
            df = self._query("""
                SELECT inst_id, name
                FROM institutions
                WHERE year = (SELECT MAX(year) FROM institutions)
            """)
            self._name_cache = {
                'id_to_name': dict(zip(df['inst_id'], df['name'])),
                'name_to_id': dict(zip(df['name'], df['inst_id'])),
            }
        return self._name_cache

    # ----------------------------------------------------------
    # Institution list for dropdowns (unchanged API)
    # ----------------------------------------------------------
    def get_institution_list(self):
        return self._query("""
            WITH recent_names AS (
                SELECT DISTINCT inst_id, name
                FROM institutions
                WHERE year = (SELECT MAX(year) FROM institutions WHERE institutions.inst_id = inst_id)
            ),
            valid_institutions AS (
                SELECT inst_id
                FROM institutions
                WHERE year BETWEEN
                    (SELECT MAX(year) - 5 FROM institutions)
                    AND (SELECT MAX(year) FROM institutions)
                GROUP BY inst_id
                HAVING COUNT(DISTINCT year) >= 5
                   AND SUM(total_rd) > 0
            )
            SELECT rn.name
            FROM recent_names rn
            INNER JOIN valid_institutions vi ON rn.inst_id = vi.inst_id
            ORDER BY rn.name;
        """)['name'].tolist()

    def get_max_year(self):
        return int(self._query(
            "SELECT MAX(year) as max_year FROM institutions"
        )['max_year'].iloc[0])

    def get_min_year(self):
        return 2010

    # ----------------------------------------------------------
    # Name resolution: abbreviations/nicknames → inst_id
    # ----------------------------------------------------------
    def resolve_institution(self, name_or_abbrev):
        """Try to resolve a casual name/abbreviation to an inst_id.

        Returns (inst_id, canonical_name) or (None, None) if not found.
        Checks alias table first, then tries LIKE matching on the DB.
        """
        cache = self._get_name_cache()

        # 1. Exact match on current names
        if name_or_abbrev in cache['name_to_id']:
            return cache['name_to_id'][name_or_abbrev], name_or_abbrev

        # 2. Check alias table (case-insensitive)
        for alias, full_name in INSTITUTION_ALIASES.items():
            if name_or_abbrev.upper() == alias.upper():
                # Find inst_id by LIKE matching the full name.
                # Use latest year + highest total_rd to prefer main campus
                # when multiple campuses match (e.g. UNT Denton vs UNT Dallas).
                df = self._query("""
                    SELECT inst_id, name FROM institutions
                    WHERE name LIKE ? AND year = (SELECT MAX(year) FROM institutions)
                    ORDER BY total_rd DESC LIMIT 1
                """, params=(f'%{full_name}%',))
                if not df.empty:
                    return df['inst_id'].iloc[0], cache['id_to_name'].get(df['inst_id'].iloc[0], df['name'].iloc[0])

        # 3. LIKE match on current names
        df = self._query("""
            SELECT inst_id, name FROM institutions
            WHERE year = (SELECT MAX(year) FROM institutions)
              AND name LIKE ?
            ORDER BY total_rd DESC LIMIT 1
        """, params=(f'%{name_or_abbrev}%',))
        if not df.empty:
            return df['inst_id'].iloc[0], df['name'].iloc[0]

        return None, None

    # ----------------------------------------------------------
    # Build context block for the LLM
    # ----------------------------------------------------------
    def _build_context_block(self, context=None):
        """Build a context block that tells the LLM about the active session.

        context dict can contain:
          - institution_name: currently selected institution
          - inst_id: its ID
          - state: two-letter state code
          - start_year / end_year: active time window
          - peer_inst_ids: list of peer institution IDs
        """
        if not context:
            return ""

        lines = ["\n=== CURRENT SESSION CONTEXT ==="]

        if context.get('institution_name'):
            lines.append(f"Selected institution: {context['institution_name']}")
        if context.get('inst_id'):
            lines.append(f"  inst_id = '{context['inst_id']}' (use this for filtering, NOT name)")
        if context.get('state'):
            lines.append(f"  state = '{context['state']}'")
        if context.get('start_year') and context.get('end_year'):
            lines.append(f"Time window: {context['start_year']} to {context['end_year']}")
            lines.append(f"  Years between = {context['end_year'] - context['start_year']}")
        if context.get('peer_inst_ids'):
            ids = context['peer_inst_ids'][:5]
            lines.append(f"Peer inst_ids (top 5 of {len(context.get('peer_inst_ids', []))}): {ids}")

        lines.append("")
        lines.append("IMPORTANT: When the question refers to 'this institution', 'my institution',")
        lines.append("'our', 'we', or uses the selected institution's name/abbreviation,")
        lines.append(f"ALWAYS filter by inst_id = '{context.get('inst_id', '')}' instead of using name LIKE.")
        lines.append(f"When the question says 'peers', use the peer_inst_ids listed above.")
        lines.append("")
        lines.append("COMPARATIVE QUESTIONS ('who beats us', 'who's ahead', 'how do we compare',")
        lines.append("'who has more', 'who outperforms us', 'where do we rank'):")
        lines.append("  - NEVER compare against all institutions nationally — that produces")
        lines.append("    obvious, unhelpful results (e.g. Johns Hopkins beats everyone).")
        lines.append("  - If the question mentions a state or 'in-state', filter to that state.")
        lines.append("  - If the question mentions 'peers', filter to peer_inst_ids.")
        lines.append("  - Otherwise, use a COMPETITIVE BAND: rank all institutions by the")
        lines.append("    SPECIFIC METRIC in the question (e.g. engineering R&D, NSF funding,")
        lines.append("    life sciences growth), find the selected institution's rank, then")
        lines.append("    show ~8 institutions above and ~7 below. This gives the VPR their")
        lines.append("    realistic competitive neighborhood for that metric.")
        lines.append("  - Always include the selected institution IN the results so the VPR")
        lines.append("    sees exactly where they sit. Add a column:")
        lines.append(f"    CASE WHEN inst_id = '{context.get('inst_id', '')}' THEN 1 ELSE 0 END as is_selected")
        lines.append("  - Always include the institution's rank and the total count in results.")
        lines.append("")
        lines.append("GROWTH / CAGR QUERIES:")
        lines.append("  - Always include start and end dollar amounts alongside the CAGR percentage.")
        lines.append("  - Filter out institutions with less than $1M in the starting year to exclude noise.")
        lines.append("  - For sub-fields eng_industrial, life_natural_resources, phys_materials,")
        lines.append("    soc_anthropology: earliest valid year is 2016. Do not use earlier start years.")
        lines.append("")
        lines.append("AVOID REDUNDANCY WITH DASHBOARD TABS:")
        lines.append("  - The dashboard already shows: national rank (by total R&D), KNN peer")
        lines.append("    comparison, state ranking, field portfolio breakdown, agency donut chart.")
        lines.append("  - If the user asks for something a tab already shows, give a brief answer")
        lines.append("    but note: 'See the [tab name] tab for the full interactive view.'")
        lines.append("  - The Q&A tab's value is TARGETED EXPLORATION: discipline-specific rankings,")
        lines.append("    cross-table queries, custom filters, and comparisons the tabs can't do.")

        return "\n".join(lines)

    # ----------------------------------------------------------
    # AI-powered query pipeline (upgraded)
    # ----------------------------------------------------------
    def _clean_sql(self, text):
        """Extract SQL from whatever the model returns."""
        text = re.sub(r'```[\w]*\n?', '', text)
        text = re.sub(r'```', '', text)

        lines = text.split('\n')
        sql_lines = []
        found_start = False

        for line in lines:
            stripped = line.strip()
            if not found_start:
                if re.match(r'^(SELECT|INSERT|UPDATE|DELETE|WITH)', stripped, re.IGNORECASE):
                    found_start = True
                    sql_lines.append(stripped)
            else:
                if ';' in stripped:
                    sql_lines.append(stripped.split(';')[0] + ';')
                    break
                elif stripped and not stripped.startswith(('Note:', 'This', 'The', '--')):
                    sql_lines.append(stripped)

        sql = '\n'.join(sql_lines).strip()
        if sql and not sql.endswith(';'):
            sql += ';'
        return sql

    # Valid codes for deterministic validation of LLM-generated SQL
    VALID_FIELD_CODES = {
        # Parents
        'cs', 'engineering', 'geosciences', 'life_sciences', 'math',
        'physical_sciences', 'psychology', 'social_sciences', 'other_sciences', 'non_se',
        # Engineering sub-fields
        'eng_aerospace', 'eng_biomedical', 'eng_chemical', 'eng_civil',
        'eng_electrical', 'eng_industrial', 'eng_mechanical', 'eng_materials', 'eng_other',
        # Life Sciences sub-fields
        'life_agricultural', 'life_biomedical', 'life_health',
        'life_natural_resources', 'life_other',
        # Physical Sciences sub-fields
        'phys_astronomy', 'phys_chemistry', 'phys_materials', 'phys_physics', 'phys_other',
        # Social Sciences sub-fields
        'soc_anthropology', 'soc_economics', 'soc_political', 'soc_sociology', 'soc_other',
        # Non-S&E sub-fields
        'nse_business', 'nse_communication', 'nse_education', 'nse_humanities',
        'nse_law', 'nse_social_work', 'nse_arts', 'nse_other',
        # Geosciences sub-fields
        'geo_atmospheric', 'geo_earth', 'geo_ocean', 'geo_other',
    }
    VALID_AGENCY_CODES = {'DOD', 'DOE', 'HHS', 'NASA', 'NSF', 'USDA', 'Other agencies'}

    def _validate_codes(self, sql):
        """Check LLM-generated SQL for invalid field_code or agency_code values.

        Returns (is_valid, error_message). If invalid, the error message includes
        the bad code and the list of valid alternatives so the retry prompt can
        guide the LLM to the correct code.
        """
        # Extract field_code values: field_code = 'xxx' or field_code='xxx'
        field_matches = re.findall(r"field_code\s*=\s*'([^']+)'", sql, re.IGNORECASE)
        for code in field_matches:
            if code not in self.VALID_FIELD_CODES:
                # Find likely category from the bad code
                prefix = code.split('_')[0] if '_' in code else code
                suggestions = sorted([c for c in self.VALID_FIELD_CODES if c.startswith(prefix + '_') or c == prefix])
                if not suggestions:
                    suggestions = sorted(self.VALID_FIELD_CODES)
                return False, (
                    f"Invalid field_code '{code}'. "
                    f"Did you mean one of: {', '.join(suggestions)}? "
                    f"All valid codes: {', '.join(sorted(self.VALID_FIELD_CODES))}"
                )

        # Extract agency_code values
        agency_matches = re.findall(r"agency_code\s*=\s*'([^']+)'", sql, re.IGNORECASE)
        for code in agency_matches:
            if code not in self.VALID_AGENCY_CODES:
                return False, (
                    f"Invalid agency_code '{code}'. "
                    f"Valid codes: {', '.join(sorted(self.VALID_AGENCY_CODES))}. "
                    "Note: NIH is part of HHS — use agency_code = 'HHS'."
                )

        return True, ""

    def generate_sql(self, question, context=None):
        """Generate SQL from a natural language question.

        Args:
            question: The user's question in plain English
            context: Optional dict with session context (institution, time window, etc.)
        """
        context_block = self._build_context_block(context)

        prompt = f"""You are an expert SQL analyst for the NSF HERD (Higher Education Research & Development) database.
Given the schema and examples below, write a SQLite query to answer the user's question.

{SCHEMA_PROMPT}

{FEW_SHOT_EXAMPLES}

{context_block}

=== USER QUESTION ===
"{question}"

=== RULES ===
1. Return ONLY the SQL query. No explanation, no markdown.
2. Use clear, descriptive column aliases (e.g. 'engineering_rd' not 'total').
3. Always include institution names in results.
4. When an inst_id is provided in context, use it for filtering instead of name LIKE.
5. For name output, prefer the latest-year name using a CTE when joining across years.
6. Always protect against division by zero: use NULLIF(denominator, 0).
7. For growth calculations, require both start and end values to be > 0.
8. When comparing to a specific institution, include that institution's value for reference.
9. LIMIT results to a reasonable number (10-20) unless the user asks for all.
10. For field queries, specify is_parent = 1 for parent fields or is_parent = 0 for sub-fields.

SQL:"""

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
        """Run a SELECT query read-only."""
        cleaned = sql.strip().upper()
        if not cleaned.startswith("SELECT") and not cleaned.startswith("WITH"):
            raise ValueError(f"Only SELECT queries are allowed. Got: {sql[:50]}...")

        conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        try:
            result = pd.read_sql(sql, conn)
        finally:
            conn.close()
        return result

    def _validate_and_retry(self, question, sql, results, context=None):
        """If results are empty or suspicious, try to fix the query.

        Common failure modes:
          - Name mismatch (LIKE pattern too broad or too narrow)
          - Wrong field_code or agency_code
          - Missing is_parent filter
        Returns (sql, results) — possibly the originals if retry isn't needed.
        """
        if results is not None and len(results) > 0:
            return sql, results  # looks fine

        # Empty results — try a retry with explicit error feedback
        retry_prompt = f"""The following SQL query returned 0 rows. Fix it.

Original question: "{question}"

Failed SQL:
{sql}

{self._build_context_block(context)}

Common causes of empty results:
1. Name LIKE pattern was too specific or too broad. Use inst_id when available.
2. Wrong field_code — check against the valid codes list.
3. Missing or wrong is_parent filter.
4. Year out of range (data is 2010-2024).
5. Using name matching across years when names changed.

Write the corrected SQL query only:"""

        try:
            response = self.client.models.generate_content(
                model='gemini-2.5-flash',
                contents=retry_prompt
            )
            retry_sql = self._clean_sql(response.text)
            if retry_sql and retry_sql != sql:
                retry_results = self.execute_sql(retry_sql)
                if retry_results is not None and len(retry_results) > 0:
                    return retry_sql, retry_results
        except Exception:
            pass

        return sql, results  # return originals if retry also failed

    def ask(self, question, context=None):
        """Main entry point for free-form questions.

        Args:
            question: Plain English question
            context: Optional dict with {institution_name, inst_id, state,
                     start_year, end_year, peer_inst_ids}
        Returns:
            (sql, results_df, summary_text)
        """
        sql = self.generate_sql(question, context=context)

        # Validate field/agency codes before executing
        is_valid, code_error = self._validate_codes(sql)
        if not is_valid:
            # Retry with the specific error so the LLM can fix the code
            retry_prompt = f"""The following SQL has an invalid code. Fix it.

Original question: "{question}"

Failed SQL:
{sql}

Error: {code_error}

{self._build_context_block(context)}

Write the corrected SQL query only:"""
            try:
                response = self.client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=retry_prompt
                )
                retry_sql = self._clean_sql(response.text)
                is_valid_retry, _ = self._validate_codes(retry_sql)
                if is_valid_retry:
                    sql = retry_sql
            except Exception:
                pass  # proceed with original SQL — execute_sql may still work

        results = self.execute_sql(sql)

        # Auto-retry if empty results
        sql, results = self._validate_and_retry(question, sql, results, context=context)

        summary = self.summarize_results(question, results)
        return sql, results, summary

    def summarize_results(self, question, results):
        """Short insight summary of query results."""
        if results is None or results.empty:
            return "No data found for this query."

        results_text = results.to_string(index=False, max_rows=20)
        row_count = len(results)

        prompt = f"""You are a research funding analyst. Based ONLY on this data, write a 2-3 sentence insight.

Question: {question}

Data ({row_count} rows):
{results_text}

Guidelines:
- Lead with the key finding
- Include specific numbers and dollar amounts
- Add context (rankings, comparisons) where the data supports it
- Keep it direct, no filler words
- If showing growth rates, mention the baseline dollar amounts for context
- If the data contains a field_code or agency_code, name it explicitly in the summary
  (e.g. "Engineering: Biomedical (eng_biomedical)" or "HHS (includes NIH)")
  so the user can confirm the right discipline/agency was queried
- If there is an is_selected column, identify which row is the user's institution
  and frame the insight from their perspective (e.g. "You rank #119...")
- Use positioning language, not judgments. Say "ranked #119 of 487" not "low ranking."

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
        summary = summary.replace('$', '\\$')
        summary = summary.replace('\\\\$', '\\$')
        summary = ' '.join(summary.split())

        return summary

    # ==================================================================
    # Everything below is UNCHANGED from the original query_engine.py
    # (Snapshot, Portfolio, Federal Landscape, PDF methods)
    # ==================================================================

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

    def get_anchor_view(self, institution_name, year=2024):
        df_ranked = self._query("""
            SELECT name, total_rd,
                   RANK() OVER (ORDER BY total_rd DESC) as national_rank
            FROM institutions
            WHERE year = ?
            ORDER BY national_rank;
        """, params=(year,))

        total_institutions = len(df_ranked)

        target_rows = df_ranked[df_ranked['name'] == institution_name]
        if target_rows.empty:
            return pd.DataFrame(), 0, total_institutions

        target_rank = int(target_rows['national_rank'].values[0])

        anchors_above = [r for r in ANCHOR_CANDIDATES if r < target_rank and (target_rank - r) > 2]
        anchors_below = [r for r in ANCHOR_CANDIDATES if r > target_rank and (r - target_rank) > 2]

        selected = set()
        selected.add(1)
        selected.add(total_institutions)
        selected.update(anchors_above[-2:])
        selected.update(anchors_below[:2])
        selected.add(target_rank)

        anchor_df = df_ranked[df_ranked['national_rank'].isin(selected)].copy()
        anchor_df['is_target'] = anchor_df['name'] == institution_name
        anchor_df = anchor_df.sort_values('national_rank').reset_index(drop=True)

        return anchor_df, target_rank, total_institutions

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
            start_year, end_year, end_year, start_year,
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

        fed_pcts = self._query("""
            SELECT federal * 100.0 / NULLIF(total_rd, 0) as federal_pct
            FROM institutions
            WHERE year = ? AND total_rd > 0
            ORDER BY federal_pct
        """, params=(end_year,))

        if not fed_pcts.empty:
            national_median = round(float(fed_pcts['federal_pct'].median()), 1)
        else:
            national_median = 0.0

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
                       MAX(CASE WHEN year = ? THEN federal END) as fed_latest,
                       RANK() OVER (ORDER BY MAX(CASE WHEN year = ? THEN total_rd END) DESC) as state_rank
                FROM institutions
                WHERE state = ?
                GROUP BY name
            )
            SELECT name, rd_latest as total_rd, state_rank,
                   ROUND((POWER(rd_latest * 1.0 / NULLIF(rd_start, 0), 1.0/?) - 1) * 100, 1) as cagr,
                   ROUND(fed_latest * 100.0 / NULLIF(rd_latest, 0), 1) as federal_pct
            FROM state_inst
            WHERE rd_latest > 0
            ORDER BY state_rank
        """, params=(year, start_year, year, year, state, year - start_year))

        target_row = state_df[state_df['name'] == institution_name]
        if target_row.empty:
            return state_df, 0, 0, state

        rank = int(target_row['state_rank'].iloc[0])
        total_state_rd = state_df['total_rd'].sum()
        target_rd = int(target_row['total_rd'].iloc[0])
        market_share = round((target_rd / total_state_rd) * 100, 1)

        return state_df, rank, market_share, state

    def generate_strategic_insight(
        self,
        institution_name,
        start_year=2019,
        end_year=2024,
        bench_trend_stats=None,
        n_peers=None,
        custom_peer_mode=False,
    ):
        """Generate a strategic insight paragraph for the institution.

        ``bench_trend_stats`` — if provided (from KNN/custom benchmarker), its
        CAGR figures are used instead of the legacy resource-parity peer set so
        the insight is consistent with the Peer Analysis section on the page.
        """
        rank_df = self.get_rank_trend(institution_name, start_year, end_year)
        if rank_df.empty:
            return "Insufficient data for analysis."

        current_rank = int(rank_df.iloc[-1]['national_rank'])
        start_rank = int(rank_df.iloc[0]['national_rank'])

        _, trend_df, national_median = self.get_funding_breakdown(institution_name, start_year, end_year)
        state_df, state_rank, _, state = self.get_state_ranking(institution_name, end_year, start_year)

        federal_pct = round(trend_df.iloc[-1]['federal_pct'], 1) if not trend_df.empty else 0

        fields = self.get_field_portfolio(institution_name, end_year)
        agencies = self.get_agency_breakdown(institution_name, end_year)

        field_context = ""
        if not fields.empty:
            top_field = fields.iloc[0]
            field_context = f"- Largest field: {top_field['field_name'].replace(', all', '')} ({top_field['portfolio_share']}% of portfolio)"

        agency_context = ""
        if not agencies.empty:
            top_agency = agencies.iloc[0]
            agency_context = f"- Top federal agency: {top_agency['agency_name']} ({top_agency['pct_of_federal']}% of federal)"

        # Use KNN/custom bench stats when available so the insight matches
        # what is shown in the Peer Analysis section.  Fall back to the legacy
        # resource-parity peer set only when no benchmarker data exists.
        if bench_trend_stats:
            target_growth = bench_trend_stats.get('target_cagr', 0)
            peer_avg      = bench_trend_stats.get('peer_avg_cagr', 0)
            if custom_peer_mode:
                peer_desc = "custom peer avg"
            elif n_peers:
                peer_desc = f"{n_peers}-peer KNN avg"
            else:
                peer_desc = "KNN peer avg"
        else:
            _, peer_stats = self.get_peer_comparison(institution_name, start_year, end_year)
            target_growth = peer_stats.get('target_growth', 0)
            peer_avg      = peer_stats.get('peer_avg', 0)
            peer_desc     = "peer avg"

        prompt = f"""You are a senior research strategy analyst writing a briefing for a Vice President of Research. Write ONE concise paragraph (2-3 sentences, max 50 words) summarizing this institution's competitive position.

Data:
- Rank: #{current_rank} nationally (was #{start_rank} in {start_year})
- Growth (CAGR): {target_growth}% vs {peer_desc} {peer_avg}%
- Federal share: {federal_pct}% (national median: {national_median}%)
- State rank: #{state_rank} in {state}
{field_context}
{agency_context}

Rules:
- Use comparative positioning, not judgments. Say "ranked Nth" not "high risk."
- Never use words like risk, warning, concern, vulnerable, or should.
- State patterns and comparisons. Let the reader draw conclusions.
- Be specific with numbers. No filler.
- The growth comparison MUST reflect the actual numbers above. If target growth is below the peer avg, do NOT say it exceeds peers."""

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

    # ==================================================================
    # Research Portfolio tab methods (field_expenditures)
    # ==================================================================

    def get_field_portfolio(self, institution_name, year):
        return self._query("""
            SELECT
                fe.field_code,
                fe.field_name,
                fe.federal,
                fe.nonfederal,
                fe.total,
                ROUND(fe.total * 100.0 / NULLIF(i.total_rd, 0), 1) as portfolio_share,
                ROUND(fe.federal * 100.0 / NULLIF(fe.total, 0), 1) as federal_pct
            FROM field_expenditures fe
            JOIN institutions i ON fe.inst_id = i.inst_id AND fe.year = i.year
            WHERE i.name = ? AND fe.year = ? AND fe.is_parent = 1
            ORDER BY fe.total DESC
        """, params=(institution_name, year))

    def get_field_drilldown(self, institution_name, year, parent_field):
        return self._query("""
            SELECT
                fe.field_code,
                fe.field_name,
                fe.federal,
                fe.nonfederal,
                fe.total,
                ROUND(fe.federal * 100.0 / NULLIF(fe.total, 0), 1) as federal_pct,
                ROUND(fe.total * 100.0 / NULLIF(parent.total, 0), 1) as share_of_parent
            FROM field_expenditures fe
            JOIN institutions i ON fe.inst_id = i.inst_id AND fe.year = i.year
            LEFT JOIN field_expenditures parent
                ON parent.inst_id = fe.inst_id AND parent.year = fe.year
                AND parent.field_code = fe.parent_field AND parent.is_parent = 1
            WHERE i.name = ? AND fe.year = ?
              AND fe.parent_field = ? AND fe.is_parent = 0
            ORDER BY fe.total DESC
        """, params=(institution_name, year, parent_field))

    def get_field_momentum(self, institution_name, start_year, end_year):
        years_diff = end_year - start_year
        return self._query("""
            WITH latest AS (
                SELECT fe.field_code, fe.field_name, fe.total,
                       ROUND(fe.total * 100.0 / NULLIF(i.total_rd, 0), 1) as portfolio_share
                FROM field_expenditures fe
                JOIN institutions i ON fe.inst_id = i.inst_id AND fe.year = i.year
                WHERE i.name = ? AND fe.year = ? AND fe.is_parent = 1
            ),
            growth AS (
                SELECT fe.field_code,
                       MAX(CASE WHEN fe.year = ? THEN fe.total END) as rd_start,
                       MAX(CASE WHEN fe.year = ? THEN fe.total END) as rd_end
                FROM field_expenditures fe
                JOIN institutions i ON fe.inst_id = i.inst_id AND fe.year = i.year
                WHERE i.name = ? AND fe.is_parent = 1 AND fe.year IN (?, ?)
                GROUP BY fe.field_code
            )
            SELECT l.field_code, l.field_name, l.total, l.portfolio_share,
                   CASE
                       WHEN g.rd_start > 0 AND g.rd_end > 0
                       THEN ROUND((POWER(g.rd_end * 1.0 / g.rd_start, 1.0 / ?) - 1) * 100, 1)
                       ELSE NULL
                   END as cagr
            FROM latest l
            LEFT JOIN growth g ON l.field_code = g.field_code
            ORDER BY l.total DESC
        """, params=(institution_name, end_year, start_year, end_year,
                     institution_name, start_year, end_year, years_diff))

    def get_field_peer_comparison(self, inst_id, peer_inst_ids, year):
        all_ids = [inst_id] + peer_inst_ids
        placeholders = ','.join(['?'] * len(all_ids))

        df = self._query(f"""
            SELECT
                fe.inst_id,
                fe.field_code,
                fe.field_name,
                fe.total,
                ROUND(fe.total * 100.0 / NULLIF(
                    (SELECT SUM(total) FROM field_expenditures
                     WHERE inst_id = fe.inst_id AND year = fe.year AND is_parent = 1), 0
                ), 1) as portfolio_pct
            FROM field_expenditures fe
            WHERE fe.inst_id IN ({placeholders})
              AND fe.year = ? AND fe.is_parent = 1
        """, params=all_ids + [year])

        if df.empty:
            return pd.DataFrame()

        target = df[df['inst_id'] == inst_id][['field_code', 'field_name', 'portfolio_pct', 'total']].copy()
        target = target.rename(columns={'portfolio_pct': 'your_pct', 'total': 'your_total'})

        peers = df[df['inst_id'] != inst_id].groupby(['field_code', 'field_name']).agg(
            peer_avg_pct=('portfolio_pct', 'mean'),
        ).round(1).reset_index()

        result = pd.merge(target, peers, on=['field_code', 'field_name'], how='outer').fillna(0)
        result['difference'] = round(result['your_pct'] - result['peer_avg_pct'], 1)
        result = result.sort_values('difference', ascending=False)

        return result

    # ==================================================================
    # Federal Landscape tab methods (agency_funding)
    # ==================================================================

    def get_agency_breakdown(self, institution_name, year):
        return self._query("""
            SELECT
                af.agency_code,
                af.agency_name,
                af.amount,
                ROUND(af.amount * 100.0 / NULLIF(i.federal, 0), 1) as pct_of_federal
            FROM agency_funding af
            JOIN institutions i ON af.inst_id = i.inst_id AND af.year = i.year
            WHERE i.name = ? AND af.year = ?
            ORDER BY af.amount DESC
        """, params=(institution_name, year))

    def get_agency_trend(self, institution_name, start_year, end_year):
        return self._query("""
            SELECT af.year, af.agency_code, af.agency_name, af.amount
            FROM agency_funding af
            JOIN institutions i ON af.inst_id = i.inst_id AND af.year = i.year
            WHERE i.name = ? AND af.year BETWEEN ? AND ?
            ORDER BY af.year, af.amount DESC
        """, params=(institution_name, start_year, end_year))

    def get_agency_concentration(self, institution_name, year):
        agencies = self.get_agency_breakdown(institution_name, year)
        if agencies.empty:
            return None

        shares = agencies['pct_of_federal'].values / 100.0
        hhi = float(np.sum(shares ** 2))
        max_diverse = 1.0 - (1.0 / 7.0)
        diversification = round((1.0 - hhi) / max_diverse * 100, 1)

        top_agency = agencies.iloc[0]['agency_name']
        top_pct = float(agencies.iloc[0]['pct_of_federal'])

        all_top_pcts = self._query("""
            WITH inst_top AS (
                SELECT inst_id,
                       MAX(amount) as top_amount,
                       SUM(amount) as total_fed
                FROM agency_funding
                WHERE year = ?
                GROUP BY inst_id
                HAVING total_fed > 0
            )
            SELECT ROUND(top_amount * 100.0 / total_fed, 1) as top_pct
            FROM inst_top
            ORDER BY top_pct
        """, params=(year,))

        percentile = 0
        if not all_top_pcts.empty:
            percentile = int(round(
                (all_top_pcts['top_pct'] < top_pct).sum() / len(all_top_pcts) * 100, 0
            ))

        return {
            'hhi': round(hhi, 4),
            'diversification_score': diversification,
            'top_agency': top_agency,
            'top_agency_pct': top_pct,
            'national_percentile': percentile,
            'total_institutions': len(all_top_pcts),
        }

    def get_agency_peer_comparison(self, inst_id, peer_inst_ids, year):
        all_ids = [inst_id] + peer_inst_ids
        placeholders = ','.join(['?'] * len(all_ids))

        df = self._query(f"""
            SELECT
                af.inst_id,
                af.agency_code,
                af.agency_name,
                ROUND(af.amount * 100.0 / NULLIF(
                    (SELECT SUM(amount) FROM agency_funding
                     WHERE inst_id = af.inst_id AND year = af.year), 0
                ), 1) as agency_pct
            FROM agency_funding af
            WHERE af.inst_id IN ({placeholders}) AND af.year = ?
        """, params=all_ids + [year])

        if df.empty:
            return pd.DataFrame()

        target = df[df['inst_id'] == inst_id][['agency_code', 'agency_name', 'agency_pct']].copy()
        target = target.rename(columns={'agency_pct': 'your_pct'})

        peers = df[df['inst_id'] != inst_id].groupby(['agency_code', 'agency_name']).agg(
            peer_avg_pct=('agency_pct', 'mean'),
        ).round(1).reset_index()

        result = pd.merge(target, peers, on=['agency_code', 'agency_name'], how='outer').fillna(0)
        result['difference'] = round(result['your_pct'] - result['peer_avg_pct'], 1)
        result = result.sort_values('difference', ascending=False)

        return result

    # ==================================================================
    # Cross-tab helpers
    # ==================================================================

    def get_snapshot_callouts(self, institution_name, year):
        fields = self.get_field_portfolio(institution_name, year)
        agencies = self.get_agency_breakdown(institution_name, year)

        result = {}
        if not fields.empty:
            top = fields.iloc[0]
            result['top_field'] = top['field_name'].replace(', all', '')
            result['top_field_pct'] = float(top['portfolio_share'])
        if not agencies.empty:
            top = agencies.iloc[0]
            result['top_agency'] = top['agency_name']
            result['top_agency_pct'] = float(top['pct_of_federal'])

        return result

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
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{institution_name}</h1>
                <div class="subtitle">Research Intelligence Report | {start_year}-{end_year} | Generated {datetime.now().strftime('%B %d, %Y')}</div>
            </div>

            <h2>Executive Summary</h2>
            <div class="metrics">
                <div class="metric">
                    <div class="metric-label">Current Rank</div>
                    <div class="metric-value">#{metrics['current_rank']}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Total R&D ({end_year})</div>
                    <div class="metric-value">${metrics['current_rd']:,.0f}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">{end_year - start_year}-Year Growth ({start_year}–{end_year})</div>
                    <div class="metric-value">{metrics['target_growth']}%</div>
                </div>
            </div>

            <div class="insight">
                <strong>Strategic Insight:</strong> {insight}
            </div>

            <h2>National Position</h2>
            <img src="data:image/png;base64,{chart_images.get('rank_trend', '')}" alt="Rank Trend">
            <img src="data:image/png;base64,{chart_images.get('anchor_view', '')}" alt="National Position">

            <h2>Peer Performance Comparison</h2>
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
