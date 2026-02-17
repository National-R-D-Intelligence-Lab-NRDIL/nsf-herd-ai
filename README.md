# NSF HERD Research Intelligence Platform

Strategic analytics dashboard for university R&D funding data. Transforms the NSF HERD Survey (1,004 institutions, 2010–2024) into competitive intelligence for Vice Presidents of Research and research administrators.

**Four core capabilities**: Institution Snapshot (competitive positioning), Research Portfolio (field-level analysis), Federal Landscape (agency funding analysis), and natural language Q&A (SQL-powered research queries).

---

## Quick Start

```bash
git clone https://github.com/yourusername/nsf-herd-mvp.git
cd nsf-herd-mvp
pip install -r requirements.txt
```

Create a `.env` file:

```
GEMINI_API_KEY=your_key_here
DATABASE_PATH=data/herd.db
PASSWORD=your_password_here
ALLOWED_INSTITUTIONS=  # optional: comma-separated inst_ids, empty = all allowed
```

Run it:

```bash
streamlit run app.py
```

Open `http://localhost:8501`, select your institution from the dropdown, and enter the password.

You need a Gemini API key from https://ai.google.dev/ and the HERD database file at `data/herd.db`. See the ETL section below if you need to build the database from scratch.

---

## What It Does

### Tab 1: Institution Snapshot

Select any institution and time window (5-year or 10-year) to get:

**Strategic Summary** – Current national rank, total R&D, 5-year growth rate, largest research field, and top federal agency. Compares your growth against your peer average with an AI-generated strategic insight that uses positioning language (never risk labels).

**Ranking Over Time** – Horizontal bar chart showing national rank for each year in the selected window. Shows net rank movement and total institutions in the survey.

**Where You Sit Nationally** – Anchor view placing your institution on a bar chart alongside benchmark positions (#1, #10, #50, #100, #250, etc.) so you can see where you sit relative to the full landscape. Your institution is highlighted in blue.

**Peer Analysis** – Uses K-Nearest Neighbors to find your 10 most similar institutions nationally across 7 funding dimensions (total R&D, federal, state/local, business, nonprofit, institutional, other). Shows:
- Growth metrics: Your CAGR, peer average CAGR, your growth rank
- Funding Profile: Grouped bar chart comparing your funding mix vs peer averages
- Growth Over Time: Line chart showing your trajectory vs your 10 peers
- Detailed gap numbers in an expander

**Funding Source Analysis** – Pie chart of current funding sources plus a federal share trend line plotted against the national median.

**State Competitive Position** – Your state rank and market share. Shows your competitive band (3 institutions above and 3 below you in state rank with federal dependency percentages). Top 10 state leaderboard available in an expander.

### Tab 2: Research Portfolio

Field-level R&D analysis across 10 parent research fields and 36 sub-fields:

**Portfolio Overview** – Stacked horizontal bar chart showing federal vs nonfederal funding for each of your 10 parent fields. Shows dollar amounts and portfolio share percentages.

**Field Momentum** – BCG-style scatter plot with portfolio share (x-axis) vs 5-year CAGR (y-axis). Each bubble is a research field. Fields in the upper-right quadrant are your strategic strengths: large and growing. Quadrant labels help identify Core Strengths, Emerging fields, Established Base, and Low Activity areas.

**Sub-field Drill-Down** – Expandable view for each parent field (Engineering, Life Sciences, Physical Sciences, etc.) showing its component disciplines with federal percentages and share of parent.

**Portfolio Distinctiveness** – Diverging bar chart comparing your field mix to your 10 nearest peers. Positive bars = you invest more in this field than peers do. Shows where your research portfolio is distinctive.

### Tab 3: Federal Landscape

Agency-level federal funding analysis across 7 agencies (DOD, DOE, HHS, NASA, NSF, USDA, Other):

**Federal Agency Breakdown** – Donut chart and data table showing your current agency distribution and each agency's share of your federal funding.

**Funding Diversification** – Three metrics:
- Diversification score (0–100%, based on Herfindahl-Hirschman Index)
- Top agency name and percentage
- National percentile (higher = more concentrated)

**Agency Funding Trends** – Multi-line chart showing funding from each agency over the selected time window. Growth summary table in an expander.

**Agency Distinctiveness** – Diverging bar chart comparing your agency mix to your 10 nearest peers. Shows which agencies you rely on more/less than similar institutions.

### Tab 4: Ask a Question

Natural language Q&A interface powered by Gemini AI:

**Context-aware SQL generation** – Automatically uses your selected institution's inst_id, state, time window, and peer list when interpreting questions like "How do we compare?" or "Show me our peers."

**Suggested questions** – Three expandable categories:
- How do we compare? (peer comparisons, state rankings, growth rates)
- Where's the momentum? (fastest growing fields, agency trends, funding shifts)
- What's distinctive? (portfolio differences, field concentration, federal positioning)

**Query execution** – Returns results as a data table with:
- Generated SQL (in an expander for transparency)
- Auto-generated chart (when applicable)
- AI-generated 2–3 sentence summary
- CSV download button

**Example questions**:
- "Which peer institutions have a larger engineering portfolio share than us?"
- "How does our life sciences R&D compare to other Texas schools?"
- "What are the fastest growing sub-fields at our institution?"
- "Which agencies increased their funding to us the most from 2019 to 2024?"
- "How concentrated is our federal funding compared to other universities?"

Rate limited to 50 queries per hour per session.

---

## Project Structure

```
nsf-herd-mvp/
├── app.py                     # Streamlit app (4 tabs, UI, charts, routing)
├── src/
│   ├── query_engine.py        # All database queries, SQL generation, AI integration
│   └── benchmarker.py         # KNN peer-finding algorithm
├── scripts/
│   └── etl/
│       ├── 1_download.py      # Fetch raw HERD data from NSF
│       ├── 2_transform.py     # Transform Q1 funding sources
│       ├── 3_load.py          # Load institutions table
│       ├── 4_transform_fields.py    # Transform Q9+Q11 field data
│       ├── 5_transform_agencies.py  # Transform Q9 agency data
│       └── 6_load_extended.py       # Load field + agency tables
├── data/
│   └── herd.db                # SQLite database (40.5 MB, not in repo)
├── .streamlit/
│   └── config.toml            # Streamlit theme and server settings
├── .devcontainer/
│   └── devcontainer.json      # GitHub Codespaces / VS Code dev container
├── requirements.txt
├── Procfile                   # Railway deployment
├── .env                       # Environment variables (gitignored)
└── .gitignore
```

### How the pieces fit together

`app.py` handles all UI rendering and calls into two backend modules. `query_engine.py` owns every database query – rank trends, anchor views, funding breakdowns, state rankings, peer comparisons, field portfolio analysis, agency breakdowns, concentration metrics, and the full LLM-to-SQL pipeline for the Q&A tab. `benchmarker.py` handles the KNN peer-finding algorithm. It loads the latest year of funding data, log-transforms and scales it, fits a NearestNeighbors model, and exposes methods to find peers, analyze gaps, and pull historical trends.

The benchmarker is fitted once at app startup and cached via `@st.cache_resource`. The institution list is cached with a 1-hour TTL via `@st.cache_data`.

---

## Database Schema (Phase 2)

Three tables with enforced invariants:

```
Table: institutions
├── inst_id         TEXT       PK with year
├── name            TEXT
├── city            TEXT
├── state           TEXT       Two-letter code (e.g. 'TX')
├── year            INTEGER    PK with inst_id (2010–2024)
├── total_rd        INTEGER    Total R&D expenditure in dollars
├── federal         INTEGER    Federal funding
├── state_local     INTEGER    State and local government funding
├── business        INTEGER    Business/industry funding
├── nonprofit       INTEGER    Nonprofit funding
├── institutional   INTEGER    Institution's own funding
└── other_sources   INTEGER    All other sources
    Source: HERD Q1 (funding sources)
    Rows: 10,084 (1,004 institutions × 15 years)

Table: field_expenditures
├── inst_id         TEXT       PK with year + field_code
├── year            INTEGER
├── field_code      TEXT       e.g. 'engineering', 'eng_biomedical'
├── parent_field    TEXT       e.g. 'engineering' (self-referencing for parents)
├── is_parent       INTEGER    1 = parent category, 0 = sub-field
├── field_name      TEXT       NSF's original label, for display
├── federal         INTEGER    Federal R&D dollars in this field
├── nonfederal      INTEGER    Nonfederal R&D dollars
└── total           INTEGER    federal + nonfederal
    Source: HERD Q9 (federal by field) + Q11 (nonfederal by field)
    Rows: 217,577 (70,426 parent + 147,151 sub-field)
    Coverage: 10 parent fields + 36 sub-fields = 46 field codes

Table: agency_funding
├── inst_id         TEXT       PK with year + agency_code
├── year            INTEGER
├── agency_code     TEXT       One of: DOD, DOE, HHS, NASA, NSF, USDA, 'Other agencies'
├── agency_name     TEXT       Display name (e.g. 'HHS (incl. NIH)')
└── amount          INTEGER    Federal dollars from this agency
    Source: HERD Q9 (federal by field and agency, row='All')
    Rows: 70,042
    Coverage: 7 federal agencies
```

### Data Invariants

These relationships always hold and are verified after ETL:

- `SUM(field_expenditures.total WHERE is_parent=1)` = `institutions.total_rd`
- `SUM(field_expenditures.federal WHERE is_parent=1)` = `institutions.federal`
- `SUM(agency_funding.amount)` = `institutions.federal`
- For any parent with sub-fields: `parent.total` = `SUM(sub-field.total)`

### The 10 Parent Fields

| field_code | Display Name | Has Sub-fields |
|---|---|---|
| cs | Computer & Info Sciences | No |
| engineering | Engineering | Yes (9 sub-fields) |
| geosciences | Geosciences & Ocean Sciences | Yes (4 sub-fields) |
| life_sciences | Life Sciences | Yes (5 sub-fields) |
| math | Mathematics & Statistics | No |
| physical_sciences | Physical Sciences | Yes (5 sub-fields) |
| psychology | Psychology | No |
| social_sciences | Social Sciences | Yes (5 sub-fields) |
| other_sciences | Other Sciences | No |
| non_se | Non-S&E | Yes (8 sub-fields) |

Four parents (CS, Math, Psychology, Other Sciences) are standalone – the HERD survey doesn't break them into sub-disciplines.

### The 7 Federal Agencies

- **DOD** – Dept of Defense
- **DOE** – Dept of Energy
- **HHS** – HHS (incl. NIH). Note: NIH is not broken out separately in the HERD survey.
- **NASA** – NASA
- **NSF** – NSF
- **USDA** – USDA
- **Other agencies** – Other Federal

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GEMINI_API_KEY` | Yes | – | Google Gemini API key |
| `PASSWORD` | Yes | – | Login password. App refuses to start without it. |
| `DATABASE_PATH` | No | `data/herd.db` | Path to SQLite database |
| `ALLOWED_INSTITUTIONS` | No | `` (empty) | Comma-separated inst_ids for access control. Empty = all allowed. |
| `GOOGLE_SHEET_ID` | No | – | Google Sheet ID for usage logging |
| `GOOGLE_SHEETS_CREDS` | No | – | Service account credentials JSON string |

---

## Deployment

### Railway

The repo includes a `Procfile`. Set your environment variables in the Railway dashboard and deploy.

### Streamlit Cloud

Push to GitHub, connect at share.streamlit.io, add `GEMINI_API_KEY`, `PASSWORD`, and optionally `ALLOWED_INSTITUTIONS` as secrets.

### GitHub Codespaces

The `.devcontainer/devcontainer.json` is pre-configured. Open the repo in Codespaces and it installs dependencies and starts the app automatically on port 8501.

---

## ETL Pipeline

When NSF releases new HERD data (annually):

```bash
cd scripts/etl
python 1_download.py            # Fetch microdata ZIPs from NSF
python 2_transform.py           # Transform Q1 → institutions CSV
python 3_load.py                # Load institutions table
python 4_transform_fields.py    # Transform Q9+Q11 → field_expenditures CSV
python 5_transform_agencies.py  # Transform Q9 → agency_funding CSV
python 6_load_extended.py       # Load field + agency tables
```

After loading new data, restart the app to refresh the benchmarker cache. The institution list and year ranges are derived from the database automatically – no code changes needed unless NSF restructures a question.

**What future NSF changes would require**:
- New survey year, same structure → Just re-run ETL pipeline
- New sub-field added → Add entry to `FIELD_TAXONOMY` in `4_transform_fields.py`
- New parent field added → Add to `FIELD_TAXONOMY`, update UI tab
- New agency column → Add to `AGENCIES` in `5_transform_agencies.py`
- New question added → New table + new ETL script

---

## How the KNN Peer Algorithm Works

The old approach (resource parity, ±20% of total R&D) failed for outliers. Johns Hopkins at $4.1B has zero institutions within 20% of its total R&D. Carthage College at $188K has one.

The KNN approach:

1. Pull one row per institution from the latest survey year (681 institutions with complete data).
2. Log-transform all 7 numeric columns with `np.log1p`. This compresses the 22,000x skew so $1M vs $2M gets similar weight to $1B vs $2B.
3. Standardize to zero-mean, unit-variance with `StandardScaler`.
4. Fit a `NearestNeighbors` model (euclidean distance) in that space.
5. For any institution, return the k=10 closest neighbors.

`total_rd` is included alongside the 6 funding source columns even though it's their sum. This is intentional – it gives overall size 2x weight in the distance calculation, which prevents a $300M school from matching to a $3B school just because they have similar funding ratios.

**Important notes**:
- Peer matching is not symmetric. If UNT lists Texas State as a peer, Texas State may not list UNT back. This is normal for KNN and matches how other benchmarking tools (IPEDS, Delaware Study) work.
- The benchmarker fits on the 681 institutions in the latest year with complete data, while the institution dropdown filters for institutions with 5+ consecutive years (612 institutions). Some institutions may be in one set but not the other. The system handles this gracefully.
- For historical trends, the benchmarker joins on `inst_id` (not name) because 259 institutions changed names over the 15-year survey period.

---

## Q&A Tab: Context-Aware SQL Generation

The Ask a Question tab uses Gemini AI to convert natural language to SQL. As of February 2026, it includes context awareness and automatic retry on failures.

**How context flows**:
1. User selects institution in the top-level picker
2. `app.py` resolves inst_id, state, start_year, end_year, and peer_inst_ids
3. Stores them in session state
4. User types a question in the Q&A tab
5. `process_question()` builds a context dict and passes it to `engine.ask()`
6. `generate_sql()` injects context into the prompt:
   - "Selected institution: University of North Texas, Denton"
   - "inst_id = '003594' (use this for filtering, NOT name)"
   - "state = 'TX'"
   - "Time window: 2019 to 2024"
   - "Peer inst_ids: ['001305', '002155', ...]"
7. LLM generates SQL using inst_id, not name LIKE patterns
8. If 0 rows → `_validate_and_retry()` sends failed SQL back for correction
9. Results → `create_visualization()` (reference columns excluded) → chart

**Name resolution**: The system includes 45+ common abbreviations (MIT, Caltech, UNT, TAMU, etc.) that map to full NSF names. When multiple campuses match (e.g., "University of North Texas" matches both Denton and Dallas), it selects the largest by total_rd.

**Example before/after**:

Before context awareness:
```sql
-- Question: "Show me which Texas schools grew faster than UNT"
WHERE name LIKE '%North Texas%'  -- matches 3 campuses
-- No reference value, no state filter, no time window
```

After context awareness:
```sql
-- Same question, with context
WHERE inst_id = '003594'  -- exact match, one institution
AND i.state = 'TX'
AND i.year IN (2019, 2024)
-- Includes UNT's CAGR as reference column for comparison
```

---

## Security Notes

The Q&A tab sends user questions to Gemini, which generates SQL. Three safeguards prevent damage:

1. `execute_sql` rejects any query that doesn't start with `SELECT` or `WITH`.
2. The SQLite connection opens in read-only mode (`file:{path}?mode=ro`).
3. Rate limiter: 50 queries per hour per session.

**Authentication**: Institution-based login. Users select their institution from a dropdown and enter a shared password. The institution name is logged for usage tracking. Access control is enforced via `ALLOWED_INSTITUTIONS` environment variable when configured.

The `PASSWORD` environment variable has no default fallback. The app will not start without it.


---

## Known Limitations

- Data is limited to institutions that report to the NSF HERD Survey.
- AI-generated insights and SQL should be reviewed – Gemini can produce incorrect queries.
- PDF export is disabled (the weasyprint dependency caused deployment issues). Use browser Print to PDF.
- The benchmarker cache does not auto-refresh when new data is loaded. Restart the app.
- Session state holds up to 20 query result DataFrames in memory per user.
- The anchor view chart gets visually compressed for extreme outliers like Johns Hopkins (#1 at $4.1B) vs the median institution at $25M.
- Institution names are inconsistent across years (259 institutions changed names). The benchmarker handles this via inst_id joins, but casual users may be confused when they search for "University of Arizona" and see "The University of Arizona" in results.

---

## Tech Stack

- **Python 3.11+**
- **Streamlit 1.29+** – UI framework
- **SQLite** – database (single 40.5 MB file, no server)
- **Google Gemini 2.5 Flash** – SQL generation, summaries, strategic insights
- **scikit-learn** – KNN peer matching (NearestNeighbors, StandardScaler)
- **Plotly** – interactive charts
- **pandas** – data manipulation
- **NumPy** – array operations
- **gspread** – optional Google Sheets logging

---

## License

Proprietary. All rights reserved.
