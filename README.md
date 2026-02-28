# NSF HERD Research Intelligence Platform

Strategic analytics dashboard for university R&D funding data. Transforms the NSF HERD Survey (1,004 institutions, 2010–2024) into competitive intelligence for Vice Presidents of Research and research administrators.

**Five core capabilities**: Landing Intelligence Briefing (instant KPI snapshot on load), Institution Snapshot (competitive positioning, peer analysis, state comparison), Research Portfolio (field-level analysis), Federal Landscape (agency funding analysis), and natural language Q&A (SQL-powered research queries).

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
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your_anon_key_here
BASE_URL=http://localhost:8501   # used for shareable permalink generation
```

Run it:

```bash
streamlit run app.py
```

Open `http://localhost:8501`. Click **Create Account** and register with your email. Institution is stored in your profile and pre-selects the dashboard on every login.

You need a Gemini API key from https://ai.google.dev/ and the HERD database file at `data/herd.db`. See the ETL section below if you need to build the database from scratch.

---

## What It Does

### Landing Intelligence Briefing (above tabs)

Displays immediately on institution select — no clicking required:

- **National Rank** — current rank out of 1,004 institutions with year-over-year movement indicator
- **Total R&D** — current year expenditure with 5-year CAGR
- **CAGR vs Benchmark Peers** — your compound growth rate delta against your 10 AI-matched peers
- **Actionable Callout** — one plain-English insight auto-selected by priority: growth rank → rank movement → field signal. Rule-based, no LLM.

### Tab 1: Institution Snapshot

Select any institution and time window (5-year, 10-year, or custom range) to get:

**Strategic Summary** — Current national rank, total R&D, growth rate, largest research field, and top federal agency. Compares your growth against your peer average with an AI-generated strategic insight that uses positioning language (never risk labels).

**Ranking Over Time** — Horizontal bar chart showing national rank for each year in the selected window. Shows net rank movement and total institutions in the survey.

**Where You Sit Nationally** — Anchor view placing your institution on a bar chart alongside benchmark positions (#1, #10, #50, #100, #250, etc.) so you can see where you sit relative to the full landscape. Your institution is highlighted in blue.

**Peer Analysis** — Uses AI matching (K-Nearest Neighbors on log-transformed funding data, k=10) to find your most similar institutions nationally across 7 funding dimensions. Custom peer groups can be set via the **Custom Peer Selection** expander:

- Pick any combination of institutions from the dropdown
- Click **Apply** to update the dashboard (picks are staged — the dashboard does not change until Apply is clicked)
- Custom peers are saved to your account per institution and restored on your next visit
- Click **Clear** to revert to AI-recommended Benchmark Peers
- When custom peers are active, a sidebar banner shows the count of selected peers
- The Q&A tab shows a warning banner listing your custom peer names

Peer analysis shows growth metrics (your CAGR, peer average CAGR, your growth rank) plus two sub-tabs:

- *Funding Profile*: Grouped bar chart comparing your funding mix vs peer averages, with a detailed gap numbers expander and a peer list expander showing full dollar amounts.
- *Growth Over Time*: Two view modes — **Summary** (your institution + shaded peer min/max band + peer average line) and **Detail** (all individual peer lines with interactive legend).

**Funding Source Analysis** — Pie chart of current funding sources plus a federal share trend line plotted against the national median.

**State Competitive Position** — Collapsed by default. Your state rank and market share. Shows your competitive band (3 institutions above and 3 below you in state rank with federal dependency percentages). Top 10 state leaderboard in an expander.

**State Peer Comparison** — AI-matched peers filtered to your state (drawn from the top 100 national neighbors, filtered to same state). Shows your rank within the state peer group and CAGR vs state peer average, a grouped bar chart comparing your funding mix to portfolio-similar in-state institutions, and an R&D growth trend vs the top institutions in your state.

### Tab 2: Research Portfolio

Field-level R&D analysis across 10 parent research fields and 36 sub-fields:

**Portfolio Overview** — Stacked horizontal bar chart showing federal vs nonfederal funding for each of your 10 parent fields. Shows dollar amounts and portfolio share percentages.

**Field Momentum** — BCG-style scatter plot with portfolio share (x-axis) vs 5-year CAGR (y-axis). Each bubble is a research field. Quadrant labels: Core Strengths, Emerging, Established Base, Smaller Base.

**Sub-field Drill-Down** — Expandable view for each parent field (Engineering, Life Sciences, Physical Sciences, etc.) showing its component disciplines with federal percentages and share of parent.

**Portfolio Distinctiveness** — Diverging bar chart comparing your field mix to your 10 nearest peers. Positive bars = you invest more in this field than peers do. Shows where your research portfolio is distinctive.

### Tab 3: Federal Landscape

Agency-level federal funding analysis across 7 agencies (DOD, DOE, HHS, NASA, NSF, USDA, Other):

**Federal Agency Breakdown** — Donut chart and data table showing your current agency distribution and each agency's share of your federal funding. Full dollar amounts shown.

**Funding Diversification** — Three metrics:
- Diversification score (0–100%, based on Herfindahl-Hirschman Index)
- Top agency name and percentage
- National percentile (higher = more concentrated)

**Agency Funding Trends** — Multi-line chart showing funding from each agency over the selected time window. Growth summary table (start year, end year, % change) in an expander.

**Agency Distinctiveness** — Diverging bar chart comparing your agency mix to your 10 nearest peers. Shows which agencies you rely on more/less than similar institutions.

### Tab 4: Ask a Question

Natural language Q&A interface powered by Gemini AI. Designed for targeted exploration beyond what the dashboard tabs show — discipline-specific rankings, cross-table queries, and custom filters.

**Context-aware SQL generation** — Automatically uses your selected institution's inst_id, state, time window, and peer list when interpreting questions like "How do we compare?" or "Show me our peers." When custom peers are active, the Q&A uses those peer IDs and displays a warning banner naming the active peers.

**Competitive band ranking** — Comparative questions produce a competitive neighborhood view: your rank in the specific metric asked about, with ~8 institutions above and ~7 below. Your institution is highlighted in results.

**Field/agency code validation** — After SQL generation, the system checks field_code and agency_code values against valid sets (46 field codes, 7 agency codes). Invalid codes trigger an automatic retry with the error and correct alternatives.

**Suggested questions** — Three expandable categories:
- How do we compare? (discipline rankings, state comparisons, growth rates)
- Where's the momentum? (fastest growing fields, agency trends, funding shifts)
- What's distinctive? (portfolio differences, field concentration, federal positioning)

**Query execution** — Returns results as a data table with generated SQL (expandable), selected institution highlighted in blue, auto-generated chart, AI-generated 2–3 sentence summary, and CSV download.

**Example questions**:
- "Where do we rank in engineering among similar institutions?"
- "How does our life sciences R&D compare to other Texas schools?"
- "What are our fastest growing sub-fields?"
- "Which agencies increased their funding to us the most?"
- "Which Texas schools get more NSF funding than us but have lower total R&D?"

Rate limited to 50 queries per hour per session.

### Shareable Permalinks

Every institution view has a persistent URL: `?inst=University+of+North+Texas&years=5`. The URL updates automatically as you change institution or time window. A **Share this view** link is shown in the sidebar — copy it to bookmark or send to a colleague.

### PNG Export

Every chart has a camera icon (top-right of chart) that downloads a 2x-resolution PNG — no server-side dependencies, fully client-side.

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

`app.py` handles all UI rendering and calls into two backend modules. `query_engine.py` owns every database query — rank trends, anchor views, funding breakdowns, state rankings, peer comparisons, field portfolio analysis, agency breakdowns, concentration metrics, and the full LLM-to-SQL pipeline for the Q&A tab. `benchmarker.py` handles the KNN peer-finding algorithm. It loads the latest year of funding data, log-transforms and scales it, fits a NearestNeighbors model, and exposes methods to find peers, analyze gaps, and pull historical trends.

The benchmarker is fitted once at app startup and cached via `@st.cache_resource`. The institution list is cached with a 1-hour TTL via `@st.cache_data`.

---

## Database Schema

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

Four parents (CS, Math, Psychology, Other Sciences) are standalone — the HERD survey doesn't break them into sub-disciplines.

### The 7 Federal Agencies

- **DOD** — Dept of Defense
- **DOE** — Dept of Energy
- **HHS** — HHS (incl. NIH). Note: NIH is not broken out separately in the HERD survey.
- **NASA** — NASA
- **NSF** — NSF
- **USDA** — USDA
- **Other agencies** — Other Federal

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GEMINI_API_KEY` | Yes | — | Google Gemini API key for SQL generation and summaries |
| `SUPABASE_URL` | Yes | — | Supabase project URL. Enables per-user auth, usage logging, and peer preference persistence. |
| `SUPABASE_ANON_KEY` | Yes | — | Supabase anon/public key. |
| `DATABASE_PATH` | No | `data/herd.db` | Path to SQLite database |
| `BASE_URL` | No | `http://localhost:8501` | Base URL for shareable permalink generation (set to your Railway/Streamlit Cloud domain in production) |
| `GOOGLE_SHEET_ID` | No | — | Google Sheet ID for Q&A usage logging (legacy, optional) |
| `GOOGLE_SHEETS_CREDS` | No | — | Service account credentials JSON string (legacy, optional) |

---

## Deployment

### Railway

The repo includes a `Procfile`. Set the following environment variables in the Railway dashboard:

```
GEMINI_API_KEY
DATABASE_PATH
SUPABASE_URL
SUPABASE_ANON_KEY
BASE_URL    # set to your Railway public domain, e.g. https://your-app.up.railway.app
```

### Streamlit Cloud

Push to GitHub, connect at share.streamlit.io, add `GEMINI_API_KEY`, `SUPABASE_URL`, and `SUPABASE_ANON_KEY` as secrets. Set `BASE_URL` to your Streamlit Cloud app URL.

### GitHub Codespaces

The `.devcontainer/devcontainer.json` is pre-configured. Open the repo in Codespaces and it installs dependencies and starts the app automatically on port 8501.

---

## Supabase Setup

1. Create a project at supabase.com
2. Disable email confirmation: Auth → Settings → "Enable email confirmations" → OFF
3. Create the `usage_log` table in the SQL editor:

```sql
CREATE TABLE usage_log (
    id          uuid DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id     uuid,
    email       text,
    full_name   text,
    event_type  text,
    institution text,
    question    text,
    created_at  timestamptz DEFAULT now()
);
ALTER TABLE usage_log DISABLE ROW LEVEL SECURITY;
```

4. Create the `user_preferences` table for custom peer persistence:

```sql
CREATE TABLE user_preferences (
    id           uuid DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id      uuid NOT NULL,
    institution  text NOT NULL,
    custom_peers jsonb DEFAULT '[]',
    updated_at   timestamptz DEFAULT now(),
    UNIQUE (user_id, institution)
);
ALTER TABLE user_preferences DISABLE ROW LEVEL SECURITY;
```

5. Add `SUPABASE_URL` and `SUPABASE_ANON_KEY` to your environment

Events logged to `usage_log`: `login`, `signup`, `institution_view` (deduplicated per session), `qa_question`.

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

After loading new data, restart the app to refresh the benchmarker cache. The institution list and year ranges are derived from the database automatically — no code changes needed unless NSF restructures a question.

**What future NSF changes would require**:
- New survey year, same structure → Just re-run ETL pipeline
- New sub-field added → Add entry to `FIELD_TAXONOMY` in `4_transform_fields.py`
- New parent field added → Add to `FIELD_TAXONOMY`, update UI tab
- New agency column → Add to `AGENCIES` in `5_transform_agencies.py`
- New question added → New table + new ETL script

---

## How the Benchmark Peer Algorithm Works

The old approach (resource parity, ±20% of total R&D) failed for outliers. Johns Hopkins at $4.1B has zero institutions within 20% of its total R&D. Carthage College at $188K has one.

The AI matching approach (KNN):

1. Pull one row per institution from the latest survey year (681 institutions with complete data).
2. Log-transform all 7 numeric columns with `np.log1p`. This compresses the 22,000x skew so $1M vs $2M gets similar weight to $1B vs $2B.
3. Standardize to zero-mean, unit-variance with `StandardScaler`.
4. Fit a `NearestNeighbors` model (euclidean distance) in that space.
5. For any institution, query k=11 neighbors (k+1 to account for self-match), exclude the target, return 10 peers.

`total_rd` is included alongside the 6 funding source columns even though it's their sum. This is intentional — it gives overall size 2x weight in the distance calculation, which prevents a $300M school from matching to a $3B school just because they have similar funding ratios.

**Important notes**:
- Peer matching is not symmetric. If UNT lists Texas State as a peer, Texas State may not list UNT back. This is normal and matches how other benchmarking tools (IPEDS, Delaware Study) work.
- The benchmarker fits on the 681 institutions in the latest year with complete data, while the institution dropdown shows institutions with 5+ consecutive years (612 institutions). The system handles the overlap gracefully.
- For historical trends, the benchmarker joins on `inst_id` (not name) because 259 institutions changed names over the 15-year survey period.
- Custom peer groups can override AI-recommended peers per institution — selections are persisted to Supabase and restored on next login.

---

## Q&A Tab: Context-Aware SQL Generation

The Ask a Question tab uses Gemini AI to convert natural language to SQL.

**How context flows**:
1. User selects institution in the top-level picker
2. `app.py` resolves inst_id, state, start_year, end_year, peer_inst_ids, and peer_mode_type (benchmark or custom)
3. Stores them in session state; custom peer mode shows a warning banner in the Q&A tab
4. User types a question
5. `process_question()` builds a context dict and passes it to `engine.ask()`
6. `generate_sql()` injects context into the prompt, including peer group type (AI-recommended vs custom) and human-readable peer names
7. LLM generates SQL using inst_id, not name LIKE patterns
8. `_validate_codes()` checks field_code/agency_code values deterministically
9. Invalid code → retry with error + valid alternatives
10. 0 rows → `_validate_and_retry()` sends failed SQL back for correction
11. Results → highlight `is_selected` row → `create_visualization()` → chart

**Competitive band ranking**: For comparative questions, the LLM ranks all institutions by the specific metric in the question, finds the selected institution's position, and shows ~8 above and ~7 below. The selected institution appears highlighted. This replaces the naive approach of showing all institutions nationally.

**Field/agency code validation**: `_validate_codes()` extracts all `field_code='...'` and `agency_code='...'` values from the SQL and checks them against hardcoded valid sets (46 field codes, 7 agency codes). Common aliases mapped in the schema prompt: "NIH" → HHS, "BME" → eng_biomedical, "defense" → DOD.

**Growth query guardrails**: Always include start/end dollar amounts alongside CAGR; filter out institutions with less than $1M in the starting year; never compute CAGR before 2016 for the four sub-fields added that year.

**Name resolution**: 45+ common abbreviations (MIT, Caltech, UNT, TAMU, etc.) map to full NSF names. When multiple campuses match, selects the largest by total_rd.

---

## Security Notes

The Q&A tab sends user questions to Gemini, which generates SQL. Four safeguards prevent damage:

1. `_validate_codes()` checks field_code and agency_code values before execution.
2. `execute_sql` rejects any query that doesn't start with `SELECT` or `WITH`.
3. The SQLite connection opens in read-only mode (`file:{path}?mode=ro`).
4. Rate limiter: 50 queries per hour per session.

**Authentication**: Per-user email + password via Supabase. Users register once with their name, institution, and email — institution is stored in Supabase user metadata and pre-selects the dashboard on every login. If `SUPABASE_URL` is not set, the app shows an error and stops — there is no fallback login.

**Usage logging**: All login, signup, institution view, and Q&A question events are logged to `usage_log` in Supabase for named-user adoption tracking.

---

## Known Limitations

- Data is limited to institutions that report to the NSF HERD Survey. Data updates annually with ~18 month lag (2024 data available in early 2026).
- AI-generated SQL should be reviewed — Gemini can produce incorrect queries. Code validation catches invalid field/agency codes but cannot verify semantic correctness.
- Four sub-fields (eng_industrial, life_natural_resources, phys_materials, soc_anthropology) have no data before 2016. Custom year ranges starting before 2016 display a caveat.
- The benchmarker cache does not auto-refresh when new data is loaded. Restart the app.
- Session state holds up to 20 query result DataFrames in memory per user. Streamlit scales memory linearly with concurrent users (~100–150MB per session).
- The anchor view chart gets visually compressed for extreme outliers like Johns Hopkins (#1 at $4.1B) vs the median institution at $25M.
- Institution names are inconsistent across years (259 institutions changed names). The benchmarker handles this via inst_id joins, but casual users may be confused when searching for "University of Arizona" vs "The University of Arizona" in results.
- Custom peer groups with no data overlap for a selected time window will produce empty peer charts. Choose peers that reported during your selected years.

---

## Tech Stack

- **Python 3.11+**
- **Streamlit >=1.30** — UI framework (stable `st.query_params` API required)
- **SQLite** — database (single 40.5 MB file, no server)
- **Google Gemini 2.5 Flash** — SQL generation, summaries, strategic insights
- **scikit-learn** — KNN peer matching (NearestNeighbors, StandardScaler)
- **Plotly** — interactive charts with client-side PNG export
- **pandas** — data manipulation
- **NumPy** — array operations
- **Supabase** — user auth (email + password), usage logging, custom peer persistence
- **gspread** — optional Google Sheets logging (legacy)

---

## License

Proprietary. All rights reserved.
