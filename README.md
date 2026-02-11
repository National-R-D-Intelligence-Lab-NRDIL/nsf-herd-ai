# NSF HERD Research Intelligence Platform

Analytics dashboard for university R&D funding data. Takes the NSF HERD Survey (1,004 institutions, 2010-2024) and turns it into strategic intelligence for VPRs and research administrators.

Two main features: an **Institution Snapshot** that generates a full competitive analysis for any institution, and a **natural language Q&A** interface that converts plain English questions into SQL queries against the HERD database.

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
```

Run it:

```bash
streamlit run app.py
```

Open `http://localhost:8501` and log in with any username and the password you set.

You need a Gemini API key from https://ai.google.dev/ and the HERD database file at `data/herd.db`. See the ETL section below if you need to build the database from scratch.

---

## What It Does

### Institution Snapshot

Select any institution from the dropdown and get:

**Executive summary** -- current national rank, total R&D, 5-year growth rate, and an AI-generated one-sentence strategic insight. A traffic light status compares your growth against your peer average.

**Rank trend** -- horizontal bar chart showing national rank for each year in the selected window (5-year or 10-year). Shows net rank movement.

**National anchor view** -- places the institution on a bar chart alongside benchmark positions (#1, #50, #100, #250, etc.) so you can see where you sit relative to the full landscape.

**KNN peer analysis** -- finds the 10 most similar institutions nationally using K-Nearest Neighbors across 7 funding dimensions (total R&D, federal, state/local, business, nonprofit, institutional, other). Shows a funding profile comparison, growth trajectories over time, and gap analysis against peer averages. This replaces the old resource parity method (+-20% of total R&D) which failed for outliers.

**Funding source breakdown** -- pie chart of current funding sources, plus a federal dependency trend line plotted against the national median. Flags high/moderate/low federal dependency risk.

**State competitive position** -- your state rank and market share, a competitive band showing the 3 institutions above and below you with federal dependency percentages, and a top-10 state leaderboard in an expander.

### Q&A Interface

Type a question in plain English. The app sends it to Gemini along with the database schema, Gemini generates SQL, the app executes it (read-only), and returns the results as a table with an auto-generated chart and a 2-3 sentence AI summary.

Example questions:
- "What is Harvard's total R&D for 2024?"
- "Show top 10 universities by R&D funding in 2024"
- "Compare MIT, Stanford, and Caltech from 2020 to 2024"
- "Which universities had the fastest R&D growth over the last 5 years?"
- "What percentage of Ohio State's 2024 funding is federal?"

Results are downloadable as CSV. Query history is kept for the session (last 20 queries).

---

## Project Structure

```
nsf-herd-mvp/
├── app.py                     # Streamlit app (UI, charts, routing)
├── src/
│   ├── query_engine.py        # SQL generation, database queries, AI summaries
│   └── benchmarker.py         # KNN peer-finding algorithm
├── scripts/
│   └── etl/
│       ├── 1_download.py      # Fetch raw HERD data from NSF
│       ├── 2_transform.py     # Clean and standardize
│       └── 3_load.py          # Load into SQLite
├── data/
│   └── herd.db                # SQLite database (not in repo)
├── Configs/
│   ├── template.yml           # Config template
│   └── unt_internal.yml       # UNT-specific config (gitignored)
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

`app.py` handles all UI rendering and calls into two backend modules. `query_engine.py` owns every database query -- rank trends, anchor views, funding breakdowns, state rankings, peer comparisons, and the full LLM-to-SQL pipeline for the Q&A tab. `benchmarker.py` handles the KNN peer-finding algorithm. It loads the latest year of funding data, log-transforms and scales it, fits a NearestNeighbors model, and exposes methods to find peers, analyze gaps, and pull historical trends.

The benchmarker is fitted once at app startup and cached via `@st.cache_resource`. The institution list is cached with a 1-hour TTL via `@st.cache_data`.

---

## Database Schema

Single table, one row per institution per year:

```
Table: institutions
├── inst_id         TEXT       -- NSF institution identifier (e.g. '003594')
├── name            TEXT       -- Full name as reported by NSF
├── city            TEXT
├── state           TEXT       -- Two-letter code (e.g. 'TX')
├── year            INTEGER    -- Fiscal year (2010-2024)
├── total_rd        INTEGER    -- Total R&D expenditure in dollars
├── federal         INTEGER    -- Federal funding
├── state_local     INTEGER    -- State and local government funding
├── business        INTEGER    -- Business/industry funding
├── nonprofit       INTEGER    -- Nonprofit funding
├── institutional   INTEGER    -- Institution's own funding
└── other_sources   INTEGER    -- All other sources
```

10,084 rows. 1,004 unique institutions. `total_rd` = sum of the six funding source columns (verified, zero mismatches). Range: Carthage College at $188K to Johns Hopkins at $4.1B.

Important: 259 institutions changed names over time (e.g. "University of Arizona" became "The University of Arizona" in 2013). The benchmarker joins on `inst_id` to avoid data loss. The other query engine methods join on `name`, which is safe because they operate within single years.

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GEMINI_API_KEY` | Yes | -- | Google Gemini API key |
| `PASSWORD` | Yes | -- | Login password. App refuses to start without it. |
| `DATABASE_PATH` | No | `data/herd.db` | Path to SQLite database |
| `GOOGLE_SHEET_ID` | No | -- | Google Sheet ID for usage logging |
| `GOOGLE_SHEETS_CREDS` | No | -- | Service account credentials JSON string |

---

## Deployment

### Railway

The repo includes a `Procfile`. Set your environment variables in the Railway dashboard and deploy.

### Streamlit Cloud

Push to GitHub, connect at share.streamlit.io, add `GEMINI_API_KEY` and `PASSWORD` as secrets.

### GitHub Codespaces

The `.devcontainer/devcontainer.json` is pre-configured. Open the repo in Codespaces and it installs dependencies and starts the app automatically on port 8501.

---

## ETL Pipeline

When NSF releases new HERD data (annually):

```bash
cd scripts/etl
python 1_download.py      # Fetches raw data from NSF
python 2_transform.py     # Cleans names, standardizes columns
python 3_load.py          # Loads into data/herd.db
```

After loading new data, restart the app to refresh the benchmarker cache. The institution list and year ranges are derived from the database automatically -- no code changes needed.

---

## How the KNN Peer Algorithm Works

The old approach (resource parity, +-20% of total R&D) failed for outliers. Johns Hopkins at $4.1B has zero institutions within 20% of its total R&D. Carthage College at $188K has one.

The KNN approach:

1. Pull one row per institution from the latest survey year (681 institutions).
2. Log-transform all 7 numeric columns with `np.log1p`. This compresses the 22,000x skew so $1M vs $2M gets similar weight to $1B vs $2B.
3. Standardize to zero-mean, unit-variance with `StandardScaler`.
4. Fit a `NearestNeighbors` model (euclidean distance) in that space.
5. For any institution, return the k closest neighbors.

`total_rd` is included alongside the 6 funding source columns even though it's their sum. This is intentional -- it gives overall size 2x weight in the distance calculation, which prevents a $300M school from matching to a $3B school just because they have similar funding ratios.

Peer matching is not symmetric. If UNT lists Texas State as a peer, Texas State may not list UNT back. This is normal for KNN and matches how other benchmarking tools (IPEDS, Delaware Study) work.

---

## Security Notes

The Q&A tab sends user questions to Gemini, which generates SQL. Two safeguards prevent damage:
1. `execute_sql` rejects any query that doesn't start with `SELECT` or `WITH`.
2. The SQLite connection opens in read-only mode (`file:{path}?mode=ro`).

Authentication is username + password. The username is used for logging (who asked what query), not for access control. There is a rate limiter of 50 queries per hour per session on the Q&A tab.

The `PASSWORD` environment variable has no default fallback. The app will not start without it.

---

## Known Limitations

- Data is limited to institutions that report to the NSF HERD Survey.
- AI-generated insights and SQL should be reviewed -- Gemini can produce incorrect queries.
- PDF export is disabled (the weasyprint dependency caused issues). Use browser Print to PDF.
- The benchmarker cache does not auto-refresh when new data is loaded. Restart the app.
- Session state holds up to 20 query result DataFrames in memory per user.
- The anchor view chart gets visually compressed for extreme outliers like Johns Hopkins.

---

## Tech Stack

- **Python 3.11+**
- **Streamlit 1.29** -- UI framework
- **SQLite** -- database (single file, no server)
- **Google Gemini 2.5 Flash** -- SQL generation, summaries, strategic insights
- **scikit-learn** -- KNN peer matching
- **Plotly** -- interactive charts
- **Matplotlib** -- static chart generation (PDF reports)
- **gspread** -- optional Google Sheets logging

---

## License

Proprietary. All rights reserved.