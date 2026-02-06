# NSF HERD Research Intelligence

Analytics platform for university R&D funding data.

## Overview

Analyze NSF HERD (Higher Education Research & Development) survey data through natural language queries and interactive snapshots.

**Dataset:** 10,084 records covering 1,004 institutions from 2010-2024

## Features

### Institution Snapshot
- **National ranking trends** - See how institutions move up/down over 5 or 10 years
- **Anchor positioning** - Compare against benchmark institutions (#1, #10, #50, etc.)
- **Funding metrics** - Current rank, total R&D, year-over-year movement

### Natural Language Queries
- **AI-powered SQL generation** - Gemini converts questions to optimized queries
- **Smart name matching** - Handles "MIT", "Ohio State", "UT Austin" variations
- **Auto-generated visualizations** - Context-aware charts based on results
- **AI summaries** - One-line insights with anti-hallucination guardrails

### Access Control
- Username + password authentication
- Approved user list for pilot programs
- Usage logging to Google Sheets

## Quick Start

### Prerequisites
- Python 3.11+
- Google Gemini API key
- Google Cloud service account (optional, for logging)

### Installation

```bash
# Clone repository
git clone https://github.com/National-R-D-Intelligence-Lab-NRDIL/nsf-herd-ai.git
cd nsf-herd-ai

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env with your values

# Run application
streamlit run app.py
```

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_gemini_api_key
DATABASE_PATH=data/herd.db
PASSWORD=your_app_password
APPROVED_USERS=user1,user2,user3
GOOGLE_SHEET_ID=your_google_sheet_id
GOOGLE_SHEETS_CREDS={"type": "service_account", ...}
```

### Streamlit Cloud Deployment

Add secrets in Streamlit Cloud dashboard under Settings → Secrets:

```toml
GEMINI_API_KEY = "your_gemini_api_key"
DATABASE_PATH = "data/herd.db"
PASSWORD = "your_app_password"
APPROVED_USERS = "user1,user2,user3"
GOOGLE_SHEET_ID = "your_google_sheet_id"
GOOGLE_SHEETS_CREDS = '{"type": "service_account", ...}'
```

## Example Use Cases

### Institution Snapshot
- Track UNT's national ranking movement from #198 (2019) → #174 (2024)
- Compare position against peer institutions (#1, #100, #250)
- Identify funding source breakdowns

### Natural Language Queries
- "Which universities had the fastest R&D growth over the last 5 years?"
- "What percentage of Ohio State's 2024 funding is federal?"
- "Show top 10 universities in Texas by R&D for 2024"
- "Compare MIT, Stanford, and Caltech from 2020 to 2024"
- "How has UCLA's federal funding changed from 2015 to 2024?"

## Data Pipeline

The database is built from raw NSF survey data through a three-step process. You don't need to run this unless you want to update the data with the latest NSF release.

### How It Works

**Step 1: Download** - Grabs the ZIP files from the NSF website  
**Step 2: Transform** - Converts the data into a format that's easy to query  
**Step 3: Load** - Creates the database file

```bash
python scripts/etl/1_download.py
python scripts/etl/2_transform.py
python scripts/etl/3_load.py
```

The transform step is smart - if NSF adds new funding categories or changes their format, the code adapts automatically. No manual updates needed.

**Name cleaning:** The transform step standardizes institution names by moving trailing ", The" to the front (e.g., "University of Alabama, The" → "The University of Alabama").

### Updating the Data

NSF releases new data every October. When that happens:

1. Run the three scripts above
2. The new data gets added automatically
3. Push the updated database to GitHub

That's it. The existing application will work with the updated data without any code changes.

## Architecture

```
├── app.py                    # Main UI (snapshot + Q&A tabs, auth, visualization)
├── src/
│   └── query_engine.py       # AI query engine + snapshot data methods
├── scripts/
│   └── etl/                  # Data pipeline scripts
│       ├── 1_download.py     # Download NSF data
│       ├── 2_transform.py    # Transform + name cleaning
│       └── 3_load.py         # Load into database
├── data/
│   └── herd.db               # SQLite database
├── requirements.txt          # Pinned dependencies
├── .env                      # Local environment variables (not in repo)
└── .gitignore                # Git ignore rules
```

## Database Schema

**Table: institutions**

| Column | Type | Description |
|--------|------|-------------|
| inst_id | TEXT | Institution identifier (e.g., '003594') |
| name | TEXT | Full institution name (cleaned) |
| city | TEXT | City location |
| state | TEXT | State code (e.g., 'TX') |
| year | INTEGER | Fiscal year (2010-2024) |
| total_rd | INTEGER | Total R&D expenditure ($) |
| federal | INTEGER | Federal funding ($) |
| state_local | INTEGER | State and local funding ($) |
| business | INTEGER | Business funding ($) |
| nonprofit | INTEGER | Nonprofit funding ($) |
| institutional | INTEGER | Institutional funding ($) |
| other_sources | INTEGER | Other funding sources ($) |

## Security

- API keys stored in environment variables / Streamlit secrets
- `.gitignore` configured to exclude sensitive files
- Password + approved user list for access control
- Usage logging with usernames for audit trail
- Pinned dependencies to prevent supply chain issues

## Usage Logging

All queries are logged to a Google Sheet with:
- Timestamp
- Username
- Question asked
- Generated SQL

Logs persist across application restarts and deployments.

## Technology Stack

| Component | Technology |
|-----------|------------|
| Frontend | Streamlit |
| AI Model | Google Gemini 2.5 Flash |
| Database | SQLite |
| Visualization | Plotly |
| Auth | Session-based |
| Logging | Google Sheets API |
| Deployment | Streamlit Cloud / Railway |

## Monitoring
Uptime monitoring via UptimeRobot - pings every 5 minutes to prevent Streamlit Cloud sleep.

## Data Source

National Science Foundation (NSF) Higher Education Research & Development (HERD) Survey

- Official source: https://ncses.nsf.gov/surveys/herd
- Updated annually
- Covers all U.S. research institutions receiving federal funding

## Changelog

### v2.0.0 (February 2026)
- Added Institution Snapshot feature with ranking trends and anchor positioning
- Implemented authentication with approved user list
- Added institution name standardization in ETL pipeline
- Updated UI to tabbed interface (Snapshot + Q&A)
- Improved query engine with dedicated snapshot data methods

### v1.1.0 (January 2026)
- Added growth rate calculation support
- Added peer institution benchmarking (Texas and National peers)
- Improved institution name matching (handles abbreviations and variations)
- Added persistent logging via Google Sheets
- Migrated repository to organization account

### v1.0.0 (January 2026)
- Initial release
- Natural language to SQL conversion
- Auto-generated visualizations
- AI-powered summaries
- Password authentication
- CSV export

## Contributing

This project is maintained by the National R&D Intelligence Lab (NRDIL). For questions or contributions, please open an issue in the repository.

## License

Internal use only. Data sourced from publicly available NSF HERD survey.