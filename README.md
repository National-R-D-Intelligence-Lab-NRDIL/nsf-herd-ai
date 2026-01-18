# NSF HERD AI Assistant

AI-powered research analytics platform for querying 15 years of university R&D funding data using natural language.

## Overview

This tool enables university administrators to analyze NSF HERD (Higher Education Research & Development) survey data through natural language queries. Built for strategic decision-making at research institutions.

**Dataset:**
- 10,084 records covering 1,004 institutions
- 15 years of data (2010-2024)
- $1+ trillion in R&D funding analyzed

## Features

- **Natural Language Queries** — Ask questions in plain English
- **AI-Powered SQL Generation** — Gemini converts questions to optimized SQL
- **Growth Rate Analysis** — Calculate and compare R&D growth across institutions and time periods
- **Peer Benchmarking** — Pre-configured peer institution lists for competitive analysis
- **Smart Name Matching** — Handles variations like "UT Austin", "Ohio State", "MIT"
- **Auto-Generated Visualizations** — Context-aware charts (line/bar) based on query results
- **AI Summaries** — One-line insights with anti-hallucination guardrails
- **Persistent Usage Logging** — Google Sheets integration for query tracking
- **Access Control** — Password-protected with username capture
- **Export** — Download results as CSV

## Quick Start

### Prerequisites
- Python 3.11+
- Google Gemini API key
- Google Cloud service account (for logging)

### Installation

```bash
# Clone repository
git clone https://github.com/National-R-D-Intelligence-Lab-NRDIL/nsf-herd-ai.git
cd nsf-herd-ai

# Install dependencies
pip install -r requirements.txt

# Configure environment variables (see Configuration section)

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
GOOGLE_SHEET_ID=your_google_sheet_id
GOOGLE_SHEETS_CREDS={"type": "service_account", ...}
```

### Streamlit Cloud Deployment

Add secrets in Streamlit Cloud dashboard under Settings → Secrets:

```toml
GEMINI_API_KEY = "your_gemini_api_key"
DATABASE_PATH = "data/herd.db"
PASSWORD = "your_app_password"
GOOGLE_SHEET_ID = "your_google_sheet_id"
GOOGLE_SHEETS_CREDS = '{"type": "service_account", ...}'
```

## Example Queries

### Growth Analysis
- "Show R&D growth from 2020 to 2024 for [institution]"
- "Which universities had the highest R&D growth from 2020 to 2024?"
- "What is the growth rate for [institution] over the last 5 years?"

### Peer Benchmarking
- "How does [institution] compare to its peers in 2024?"
- "Compare [institution] to peer institutions for 2024"

### Competitive Intelligence
- "Compare [institution A], [institution B], and [institution C] for 2024"
- "Show top 10 universities by R&D in [state] for 2024"
- "What is [institution]'s total R&D for 2024?"

### Funding Analysis
- "What percentage of [institution]'s 2024 funding is federal?"
- "Show [institution]'s institutional funding from 2020 to 2024"
- "Break down [institution]'s funding sources for 2024"

## Architecture

```
├── app.py                    # Streamlit web interface
├── src/
│   └── query_engine.py       # AI query engine (SQL generation, execution, summarization)
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
| name | TEXT | Full institution name |
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
- Password-protected application access
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
| Logging | Google Sheets API |
| Deployment | Streamlit Cloud |

## Data Source

National Science Foundation (NSF) Higher Education Research & Development (HERD) Survey

- Official source: https://ncses.nsf.gov/surveys/herd
- Updated annually
- Covers all U.S. research institutions receiving federal funding

## Changelog

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