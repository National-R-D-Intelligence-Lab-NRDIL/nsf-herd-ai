# NSF HERD AI Assistant

AI-powered research analytics platform for querying 15 years of university R&D funding data using natural language.

## 🎯 Overview

This tool enables university administrators to analyze NSF HERD (Higher Education Research & Development) survey data through natural language queries. Built for strategic decision-making at research institutions.

**Dataset:**
- 📊 10,084 records
- 🏫 1,004 institutions
- 📅 15 years (2010-2024)
- 💰 $1+ trillion in R&D funding analyzed

## ✨ Features

- **Natural Language Queries** - Ask questions in plain English
- **AI-Powered SQL Generation** - Google Gemini converts questions to SQL
- **Auto-Generated Visualizations** - Smart charts (line/bar) based on data
- **AI Summaries** - One-line insights with anti-hallucination guardrails
- **Access Control** - Password-protected with usage logging
- **Export** - Download results as CSV

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Gemini API key

### Installation
```bash
# Clone repository
git clone https://github.com/Kalyan8358/nsf-herd-ai.git
cd nsf-herd-ai

# Install dependencies
pip install -r requirements.txt

# Setup environment variables
# Create .env file with:
# GEMINI_API_KEY=your_key_here
# DATABASE_PATH=data/herd.db

# Run application
streamlit run app.py
```

### First Login
- Default password: `unt2026`
- Change in `app.py` check_login() function

## 📊 Example Questions

**Growth Analysis:**
- "Show UNT's R&D growth from 2020 to 2024"
- "Which Texas universities had highest growth 2020-2023?"

**Competitive Intelligence:**
- "Compare UNT Denton, Texas Tech, and Houston's total R&D for 2024"
- "Show top 10 Texas universities by R&D in 2024"

**Strategic Metrics:**
- "What percentage of UNT's 2024 funding is federal?"
- "Show UNT Denton's institutional funding from 2020 to 2024"

## 🏗️ Architecture
```
├── app.py                    # Streamlit web interface
├── src/
│   └── query_engine.py      # AI query logic (Gemini integration)
├── data/
│   └── herd.db              # SQLite database (1.47 MB)
├── requirements.txt          # Python dependencies
└── .env                     # Environment variables (not in repo)
```

## 🔒 Security

- ✅ API keys in environment variables
- ✅ `.gitignore` protects secrets
- ✅ Password-protected access
- ✅ Usage logging with usernames
- ✅ Pinned dependencies

## 🛠️ Technology Stack

- **Frontend:** Streamlit
- **AI:** Google Gemini 2.5 Flash
- **Database:** SQLite
- **Visualization:** Plotly
- **Deployment:** Streamlit Cloud

## 📝 Data Source

National Science Foundation (NSF) Higher Education Research & Development (HERD) Survey
- Public data: https://ncses.nsf.gov/surveys/herd
- Updated annually
- Covers all U.S. research institutions

## 📊 Usage Logs

Access logs stored in `usage_log.txt` (not tracked in Git)

Format: `timestamp - User: username | Question: query`
