import streamlit as st
import sys
from pathlib import Path
from dotenv import load_dotenv
import os
import plotly.express as px
from datetime import datetime
import json
from zoneinfo import ZoneInfo
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Environment setup - works across Railway, Streamlit Cloud, and local
# Railway and Streamlit Cloud both expose secrets as env vars.
# Locally, we fall back to .env file via dotenv.
# ============================================================
load_dotenv()

def get_env(key, default=None):
    """Single place to pull secrets — env vars first, then .env fallback."""
    return os.getenv(key, default)

GEMINI_API_KEY = get_env('GEMINI_API_KEY')
DATABASE_PATH = get_env('DATABASE_PATH', 'data/herd.db')
PASSWORD = get_env('PASSWORD', 'demo2026')
GOOGLE_SHEET_ID = get_env('GOOGLE_SHEET_ID')
GOOGLE_SHEETS_CREDS = get_env('GOOGLE_SHEETS_CREDS')

if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY is not set. Add it to your environment variables.")
    st.stop()

# ============================================================
# Authentication
# ============================================================
def check_login():
    if st.session_state.get('logged_in'):
        return

    st.title("🔐 Login Required")
    username = st.text_input("Username", placeholder="Enter your name")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username and password == PASSWORD:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.rerun()
        else:
            st.error("Invalid credentials")
    st.stop()

check_login()

# ============================================================
# Google Sheets logging - fires and forgets.
# If credentials aren't configured, it just skips silently.
# ============================================================
def get_gsheet_client():
    if not GOOGLE_SHEETS_CREDS:
        return None
    try:
        import gspread
        from google.oauth2.service_account import Credentials

        scopes = [
            'https://www.googleapis.com/auth/spreadsheets',
            'https://www.googleapis.com/auth/drive'
        ]
        creds = Credentials.from_service_account_info(
            json.loads(GOOGLE_SHEETS_CREDS), scopes=scopes
        )
        return gspread.authorize(creds)
    except Exception:
        return None

def log_to_sheets(username, question, sql):
    if not GOOGLE_SHEET_ID:
        return
    try:
        client = get_gsheet_client()
        if not client:
            return
        sheet = client.open_by_key(GOOGLE_SHEET_ID).sheet1
        timestamp = datetime.now(ZoneInfo('America/Chicago')).strftime('%Y-%m-%d %H:%M:%S')
        sheet.append_row([timestamp, username, question, sql])
    except Exception:
        pass

# ============================================================
# Query engine setup
# ============================================================
sys.path.append(str(Path(__file__).parent / 'src'))
from query_engine import HERDQueryEngine

engine = HERDQueryEngine(GEMINI_API_KEY, DATABASE_PATH)

# ============================================================
# Session state
# ============================================================
if 'history' not in st.session_state:
    st.session_state.history = []

# ============================================================
# UI Layout
# ============================================================
st.title("NSF HERD Research Intelligence")
st.markdown("Ask questions about university R&D funding across 1,004 institutions (2010–2024)")

with st.sidebar:
    st.header("Visualization")
    enable_viz = st.checkbox("Auto-generate charts", value=True)
    chart_type = st.selectbox("Chart type", ["Auto", "Bar", "Line", "Scatter"])

    if st.button("Clear History"):
        st.session_state.history = []
        st.rerun()

# Example questions that showcase what the tool can actually do
with st.expander("Example Questions"):
    st.markdown("""
    - What is Harvard's total R&D for 2024?
    - Show top 10 universities by R&D funding in 2024
    - Compare MIT, Stanford, and Caltech from 2020 to 2024
    - Which universities had the fastest R&D growth over the last 5 years?
    - What percentage of Ohio State's 2024 funding is federal?
    - Show all Texas universities ranked by total R&D in 2024
    - How has UCLA's federal funding changed from 2015 to 2024?
    - Which states have the highest total R&D across all institutions?
    """)

# ============================================================
# Visualization logic
# Picks chart type based on the shape of the data and what the
# user actually asked about. Not perfect, but handles the common cases.
# ============================================================
def create_visualization(df, question):
    if df is None or df.empty:
        return None

    has_year = 'year' in df.columns
    has_name = 'name' in df.columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    if not numeric_cols:
        return None

    question_lower = question.lower()

    # Figure out which column is the most relevant y-axis
    if any(w in question_lower for w in ['growth', 'cagr', 'rate', 'fastest']):
        candidates = [c for c in numeric_cols if any(k in c.lower() for k in ['cagr', 'growth', 'pct'])]
        y_col = candidates[0] if candidates else numeric_cols[-1]

    elif any(w in question_lower for w in ['federal', 'institutional', 'business', 'nonprofit', 'funding source']):
        candidates = [c for c in numeric_cols if any(k in c.lower() for k in ['federal', 'institutional', 'business', 'state', 'nonprofit'])]
        y_col = candidates[0] if candidates else numeric_cols[0]

    elif any(w in question_lower for w in ['total', 'compare', 'top', 'rank']):
        candidates = [c for c in numeric_cols if 'total' in c.lower() or c == 'total_rd']
        y_col = candidates[-1] if candidates else numeric_cols[-1]

    else:
        y_col = numeric_cols[-1]

    # Time series → line chart. Multiple institutions over time get color-coded.
    if has_year and len(df) > 1 and df['year'].nunique() > 1:
        if has_name and df['name'].nunique() > 1:
            return px.line(df, x='year', y=y_col, color='name',
                           title=f"{y_col} over time", markers=True)
        return px.line(df, x='year', y=y_col,
                       title=f"{y_col} over time", markers=True)

    # Single snapshot in time with multiple institutions → bar chart
    if has_name and len(df) > 1:
        df_sorted = df.sort_values(by=y_col, ascending=False)
        fig = px.bar(df_sorted, x='name', y=y_col, title=f"{y_col} by institution")
        fig.update_xaxes(tickangle=-45)
        return fig

    return None

# ============================================================
# Conversation history - renders previous Q&A pairs
# ============================================================
for item in st.session_state.history:
    with st.chat_message("user"):
        st.write(item['question'])
    with st.chat_message("assistant"):
        with st.expander("Generated SQL"):
            st.code(item['sql'], language="sql")

        if item.get('results') is not None and len(item['results']) > 0:
            st.dataframe(item['results'], use_container_width=True)

        if item.get('summary'):
            st.info(f"📊 {item['summary']}")

        if enable_viz and item.get('results') is not None and len(item['results']) > 0:
            chart = create_visualization(item['results'], item['question'])
            if chart:
                st.plotly_chart(chart, use_container_width=True)

        if item.get('results') is not None and len(item['results']) > 0:
            csv_data = item['results'].to_csv(index=False)
            st.download_button(
                "Download CSV", csv_data,
                f"results_{item['question'][:20].replace(' ', '_')}.csv",
                "text/csv",
                key=f"download_{hash(item['question'])}"
            )

# ============================================================
# Main query input
# ============================================================
question = st.chat_input("Ask a question about university R&D funding...")

if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                sql, results, summary = engine.ask(question)
                log_to_sheets(st.session_state.username, question, sql)

                with st.expander("Generated SQL"):
                    st.code(sql, language="sql")

                if results is not None and len(results) > 0:
                    st.dataframe(results, use_container_width=True)
                    st.success("Query executed successfully")
                else:
                    st.warning("Query returned no results. Try rephrasing your question.")

                if summary:
                    st.info(f"📊 {summary}")

                if enable_viz and results is not None and len(results) > 0:
                    chart = create_visualization(results, question)
                    if chart:
                        st.plotly_chart(chart, use_container_width=True)

                # Keep last 20 exchanges in memory
                st.session_state.history.append({
                    'question': question,
                    'sql': sql,
                    'results': results,
                    'summary': summary
                })
                if len(st.session_state.history) > 20:
                    st.session_state.history = st.session_state.history[-20:]

                # CSV download for current result
                if results is not None and len(results) > 0:
                    csv_data = results.to_csv(index=False)
                    st.download_button(
                        "Download CSV", csv_data,
                        f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        "text/csv"
                    )

            except Exception as e:
                st.error(f"Something went wrong: {str(e)}")
                st.session_state.history.append({
                    'question': question,
                    'sql': 'Error generating SQL',
                    'results': None,
                    'summary': f"Error: {str(e)}"
                })