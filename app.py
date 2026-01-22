import streamlit as st
import sys
from pathlib import Path
from dotenv import load_dotenv
import os
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json

# ============================================================
# GOOGLE SHEETS LOGGING
# ============================================================
def get_gsheet_client():
    """Initialize Google Sheets client"""
    try:
        import gspread
        from google.oauth2.service_account import Credentials
        
        # Try Streamlit secrets first, then environment variable
        if hasattr(st, 'secrets') and 'GOOGLE_SHEETS_CREDS' in st.secrets:
            creds_dict = json.loads(st.secrets['GOOGLE_SHEETS_CREDS'])
        elif os.getenv('GOOGLE_SHEETS_CREDS'):
            creds_dict = json.loads(os.getenv('GOOGLE_SHEETS_CREDS'))
        else:
            return None
        
        scopes = [
            'https://www.googleapis.com/auth/spreadsheets',
            'https://www.googleapis.com/auth/drive'
        ]
        creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
        client = gspread.authorize(creds)
        return client
    except Exception as e:
        st.warning(f"Could not connect to Google Sheets: {e}")
        return None

def log_to_sheets(username, question, sql):
    """Log user question to Google Sheets"""
    try:
        client = get_gsheet_client()
        if client is None:
            return
        
        # Get sheet ID from Streamlit secrets or environment variable
        if hasattr(st, 'secrets') and 'GOOGLE_SHEET_ID' in st.secrets:
            sheet_id = st.secrets['GOOGLE_SHEET_ID']
        else:
            sheet_id = os.getenv('GOOGLE_SHEET_ID', '1QBwo8bAPBEDW9DVqtKh9VZEb8CgCLihXC8As_sO16o8')
        
        sheet = client.open_by_key(sheet_id).sheet1
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        sheet.append_row([timestamp, username, question, sql])
    except Exception as e:
        # Silent fail - don't break the app if logging fails
        pass

# ============================================================
# AUTHENTICATION SYSTEM
# ============================================================
def check_login():
    """Simple login system"""
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    
    if not st.session_state.logged_in:
        st.title("🔐 Login Required")
        
        username = st.text_input("Username", placeholder="Enter your first name")
        password = st.text_input("Password", type="password")
        
        if st.button("Login"):
            # Get password from Streamlit secrets or environment variable
            if hasattr(st, 'secrets') and 'PASSWORD' in st.secrets:
                correct_password = st.secrets['PASSWORD']
            else:
                correct_password = os.getenv('PASSWORD', 'unt2026')
            
            if username and password == correct_password:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.rerun()
            else:
                st.error("Invalid credentials")
        
        st.stop()  # Stop app execution until logged in

# Check login at app start
check_login()

# Add src to path
sys.path.append('src')
from query_engine import HERDQueryEngine

# Load environment variables
load_dotenv()

# Get API key and database path from Streamlit secrets or environment variables
if hasattr(st, 'secrets') and 'GEMINI_API_KEY' in st.secrets:
    api_key = st.secrets['GEMINI_API_KEY']
else:
    api_key = os.getenv('GEMINI_API_KEY')

if hasattr(st, 'secrets') and 'DATABASE_PATH' in st.secrets:
    db_path = st.secrets['DATABASE_PATH']
else:
    db_path = os.getenv('DATABASE_PATH')

# Initialize query engine
engine = HERDQueryEngine(api_key, db_path)

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []

# Streamlit UI
st.title("NSF HERD AI Assistant")
st.markdown("Ask questions about university R&D funding (2010-2024)")

# Sidebar for visualization options
with st.sidebar:
    st.header("Visualization")
    enable_viz = st.checkbox("Auto-generate charts", value=True)
    chart_type = st.selectbox("Chart type", ["Auto", "Bar", "Line", "Scatter"])

# Example questions
with st.expander("Example Questions"):
    st.markdown("""
    - What is UNT's total R&D for 2024?
    - Compare UNT and Texas Tech for 2024
    - Show top 5 Texas universities by R&D in 2024
    - Show UNT's R&D from 2020 to 2024
    - What percentage of UNT's 2024 funding is federal?
    - How does UNT compare to its Texas peers in 2024?
    - How does UNT compare to its national peers in 2024?
    - Show Texas universities with the highest R&D growth from 2020 to 2024
    """)

def create_visualization(df, question):
    """Auto-generate appropriate chart based on data and question"""
    if df.empty or len(df) == 0:
        return None
    
    # Detect data patterns
    has_year = 'year' in df.columns
    has_name = 'name' in df.columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    if len(numeric_cols) == 0:
        return None
    
    # Smart column selection based on question keywords
    question_lower = question.lower()
    
    # Priority 1: Growth/CAGR questions - use CAGR or growth column
    if any(word in question_lower for word in ['growth', 'cagr', 'growing', 'fastest', 'rate']):
        cagr_cols = [col for col in numeric_cols if 'cagr' in col.lower() or 'growth' in col.lower() or 'pct' in col.lower()]
        y_col = cagr_cols[0] if cagr_cols else numeric_cols[-1]
    # Priority 2: Funding source questions
    elif any(word in question_lower for word in ['federal', 'institutional', 'business', 'funding', 'source']):
        # For funding source CAGR, prefer the CAGR columns
        if 'cagr' in question_lower:
            cagr_cols = [col for col in numeric_cols if 'cagr' in col.lower()]
            y_col = cagr_cols[0] if cagr_cols else numeric_cols[-1]
        else:
            funding_cols = [col for col in numeric_cols if any(f in col.lower() for f in ['federal', 'institutional', 'business', 'state', 'nonprofit'])]
            y_col = funding_cols[0] if funding_cols else numeric_cols[0]
    # Priority 3: Total/comparison questions - use latest year or total
    elif any(word in question_lower for word in ['total', 'compare', 'top', 'rank', 'peers']):
        total_cols = [col for col in numeric_cols if 'total' in col.lower() or col == 'total_rd']
        latest_cols = [col for col in numeric_cols if '2024' in col]
        if latest_cols:
            y_col = latest_cols[0]
        elif total_cols:
            y_col = total_cols[-1]
        else:
            y_col = numeric_cols[-1]
    # Default: use last numeric column (often the calculated result)
    else:
        y_col = numeric_cols[-1]
    
    # Time series (line chart) - when showing trends over years
    if has_year and len(df) > 1 and df['year'].nunique() > 1:
        fig = px.line(df, x='year', y=y_col, 
                     title=f"{y_col} over time",
                     markers=True)
        if has_name and df['name'].nunique() > 1:
            fig = px.line(df, x='year', y=y_col, color='name',
                         title=f"{y_col} over time by institution",
                         markers=True)
        return fig
    
    # Bar chart - for comparisons across institutions
    elif has_name and len(df) > 1:
        df_sorted = df.sort_values(by=y_col, ascending=False)
        
        fig = px.bar(df_sorted, x='name', y=y_col,
                    title=f"{y_col} by institution")
        fig.update_xaxes(tickangle=-45)
        return fig
    
    return None
    
    # Bar chart - for comparisons across institutions
    elif has_name and len(df) > 1:
        # Sort by y_col for better visualization
        df_sorted = df.sort_values(by=y_col, ascending=False)
        
        fig = px.bar(df_sorted, x='name', y=y_col,
                    title=f"{y_col} by institution")
        fig.update_xaxes(tickangle=-45)
        return fig
    
    return None


# Display conversation history
for item in st.session_state.history:
    with st.chat_message("user"):
        st.write(item['question'])
    with st.chat_message("assistant"):
        with st.expander("Generated SQL"):
            st.code(item['sql'], language="sql")
        st.dataframe(item['results'], use_container_width=True)
        if item.get('chart'):
            st.plotly_chart(item['chart'], use_container_width=True)

# Query input
question = st.chat_input("Ask a question about university R&D funding...")

if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Generating SQL and fetching results..."):
            try:
                sql, results, summary = engine.ask(question)

                # Log to Google Sheets (only once per question)
                if 'last_logged_question' not in st.session_state or st.session_state.last_logged_question != question:
                    log_to_sheets(st.session_state.username, question, sql)
                    st.session_state.last_logged_question = question

                # Show SQL
                with st.expander("Generated SQL"):
                    st.code(sql, language="sql")

                # Show results
                st.dataframe(results, use_container_width=True)
                st.success("Query executed successfully")
                st.info(f"📊 {summary}")

                # Generate visualization
                chart = None
                if enable_viz and len(results) > 0:
                    chart = create_visualization(results, question)
                    if chart:
                        st.plotly_chart(chart, use_container_width=True)

                # Download button
                csv = results.to_csv(index=False)
                st.download_button("Download CSV", csv, "results.csv", "text/csv")

                # Save to history
                st.session_state.history.append({
                    'question': question,
                    'sql': sql,
                    'results': results,
                    'chart': chart
                })

            except Exception as e:
                st.error(f"Error: {str(e)}")
