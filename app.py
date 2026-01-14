import streamlit as st
import sys
from pathlib import Path
from dotenv import load_dotenv
import os
import plotly.express as px
import plotly.graph_objects as go
import logging
from datetime import datetime

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
            # Simple password check (you can customize)
            if username and password == "unt2026":
                st.session_state.logged_in = True
                st.session_state.username = username
                st.rerun()
            else:
                st.error("Invalid credentials")
        
        st.stop()  # Stop app execution until logged in

# Check login at app start
check_login()

# Setup logging for user questions only
user_logger = logging.getLogger('user_questions')
user_logger.setLevel(logging.INFO)
handler = logging.FileHandler('usage_log.txt')
handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
user_logger.addHandler(handler)

# Add src to path
sys.path.append('src')
from query_engine import HERDQueryEngine

# Load environment variables
load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')
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
    """)

def create_visualization(df, question):
    """Auto-generate appropriate chart based on data"""
    if df.empty or len(df) == 0:
        return None
    
    # Detect data patterns
    has_year = 'year' in df.columns
    has_name = 'name' in df.columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    if len(numeric_cols) == 0:
        return None
    
    # Time series (line chart)
    if has_year and len(df) > 1:
        # Smart column selection
        if 'growth' in question.lower() or 'yoy' in question.lower():
            growth_cols = [col for col in numeric_cols if 'growth' in col.lower() or 'percent' in col.lower()]
            y_col = growth_cols[0] if growth_cols else numeric_cols[0]
        else:
            y_col = numeric_cols[0]
        
        fig = px.line(df, x='year', y=y_col, 
                     title=f"{y_col} over time",
                     markers=True)
        if has_name and df['name'].nunique() > 1:
            fig = px.line(df, x='year', y=y_col, color='name',
                         title=f"{y_col} over time by institution",
                         markers=True)
        return fig
    
    # Comparison (bar chart)
    elif has_name and len(df) > 1:
        if 'growth' in question.lower() or 'percent' in question.lower():
            growth_cols = [col for col in numeric_cols if 'growth' in col.lower() or 'percent' in col.lower()]
            y_col = growth_cols[0] if growth_cols else numeric_cols[0]
        else:
            y_col = numeric_cols[0]
        
        fig = px.bar(df, x='name', y=y_col,
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
    # Log the question (only once)
    if 'last_logged_question' not in st.session_state or st.session_state.last_logged_question != question:
        user_logger.info(f"User: {st.session_state.username} | Question: {question}")
        st.session_state.last_logged_question = question
    
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Generating SQL and fetching results..."):
            try:
                sql, results, summary = engine.ask(question)

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
