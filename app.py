import streamlit as st
import sys
from pathlib import Path
from dotenv import load_dotenv
import os
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import yaml
from zoneinfo import ZoneInfo
# Debug: Print all environment variables
import sys
print("=" * 50, file=sys.stderr)
print("ENVIRONMENT VARIABLES:", file=sys.stderr)
for key in ['GEMINI_API_KEY', 'DATABASE_PATH', 'PASSWORD']:
    val = os.getenv(key)
    print(f"{key} = {val}", file=sys.stderr)
print("=" * 50, file=sys.stderr)

# Initialize session state first
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = None
    
# Load config
with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)

# ============================================================
# GOOGLE SHEETS LOGGING
# ============================================================
def get_gsheet_client():
    """Initialize Google Sheets client"""
    try:
        import gspread
        from google.oauth2.service_account import Credentials
        
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
        
        if hasattr(st, 'secrets') and 'GOOGLE_SHEET_ID' in st.secrets:
            sheet_id = st.secrets['GOOGLE_SHEET_ID']
        else:
            sheet_id = os.getenv('GOOGLE_SHEET_ID', '1QBwo8bAPBEDW9DVqtKh9VZEb8CgCLihXC8As_sO16o8')
        
        sheet = client.open_by_key(sheet_id).sheet1
        timestamp = datetime.now(ZoneInfo('America/Chicago')).strftime('%Y-%m-%d %H:%M:%S')
        sheet.append_row([timestamp, username, question, sql])
    except Exception as e:
        pass

# ============================================================
# AUTHENTICATION SYSTEM
# ============================================================
def check_login():
    """Simple login system"""
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'username' not in st.session_state:
        st.session_state.username = None
    
    if not st.session_state.logged_in:
        st.title("🔐 Login Required")
        
        username = st.text_input("Username", placeholder="Enter your first name")
        password = st.text_input("Password", type="password")
        
        if st.button("Login"):
            try:
                correct_password = st.secrets.get('PASSWORD', None)
            except (FileNotFoundError, AttributeError):
                correct_password = None
            
            if correct_password is None:
                correct_password = os.getenv('PASSWORD', config['auth']['default_password'])
            
            if username and password == correct_password:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.rerun()
            else:
                st.error("Invalid credentials")
        
        st.stop()

check_login()

# Add src to path
sys.path.append('src')
from query_engine import HERDQueryEngine

# Load environment variables
load_dotenv()

api_key = os.getenv('GEMINI_API_KEY')
if not api_key:
    try:
        api_key = st.secrets.get('GEMINI_API_KEY')
    except:
        api_key = None

db_path = os.getenv('DATABASE_PATH')
if not db_path:
    try:
        db_path = st.secrets.get('DATABASE_PATH')
    except:
        db_path = None

# Initialize query engine
engine = HERDQueryEngine(api_key, db_path)

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []

# Get config values
short_name = config['institution']['short_name']
inst_name = config['institution']['name']
primary_color = config['branding']['primary_color']
header_text = config['branding']['header_text']
default_target = config['targets']['default_amount_millions']
default_year = config['targets']['default_target_year']

# Streamlit UI
st.title(f"NSF HERD AI Assistant - {short_name}")
st.markdown(f"Ask questions about university R&D funding ({config['analysis']['data_start_year']}-{config['analysis']['data_end_year']})")

# Sidebar for visualization options
with st.sidebar:
    st.header("Visualization")
    enable_viz = st.checkbox("Auto-generate charts", value=True)
    chart_type = st.selectbox("Chart type", ["Auto", "Bar", "Line", "Scatter"])
    
    if st.button("Clear History"):
        st.session_state.history = []
        st.rerun()

# ============================================================
# STRATEGIC REPORT GENERATOR
# ============================================================
with st.sidebar:
    st.divider()
    st.header("📊 Strategic Report")
    
    analysis_period = st.selectbox(
        "Analysis Period",
        options=["5-Year (2019-2024)", "10-Year (2014-2024)", "Custom"],
        index=0
    )
    
    if analysis_period == "Custom":
        col1, col2 = st.columns(2)
        with col1:
            start_year = st.number_input("Start Year", min_value=2010, max_value=2023, value=2015)
        with col2:
            end_year = st.number_input("End Year", min_value=2011, max_value=2024, value=2024)
    elif analysis_period == "5-Year (2019-2024)":
        start_year = 2019
        end_year = 2024
    else:
        start_year = 2014
        end_year = 2024
    
    target_amount = st.number_input(
        "Target R&D ($M)", 
        min_value=100, 
        max_value=500, 
        value=default_target,
        step=10
    )
    
    target_year = st.selectbox(
        "Target Year",
        options=[2028, 2029, 2030, 2031, 2032],
        index=[2028, 2029, 2030, 2031, 2032].index(default_year)
    )
    
    peer_group = st.selectbox(
        "Peer Group",
        options=["Texas Peers", "National Peers"],
        index=0
    )
    
    generate_report = st.button("🚀 Generate Report", type="primary", use_container_width=True)

# Example questions
with st.expander("Example Questions"):
    st.markdown(f"""
    - What is {short_name}'s total R&D for 2024?
    - Compare {short_name} and Texas Tech for 2024
    - Show top 5 Texas universities by R&D in 2024
    - Show {short_name}'s R&D from 2020 to 2024
    - What percentage of {short_name}'s 2024 funding is federal?
    - How does {short_name} compare to its Texas peers in 2024?
    - How does {short_name} compare to its national peers in 2024?
    """)

def create_visualization(df, question):
    """Auto-generate appropriate chart based on data and question"""
    if df is None or df.empty or len(df) == 0:
        return None
    
    has_year = 'year' in df.columns
    has_name = 'name' in df.columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    if len(numeric_cols) == 0:
        return None
    
    question_lower = question.lower()
    
    if any(word in question_lower for word in ['growth', 'cagr', 'growing', 'fastest', 'rate']):
        cagr_cols = [col for col in numeric_cols if 'cagr' in col.lower() or 'growth' in col.lower() or 'pct' in col.lower()]
        y_col = cagr_cols[0] if cagr_cols else numeric_cols[-1]
    elif any(word in question_lower for word in ['federal', 'institutional', 'business', 'funding', 'source']):
        if 'cagr' in question_lower:
            cagr_cols = [col for col in numeric_cols if 'cagr' in col.lower()]
            y_col = cagr_cols[0] if cagr_cols else numeric_cols[-1]
        else:
            funding_cols = [col for col in numeric_cols if any(f in col.lower() for f in ['federal', 'institutional', 'business', 'state', 'nonprofit'])]
            y_col = funding_cols[0] if funding_cols else numeric_cols[0]
    elif any(word in question_lower for word in ['total', 'compare', 'top', 'rank', 'peers']):
        total_cols = [col for col in numeric_cols if 'total' in col.lower() or col == 'total_rd']
        latest_cols = [col for col in numeric_cols if '2024' in col]
        if latest_cols:
            y_col = latest_cols[0]
        elif total_cols:
            y_col = total_cols[-1]
        else:
            y_col = numeric_cols[-1]
    else:
        y_col = numeric_cols[-1]
    
    if has_year and len(df) > 1 and df['year'].nunique() > 1:
        fig = px.line(df, x='year', y=y_col, title=f"{y_col} over time", markers=True)
        if has_name and df['name'].nunique() > 1:
            fig = px.line(df, x='year', y=y_col, color='name', title=f"{y_col} over time by institution", markers=True)
        return fig
    
    elif has_name and len(df) > 1:
        df_sorted = df.sort_values(by=y_col, ascending=False)
        fig = px.bar(df_sorted, x='name', y=y_col, title=f"{y_col} by institution")
        fig.update_xaxes(tickangle=-45)
        return fig
    
    return None


# ============================================================
# REPORT GENERATION LOGIC
# ============================================================
def run_strategic_queries(engine, peer_group, start_year, end_year):
    """Run all queries needed for strategic report"""
    
    peer_label = "Texas peers" if peer_group == "Texas Peers" else "national peers"
    num_years = end_year - start_year
    inst = config['institution']['short_name']
    
    queries = [
        {
            "name": f"{inst} Trajectory",
            "question": f"Show {inst} Denton's total R&D, federal, state_local, business, nonprofit, and institutional funding from {start_year} to {end_year}"
        },
        {
            "name": "Current Position", 
            "question": f"Compare {inst} Denton to its {peer_label} for {end_year} total R&D, sorted highest to lowest"
        },
        {
            "name": f"Total R&D Growth ({num_years}-Year)",
            "question": f"Show total R&D CAGR from {start_year} to {end_year} for {inst} Denton and its {peer_label}, sorted highest to lowest"
        },
        {
            "name": f"Federal Funding Growth ({num_years}-Year)",
            "question": f"Show federal funding CAGR from {start_year} to {end_year} for {inst} Denton and its {peer_label}, sorted highest to lowest"
        },
        {
            "name": f"Institutional Investment Growth ({num_years}-Year)",
            "question": f"Show institutional funding CAGR from {start_year} to {end_year} for {inst} Denton and its {peer_label}, sorted highest to lowest"
        }
    ]
    
    results = []
    for q in queries:
        try:
            sql, df, summary = engine.ask(q["question"])
            results.append({
                "name": q["name"],
                "question": q["question"],
                "sql": sql,
                "data": df,
                "summary": summary
            })
        except Exception as e:
            results.append({
                "name": q["name"],
                "question": q["question"],
                "sql": None,
                "data": None,
                "summary": f"Error: {str(e)}"
            })
    
    return results


def calculate_required_cagr(current_rd, target_rd, years):
    """Calculate CAGR needed to reach target"""
    if years <= 0 or current_rd <= 0:
        return 0
    return ((target_rd / current_rd) ** (1 / years) - 1) * 100


def generate_executive_narrative(engine, report_data, target_amount, target_year, current_rd, start_year, end_year):
    """Use Gemini to synthesize findings into executive narrative"""
    
    findings = "\n".join([
        f"- {r['name']}: {r['summary']}" 
        for r in report_data if r['summary'] and not r['summary'].startswith('Error')
    ])
    
    required_cagr = calculate_required_cagr(current_rd, target_amount, target_year - end_year)
    num_years = end_year - start_year
    inst = config['institution']['short_name']
    
    prompt = f"""Based on these research funding analysis findings for {inst} ({num_years}-year analysis from {start_year} to {end_year}):

{findings}

Key metrics:
- Current R&D ({end_year}): ${current_rd}M
- Target R&D: ${target_amount}M by {target_year}
- Required CAGR: {required_cagr:.1f}%

Write a 3-sentence executive summary for the university cabinet:
1. Where we are (current position and growth)
2. The gap (what's holding us back)
3. The ask (what we need to do)

No fluff. No bullet points. Direct and action-oriented."""

    return engine.generate_narrative(prompt)


def generate_pdf_report(report_data, narrative, params):
    """Generate PDF report from collected data"""
    from fpdf import FPDF
    
    # Parse primary color from hex
    pc = config['branding']['primary_color'].lstrip('#')
    pr, pg, pb = int(pc[0:2], 16), int(pc[2:4], 16), int(pc[4:6], 16)
    
    class PDF(FPDF):
        def header(self):
            self.set_font('Helvetica', 'B', 10)
            self.set_text_color(pr, pg, pb)
            self.cell(0, 10, config['branding']['header_text'], 0, 1, 'R')
            self.ln(5)
        
        def footer(self):
            self.set_y(-15)
            self.set_font('Helvetica', 'I', 8)
            self.set_text_color(128, 128, 128)
            self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
    
    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    # Title
    pdf.set_font('Helvetica', 'B', 24)
    pdf.set_text_color(pr, pg, pb)
    pdf.cell(0, 15, f'{short_name} Research: Path to ${params["target_amount"]}M', 0, 1, 'C')
    
    # Subtitle
    pdf.set_font('Helvetica', '', 12)
    pdf.set_text_color(100, 100, 100)
    analysis_label = f'{params["start_year"]}-{params["end_year"]} Analysis'
    pdf.cell(0, 10, f'Strategic Briefing | {analysis_label} | Generated {datetime.now().strftime("%B %d, %Y")}', 0, 1, 'C')
    pdf.ln(10)
    
    # Key Metrics Box
    pdf.set_fill_color(245, 247, 245)
    pdf.set_draw_color(pr, pg, pb)
    pdf.rect(10, pdf.get_y(), 190, 25, 'DF')
    
    pdf.set_font('Helvetica', 'B', 11)
    pdf.set_text_color(0, 0, 0)
    y_pos = pdf.get_y() + 5
    
    pdf.set_xy(15, y_pos)
    pdf.cell(60, 6, f'Current R&D ({params["end_year"]})', 0, 0, 'C')
    pdf.cell(60, 6, f'Target', 0, 0, 'C')
    pdf.cell(60, 6, f'Required CAGR', 0, 1, 'C')
    
    pdf.set_font('Helvetica', 'B', 14)
    pdf.set_text_color(pr, pg, pb)
    pdf.set_x(15)
    pdf.cell(60, 10, f'${params["current_rd"]:.1f}M', 0, 0, 'C')
    pdf.cell(60, 10, f'${params["target_amount"]}M by {params["target_year"]}', 0, 0, 'C')
    pdf.cell(60, 10, f'{params["required_cagr"]:.1f}%', 0, 1, 'C')
    
    pdf.ln(15)
    
    # Executive Summary
    pdf.set_font('Helvetica', 'B', 14)
    pdf.set_text_color(pr, pg, pb)
    pdf.cell(0, 10, 'Executive Summary', 0, 1, 'L')
    
    pdf.set_font('Helvetica', '', 11)
    pdf.set_text_color(0, 0, 0)
    pdf.multi_cell(0, 6, narrative)
    pdf.ln(10)
    
    # Findings sections
    for i, section in enumerate(report_data):
        if section['data'] is None:
            continue
            
        pdf.set_font('Helvetica', 'B', 12)
        pdf.set_text_color(pr, pg, pb)
        pdf.cell(0, 10, f'{i+1}. {section["name"]}', 0, 1, 'L')
        
        if section.get('summary') and not section['summary'].startswith('Error'):
            pdf.set_font('Helvetica', 'I', 10)
            pdf.set_text_color(80, 80, 80)
            summary_clean = section['summary'].replace('$', '').replace('\\', '')
            pdf.multi_cell(0, 5, summary_clean)
            pdf.ln(3)
        
        df = section['data'].head(10)
        if len(df) > 0:
            pdf.set_font('Helvetica', '', 9)
            pdf.set_text_color(0, 0, 0)
            
            cols = df.columns.tolist()
            col_width = 190 / len(cols)
            
            pdf.set_font('Helvetica', 'B', 9)
            pdf.set_fill_color(pr, pg, pb)
            pdf.set_text_color(255, 255, 255)
            for col in cols:
                col_display = str(col)[:20]
                pdf.cell(col_width, 7, col_display, 1, 0, 'C', True)
            pdf.ln()
            
            pdf.set_font('Helvetica', '', 8)
            pdf.set_text_color(0, 0, 0)
            for idx, row in df.iterrows():
                for col in cols:
                    val = row[col]
                    if isinstance(val, float):
                        val_str = f'{val:,.1f}'
                    elif isinstance(val, int):
                        val_str = f'{val:,}'
                    else:
                        val_str = str(val)[:25]
                    pdf.cell(col_width, 6, val_str, 1, 0, 'C')
                pdf.ln()
        
        pdf.ln(8)
        
        if pdf.get_y() > 250:
            pdf.add_page()
    
    return bytes(pdf.output())


# Handle report generation
if generate_report:
    st.divider()
    
    num_years = end_year - start_year
    st.header(f"📊 {short_name} Research: Path to ${target_amount}M ({num_years}-Year Analysis)")
    
    with st.spinner("Running strategic analysis..."):
        report_data = run_strategic_queries(engine, peer_group, start_year, end_year)
    
    for i, section in enumerate(report_data):
        st.subheader(f"{i+1}. {section['name']}")
        
        if section['data'] is not None and len(section['data']) > 0:
            st.dataframe(section['data'], use_container_width=True)
            
            chart = create_visualization(section['data'], section['question'])
            if chart:
                st.plotly_chart(chart, use_container_width=True)
            
            if section['summary']:
                st.info(section['summary'])
        else:
            st.warning(f"Could not retrieve data: {section['summary']}")
    
    st.subheader("6. Path to Target")
    
    try:
        unt_data = report_data[0]['data']
        current_rd = unt_data[unt_data['year'] == end_year]['total_rd'].values[0] / 1_000_000
    except:
        current_rd = 124.2
    
    years_to_target = target_year - end_year
    required_cagr = calculate_required_cagr(current_rd, target_amount, years_to_target)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(f"Current R&D ({end_year})", f"${current_rd:.1f}M")
    with col2:
        st.metric("Target", f"${target_amount}M by {target_year}")
    with col3:
        st.metric("Required CAGR", f"{required_cagr:.1f}%")
    
    st.subheader("Executive Summary")
    with st.spinner("Generating executive narrative..."):
        try:
            narrative = generate_executive_narrative(
                engine, report_data, target_amount, target_year, current_rd, start_year, end_year
            )
            narrative = narrative.replace('$', '')
            narrative = narrative.replace('\\', '')
            narrative = ' '.join(narrative.split())
            st.markdown(narrative)
        except Exception as e:
            st.error(f"Could not generate narrative: {str(e)}")
            narrative = "Executive summary could not be generated."
    
    try:
        pdf_bytes = generate_pdf_report(report_data, narrative, {
            "target_amount": target_amount,
            "target_year": target_year,
            "current_rd": current_rd,
            "required_cagr": required_cagr,
            "start_year": start_year,
            "end_year": end_year
        })
        
        st.download_button(
            label="📥 Download PDF Report",
            data=pdf_bytes,
            file_name=f"{short_name}_Path_to_{target_amount}M_{num_years}yr_{datetime.now().strftime('%Y%m%d')}.pdf",
            mime="application/pdf",
            type="primary",
            use_container_width=True
        )
        st.success("✅ Report generated!")
    except Exception as e:
        st.error(f"Could not generate PDF: {str(e)}")


# Display conversation history
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
            csv = item['results'].to_csv(index=False)
            st.download_button(
                "Download CSV",
                csv,
                f"results_{item['question'][:20].replace(' ', '_')}.csv",
                "text/csv",
                key=f"download_{hash(item['question'])}"
            )

# Query input
question = st.chat_input("Ask a question about university R&D funding...")

if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Generating SQL and fetching results..."):
            try:
                sql, results, summary = engine.ask(question)
                log_to_sheets(st.session_state.username, question, sql)

                with st.expander("Generated SQL"):
                    st.code(sql, language="sql")

                if results is not None and len(results) > 0:
                    st.dataframe(results, use_container_width=True)
                    st.success("Query executed successfully")
                else:
                    st.warning("Query returned no results")

                if summary:
                    st.info(f"📊 {summary}")

                chart = None
                if enable_viz and results is not None and len(results) > 0:
                    chart = create_visualization(results, question)
                    if chart:
                        st.plotly_chart(chart, use_container_width=True)

                st.session_state.history.append({
                    'question': question,
                    'sql': sql,
                    'results': results,
                    'summary': summary
                })
                
                if len(st.session_state.history) > 20:
                    st.session_state.history = st.session_state.history[-20:]

            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.session_state.history.append({
                    'question': question,
                    'sql': 'Error generating SQL',
                    'results': None,
                    'summary': f"Error: {str(e)}"
                })