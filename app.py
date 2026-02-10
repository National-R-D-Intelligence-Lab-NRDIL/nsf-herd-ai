import streamlit as st
import sys
from pathlib import Path
from dotenv import load_dotenv
import os
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
from zoneinfo import ZoneInfo
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Environment setup - works across Railway, Streamlit Cloud, and local.
# Railway and Streamlit Cloud both expose secrets as env vars.
# Locally, we fall back to .env file via dotenv.
# ============================================================
load_dotenv()

def get_env(key, default=None):
    """Single place to pull secrets — env vars first, then .env fallback."""
    return os.getenv(key, default)

GEMINI_API_KEY = get_env('GEMINI_API_KEY')
DATABASE_PATH  = get_env('DATABASE_PATH', 'data/herd.db')
PASSWORD       = get_env('PASSWORD', 'unt2026')
GOOGLE_SHEET_ID    = get_env('GOOGLE_SHEET_ID')
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
# Google Sheets logging — fires and forgets.
# If credentials aren't configured, it skips silently.
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
from benchmarker import fetch_university_features, AutoBenchmarker

engine = HERDQueryEngine(GEMINI_API_KEY, DATABASE_PATH)

# ============================================================
# Benchmarker setup — cached so the KNN model is fitted once.
# ============================================================
@st.cache_resource(show_spinner="Loading benchmarking model...")
def load_benchmarker():
    df = fetch_university_features(DATABASE_PATH)
    bench = AutoBenchmarker(n_peers=10)
    bench.fit(df)
    return bench

benchmarker = load_benchmarker()

# ============================================================
# Session state
# ============================================================
if 'history' not in st.session_state:
    st.session_state.history = []
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = 'snapshot'

# Cache the institution list — it's a full table scan, no point
# running it on every rerender.
@st.cache_data(ttl=3600)
def load_institution_list():
    return engine.get_institution_list()

# ============================================================
# Helper: format dollar amounts for display
# ============================================================
def fmt_dollars(n):
    if n >= 1e9:
        return f"${n / 1e9:.2f}B"
    if n >= 1e6:
        return f"${n / 1e6:.1f}M"
    return f"${n:,.0f}"

# ============================================================
# Executive Summary tab
# National peer gap + state context in one view.
# ============================================================
METRIC_LABELS = {
    "total_rd":       "Total R&D",
    "federal":        "Federal",
    "state_local":    "State & Local",
    "business":       "Business",
    "nonprofit":      "Nonprofit",
    "institutional":  "Institutional",
    "other_sources":  "Other Sources",
}

def render_executive_summary_tab():
    institution_list = load_institution_list()

    selected = st.selectbox(
        "Select your institution",
        options=[""] + institution_list,
        index=0,
        placeholder="Select an institution...",
        format_func=lambda x: "Select an institution..." if x == "" else x,
        key="exec_summary_institution_picker",
    )

    if not selected or selected == "":
        st.info("Select an institution above to generate an executive summary.")
        return

    # --- Map name → inst_id via the benchmarker's fitted data ---
    match = benchmarker.data[benchmarker.data["name"] == selected]
    if match.empty:
        st.warning(f"'{selected}' was not found in the benchmarking dataset.")
        return
    inst_id = match["inst_id"].values[0]

    # --- Pull all three analyses ---
    try:
        gap_data    = benchmarker.analyze_gap(inst_id)
        state_ctx   = benchmarker.analyze_state_context(inst_id)
        peer_names  = benchmarker.get_peers(inst_id)
    except Exception as e:
        st.error(f"Benchmarking error: {e}")
        return

    target_rd    = next(g for g in gap_data if g["metric"] == "total_rd")
    target_total = target_rd["my_val"]
    peer_avg     = target_rd["peer_avg"]

    # ==============================================================
    # Row 1 — headline metrics
    # ==============================================================
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(
            "Total R&D",
            fmt_dollars(target_total),
            help="Your institution's total R&D expenditures",
        )
    with c2:
        delta = target_total - peer_avg
        st.metric(
            "vs Peer Average",
            fmt_dollars(peer_avg),
            delta=f"{'+' if delta >= 0 else ''}{fmt_dollars(abs(delta))}",
            delta_color="normal" if delta >= 0 else "inverse",
            help="Average total R&D of your 10 closest national peers",
        )
    with c3:
        st.metric(
            "State Rank",
            f"#{state_ctx['state_rank']} of {state_ctx['total_in_state']}",
            help=f"Rank by total R&D among all institutions in {state_ctx['state']}",
        )
    with c4:
        st.metric(
            "State Funding Share",
            f"{state_ctx['state_funding_share']:.1f}%",
            help="Your share of all state & local government R&D funding in your state",
        )

    st.divider()

    # ==============================================================
    # Row 2 — two-column deep dive
    # ==============================================================
    left, right = st.columns([3, 2])

    # --- LEFT: National Peer Gap Chart ---
    with left:
        st.subheader("National Peer Comparison")
        st.caption(f"Your funding profile vs. the average of your {len(peer_names)} closest peers")

        labels   = [METRIC_LABELS.get(g["metric"], g["metric"]) for g in gap_data]
        my_vals  = [g["my_val"] for g in gap_data]
        avg_vals = [g["peer_avg"] for g in gap_data]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=labels, x=my_vals, orientation="h",
            name=selected.split(",")[0],  # short name
            marker_color="#2563EB",
            text=[fmt_dollars(v) for v in my_vals],
            textposition="outside",
            textfont=dict(size=11),
        ))
        fig.add_trace(go.Bar(
            y=labels, x=avg_vals, orientation="h",
            name="Peer Average",
            marker_color="#D1D5DB",
            text=[fmt_dollars(v) for v in avg_vals],
            textposition="outside",
            textfont=dict(size=11),
        ))
        fig.update_layout(
            barmode="group",
            height=340,
            margin=dict(l=10, r=80, t=10, b=30),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            xaxis=dict(title=None, showgrid=True, gridcolor="#F3F4F6", tickformat="$,.0f"),
            yaxis=dict(title=None, autorange="reversed"),
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Gap table in an expander
        with st.expander("Detailed gap numbers"):
            gap_df = []
            for g in gap_data:
                label = METRIC_LABELS.get(g["metric"], g["metric"])
                gap_val = g["gap"]
                gap_df.append({
                    "Metric": label,
                    "You": fmt_dollars(g["my_val"]),
                    "Peer Avg": fmt_dollars(g["peer_avg"]),
                    "Gap": f"{'+'if gap_val>=0 else ''}{fmt_dollars(abs(gap_val))}",
                })
            st.dataframe(gap_df, use_container_width=True, hide_index=True)

    # --- RIGHT: State Context ---
    with right:
        st.subheader(f"State Context — {state_ctx['state']}")

        # Top competitor callout
        if state_ctx["top_competitor"]:
            st.markdown(
                f"**#1 in {state_ctx['state']}:** {state_ctx['top_competitor']}"
            )
        else:
            st.success(f"You are #1 in {state_ctx['state']}")

        # State rank position visual — horizontal bar showing where
        # the institution sits among all in-state schools.
        state_df = (
            benchmarker.data[benchmarker.data["state"] == state_ctx["state"]]
            .sort_values("total_rd", ascending=False)
            .head(15)  # top 15 keeps the chart readable
        )

        is_target = state_df["inst_id"] == inst_id
        colors = ["#2563EB" if t else "#93C5FD" for t in is_target]

        # Truncate long names
        display_names = []
        for n, t in zip(state_df["name"], is_target):
            label = n if len(n) < 35 else n[:32] + "…"
            if t:
                label = f"► {label}"
            display_names.append(label)

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=state_df["total_rd"].tolist(),
            y=display_names,
            orientation="h",
            marker_color=colors,
            text=[fmt_dollars(v) for v in state_df["total_rd"]],
            textposition="outside",
            textfont=dict(size=11, color="#374151"),
            hovertemplate="%{y}<br>%{text}<extra></extra>",
        ))
        fig2.update_layout(
            height=max(200, len(state_df) * 36),
            margin=dict(l=220, r=70, t=10, b=30),
            xaxis=dict(title="Total R&D", showgrid=True, gridcolor="#F3F4F6", tickformat="$,.0f"),
            yaxis=dict(title=None, categoryorder="array", categoryarray=display_names),
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        st.plotly_chart(fig2, use_container_width=True)

        if state_ctx["total_in_state"] > 15:
            st.caption(
                f"Showing top 15 of {state_ctx['total_in_state']} institutions in {state_ctx['state']}"
            )

    st.divider()

    # ==============================================================
    # Row 3 — Peer list
    # ==============================================================
    with st.expander(f"Your {len(peer_names)} National Peers (by funding similarity)"):
        # Display as a numbered list with each peer's total R&D for context
        peer_rows = []
        for name in peer_names:
            row = benchmarker.data[benchmarker.data["name"] == name]
            if not row.empty:
                peer_rows.append({
                    "Institution": name,
                    "State": row["state"].values[0],
                    "Total R&D": fmt_dollars(float(row["total_rd"].values[0])),
                })
        st.dataframe(peer_rows, use_container_width=True, hide_index=True)

# ============================================================
# Snapshot: rank trend visualization
# Horizontal bars per year. Latest year is darker.
# A badge at top shows net movement over the window.
# ============================================================
def render_rank_trend(df, total_institutions):
    if df.empty:
        st.warning("No ranking data found for this institution.")
        return

    first_rank = int(df.iloc[0]['national_rank'])
    last_rank  = int(df.iloc[-1]['national_rank'])
    moved = first_rank - last_rank  # positive = climbed

    # Badge: net movement
    if moved > 0:
        st.success(f"↑ Climbed {moved} positions over this period")
    elif moved < 0:
        st.error(f"↓ Dropped {abs(moved)} positions over this period")
    else:
        st.info("Rank unchanged over this period")

    # Build the horizontal bar chart with Plotly
    years = df['year'].tolist()
    ranks = df['national_rank'].tolist()

    # Color: latest year is bold blue, others are light
    colors = ['#93C5FD'] * len(years)
    colors[-1] = '#2563EB'

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=ranks,
        y=[str(y) for y in years],
        orientation='h',
        marker_color=colors,
        text=[f"#{r}" for r in ranks],
        textposition='outside',
        textfont=dict(size=14, color='#374151'),
        hovertemplate='Year: %{y}<br>Rank: #%{x}<extra></extra>'
    ))

    # Invert x-axis so #1 is on the right (better = further right)
    max_rank_show = max(ranks) + 15
    fig.update_layout(
        xaxis=dict(range=[max_rank_show, 0], title='National Rank', showgrid=True, gridcolor='#F3F4F6'),
        yaxis=dict(title=None, categoryorder='array', categoryarray=[str(y) for y in years]),
        height=220,
        margin=dict(l=50, r=60, t=10, b=30),
        plot_bgcolor='white',
        paper_bgcolor='white',
    )
    fig.update_xaxes(tickprefix='#')

    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Ranked out of {total_institutions:,} institutions nationally")

# ============================================================
# Snapshot: anchor view visualization
# Shows the target institution in context with benchmark schools
# above and below. Target row is visually distinct.
# ============================================================
def render_anchor_view(anchor_df, target_rank, total_institutions):
    if anchor_df.empty:
        st.warning("Could not build anchor view.")
        return

    names  = anchor_df['name'].tolist()
    rd     = anchor_df['total_rd'].tolist()
    ranks  = anchor_df['national_rank'].tolist()
    is_tgt = anchor_df['is_target'].tolist()

    # Color and label: target stands out
    colors = ['#2563EB' if t else '#9CA3AF' for t in is_tgt]
    labels = [f"#{r}  {fmt_dollars(v)}" for r, v in zip(ranks, rd)]

    # Truncate long names for the axis
    display_names = []
    for n, t in zip(names, is_tgt):
        label = n if len(n) < 38 else n[:35] + "…"
        if t:
            label = f"► {label}"
        display_names.append(label)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=rd,
        y=display_names,
        orientation='h',
        marker_color=colors,
        text=labels,
        textposition='outside',
        textfont=dict(size=11, color='#374151'),
        hovertemplate='%{y}<br>R&D: %{text}<extra></extra>'
    ))

    fig.update_layout(
        xaxis=dict(title='Total R&D', showgrid=True, gridcolor='#F3F4F6',
                   tickformat='$,.0f'),
        yaxis=dict(title=None, categoryorder='array', categoryarray=display_names),
        height=max(200, len(names) * 52),
        margin=dict(l=280, r=100, t=10, b=30),
        plot_bgcolor='white',
        paper_bgcolor='white',
    )

    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# Snapshot tab: the full institution snapshot experience.
# User picks a school and a time window, we show:
#   - 3 stat cards (current rank, R&D, movement)
#   - Rank trend over the window
#   - Anchor view for context
# ============================================================


def render_executive_summary(metrics, insight, selected_institution, start_year, end_year, all_charts):
    st.subheader("Strategic Summary")
    
    if not metrics:
        st.warning("Unable to generate summary")
        return
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(
            "Current Rank",
            f"#{metrics['current_rank']}",
            delta=metrics['rank_change'] if metrics['rank_change'] != 0 else None,
            delta_color="normal"
        )
    with c2:
        st.metric(
            "Total R&D (2024)",
            fmt_dollars(metrics['current_rd']),
            delta=fmt_dollars(metrics['rd_change'])
        )
    with c3:
        growth_delta = round(metrics['target_growth'] - metrics['peer_avg'], 1)
        st.metric(
            "5-Year Growth",
            f"{metrics['target_growth']}%",
            delta=f"{growth_delta}% vs peers"
        )
    
    if metrics['target_growth'] > metrics['peer_avg'] + 5:
        status = "🟢 Strong - Growing faster than peers"
    elif metrics['target_growth'] < metrics['peer_avg'] - 5:
        status = "🔴 Lagging - Growing slower than peers"
    else:
        status = "🟡 Moderate - Growth aligned with peers"
    
    st.info(f"**Status:** {status}")
    st.info(f"💡 **Strategic Insight:** {insight}")
    
#    if st.button("📄 Download Strategic Report (PDF)", type="primary"):
#        with st.spinner("Generating PDF report..."):
#            pdf_bytes = engine.generate_pdf_report(
#                selected_institution,
#                start_year,
#                end_year,
#                all_charts
#            )
#            
#            if pdf_bytes:
#                st.download_button(
#                    label="Download PDF",
#                    data=pdf_bytes,
#                    file_name=f"{selected_institution.replace(' ', '_')}_Report_{start_year}-{end_year}.pdf",
#                    mime="application/pdf"
#                )
#            else:
#                st.error("Failed to generate PDF. Please try again.")
#    
#    st.markdown("---")

def render_peer_comparison(peers_df, stats, selected_institution):
    if peers_df.empty:
        st.warning("Insufficient peer data")
        return
    
    st.subheader("Peer Performance Comparison")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Your Growth", f"{stats['target_growth']}%")
    with c2:
        st.metric("Peer Average", f"{stats['peer_avg']}%")
    with c3:
        st.metric("Rank Among Peers", f"#{stats['rank']} of {stats['total_peers']}")
    
    fig = go.Figure()
    
    for name in peers_df['name'].unique():
        inst_data = peers_df[peers_df['name'] == name]
        is_target = name == selected_institution
        
        fig.add_trace(go.Scatter(
            x=inst_data['year'],
            y=inst_data['total_rd'],
            mode='lines+markers',
            name=name if is_target else None,
            line=dict(
                width=3 if is_target else 1,
                color='#2563EB' if is_target else '#9CA3AF',
                dash='solid' if is_target else 'dot'
            ),
            marker=dict(size=8 if is_target else 4),
            showlegend=is_target,
            hovertemplate=f'{name}<br>%{{y:$,.0f}}<extra></extra>'
        ))
    
    fig.update_layout(
        xaxis_title='Year',
        yaxis_title='Total R&D',
        height=400,
        hovermode='x unified',
        plot_bgcolor='white',
        yaxis=dict(tickformat='$,.0s')
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Comparing {selected_institution} to 8 institutions of similar size nationally")
    st.markdown("---")

def render_funding_breakdown(breakdown_df, trend_df, national_median, end_year):
    if breakdown_df.empty:
        st.warning("No funding data available")
        return
    
    st.subheader("Funding Source Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        row = breakdown_df.iloc[0]
        sources = {
            'Federal': int(row['federal']),
            'Institutional': int(row['institutional']),
            'State/Local': int(row['state_local']),
            'Business': int(row['business']),
            'Nonprofit': int(row['nonprofit']),
            'Other': int(row['other_sources'])
        }
        
        fig_pie = px.pie(
            values=list(sources.values()),
            names=list(sources.keys()),
            title=f'{end_year} Funding Sources'
        )
        fig_pie.update_traces(textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        fig_line = px.line(
            trend_df,
            x='year',
            y='federal_pct',
            title='Federal Dependency Over Time',
            markers=True
        )
        fig_line.add_hline(
            y=national_median,
            line_dash='dash',
            line_color='red',
            annotation_text=f'National Median ({national_median}%)'
        )
        fig_line.update_layout(
            xaxis_title='Year',
            yaxis_title='Federal %',
            yaxis_range=[0, 100]
        )
        st.plotly_chart(fig_line, use_container_width=True)
    
    latest_federal = trend_df.iloc[-1]['federal_pct']
    if latest_federal > 70:
        risk = "HIGH RISK - Heavy federal dependence (>70%)"
        color = "error"
    elif latest_federal > 60:
        risk = "MODERATE RISK - Significant federal dependence (>60%)"
        color = "warning"
    else:
        risk = "LOW RISK - Diversified funding base"
        color = "success"
    
    if color == "error":
        st.error(f"⚠️ {risk}")
    elif color == "warning":
        st.warning(f"⚠️ {risk}")
    else:
        st.success(f"✓ {risk}")
    
    st.markdown("---")

def render_state_ranking(state_df, rank, market_share, state_name, selected_institution):
    if state_df.empty:
        st.warning("No state data available")
        return
    
    st.subheader(f"{state_name} Competitive Position")
    
    c1, c2 = st.columns(2)
    with c1:
        st.metric("State Rank", f"#{rank}")
    with c2:
        st.metric("State Market Share", f"{market_share}%")
    
    display_df = state_df.head(10).copy()
    display_df['total_rd'] = display_df['total_rd'].apply(lambda x: fmt_dollars(x))
    display_df['cagr'] = display_df['cagr'].apply(lambda x: f"{x}%")
    display_df = display_df[['state_rank', 'name', 'total_rd', 'cagr']]
    display_df.columns = ['Rank', 'Institution', '2024 R&D', '5-Yr CAGR']
    
    def highlight_target(row):
        return ['background-color: #EFF6FF' if row['Institution'] == selected_institution else '' for _ in row]
    
    st.dataframe(
        display_df.style.apply(highlight_target, axis=1),
        use_container_width=True,
        hide_index=True
    )
    
    st.caption(f"Top 10 institutions in {state_name} by R&D expenditure")
    st.markdown("---")

def render_snapshot_tab():
    institution_list = load_institution_list()

    col_pick, col_window = st.columns([2, 1])
    with col_pick:
        selected_institution = st.selectbox(
            "Pick an institution",
            options=institution_list,
            index=None
        )
    with col_window:
        time_window = st.selectbox(
            "Time window",
            options=["5-Year (2019–2024)", "10-Year (2014–2024)"],
            index=0
        )

    if not selected_institution or selected_institution == "":
        st.info("Select an institution above to view their R&D funding snapshot")
        return

    start_year = 2019 if "5-Year" in time_window else 2014
    end_year   = 2024

    rank_df = engine.get_rank_trend(selected_institution, start_year, end_year)
    if rank_df.empty:
        st.warning(f"No data found for {selected_institution}")
        return

    metrics = engine.get_executive_metrics(selected_institution, start_year, end_year)
    insight = engine.generate_strategic_insight(selected_institution, start_year, end_year)
    
    all_charts = {}
    
    anchor_df, target_rank, total_inst = engine.get_anchor_view(selected_institution, end_year)
    current_rd   = int(rank_df.iloc[-1]['total_rd'])
    current_rank = int(rank_df.iloc[-1]['national_rank'])
    first_rank   = int(rank_df.iloc[0]['national_rank'])
    moved        = first_rank - current_rank
    
    years = rank_df['year'].tolist()
    ranks = rank_df['national_rank'].tolist()
    colors = ['#93C5FD'] * len(years)
    colors[-1] = '#2563EB'
    
    fig_rank = go.Figure()
    fig_rank.add_trace(go.Bar(
        x=ranks,
        y=[str(y) for y in years],
        orientation='h',
        marker_color=colors,
        text=[f"#{r}" for r in ranks],
        textposition='outside',
        textfont=dict(size=14, color='#374151'),
        hovertemplate='Year: %{y}<br>Rank: #%{x}<extra></extra>'
    ))
    max_rank_show = max(ranks) + 15
    fig_rank.update_layout(
        xaxis=dict(range=[max_rank_show, 0], title='National Rank', showgrid=True, gridcolor='#F3F4F6'),
        yaxis=dict(title=None, categoryorder='array', categoryarray=[str(y) for y in years]),
        height=220,
        margin=dict(l=50, r=60, t=10, b=30),
        plot_bgcolor='white',
        paper_bgcolor='white',
    )
    fig_rank.update_xaxes(tickprefix='#')
    all_charts['rank_trend'] = fig_rank
    
    names  = anchor_df['name'].tolist()
    rd     = anchor_df['total_rd'].tolist()
    ranks_anchor  = anchor_df['national_rank'].tolist()
    is_tgt = anchor_df['is_target'].tolist()
    colors_anchor = ['#2563EB' if t else '#9CA3AF' for t in is_tgt]
    labels = [f"#{r}  {fmt_dollars(v)}" for r, v in zip(ranks_anchor, rd)]
    display_names = []
    for n, t in zip(names, is_tgt):
        label = n if len(n) < 38 else n[:35] + "…"
        if t:
            label = f"► {label}"
        display_names.append(label)
    
    fig_anchor = go.Figure()
    fig_anchor.add_trace(go.Bar(
        x=rd,
        y=display_names,
        orientation='h',
        marker_color=colors_anchor,
        text=labels,
        textposition='outside',
        textfont=dict(size=11, color='#374151'),
        hovertemplate='%{y}<br>R&D: %{text}<extra></extra>'
    ))
    fig_anchor.update_layout(
        xaxis=dict(title='Total R&D', showgrid=True, gridcolor='#F3F4F6', tickformat='$,.0f'),
        yaxis=dict(title=None, categoryorder='array', categoryarray=display_names),
        height=max(200, len(names) * 52),
        margin=dict(l=280, r=100, t=10, b=30),
        plot_bgcolor='white',
        paper_bgcolor='white',
    )
    all_charts['anchor_view'] = fig_anchor

    peers_df, peer_stats = engine.get_peer_comparison(selected_institution, start_year, end_year)
    if not peers_df.empty:
        fig_peers = go.Figure()
        for name in peers_df['name'].unique():
            inst_data = peers_df[peers_df['name'] == name]
            is_target = name == selected_institution
            fig_peers.add_trace(go.Scatter(
                x=inst_data['year'],
                y=inst_data['total_rd'],
                mode='lines+markers',
                name=name if is_target else None,
                line=dict(
                    width=3 if is_target else 1,
                    color='#2563EB' if is_target else '#9CA3AF',
                    dash='solid' if is_target else 'dot'
                ),
                marker=dict(size=8 if is_target else 4),
                showlegend=is_target,
                hovertemplate=f'{name}<br>%{{y:$,.0f}}<extra></extra>'
            ))
        fig_peers.update_layout(
            xaxis_title='Year',
            yaxis_title='Total R&D',
            height=400,
            hovermode='x unified',
            plot_bgcolor='white',
            yaxis=dict(tickformat='$,.0s')
        )
        all_charts['peer_comparison'] = fig_peers

    breakdown_df, trend_df, national_median = engine.get_funding_breakdown(selected_institution, start_year, end_year)
    if not breakdown_df.empty:
        row = breakdown_df.iloc[0]
        sources = {
            'Federal': int(row['federal']),
            'Institutional': int(row['institutional']),
            'State/Local': int(row['state_local']),
            'Business': int(row['business']),
            'Nonprofit': int(row['nonprofit']),
            'Other': int(row['other_sources'])
        }
        fig_pie = px.pie(
            values=list(sources.values()),
            names=list(sources.keys()),
            title=f'{end_year} Funding Sources'
        )
        fig_pie.update_traces(textinfo='percent+label')
        all_charts['funding_pie'] = fig_pie
        
        fig_fed = px.line(
            trend_df,
            x='year',
            y='federal_pct',
            title='Federal Dependency Over Time',
            markers=True
        )
        fig_fed.add_hline(
            y=national_median,
            line_dash='dash',
            line_color='red',
            annotation_text=f'National Median ({national_median}%)'
        )
        fig_fed.update_layout(
            xaxis_title='Year',
            yaxis_title='Federal %',
            yaxis_range=[0, 100]
        )
        all_charts['federal_trend'] = fig_fed

    render_executive_summary(metrics, insight, selected_institution, start_year, end_year, all_charts)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Current Rank", f"#{current_rank}", help=f"Out of {total_inst:,} institutions")
    with c2:
        st.metric(f"Total R&D ({end_year})", fmt_dollars(current_rd))
    with c3:
        st.metric(
            f"{end_year - start_year}-Yr Movement",
            f"↑{moved}" if moved > 0 else (f"↓{abs(moved)}" if moved < 0 else "—"),
            delta=moved,
            delta_color="normal"
        )

    st.subheader("Ranking Over Time")
    st.plotly_chart(fig_rank, use_container_width=True)
    st.caption(f"Ranked out of {total_inst:,} institutions nationally")

    st.subheader("Where They Sit Nationally")
    st.plotly_chart(fig_anchor, use_container_width=True)

    if not peers_df.empty:
        render_peer_comparison(peers_df, peer_stats, selected_institution)

    if not breakdown_df.empty:
        render_funding_breakdown(breakdown_df, trend_df, national_median, end_year)

    state_df, state_rank, market_share, state_name = engine.get_state_ranking(selected_institution, end_year, start_year)
    if not state_df.empty:
        render_state_ranking(state_df, state_rank, market_share, state_name, selected_institution)


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

    # Time series → line chart
    if has_year and len(df) > 1 and df['year'].nunique() > 1:
        if has_name and df['name'].nunique() > 1:
            return px.line(df, x='year', y=y_col, color='name',
                           title=f"{y_col} over time", markers=True)
        return px.line(df, x='year', y=y_col,
                       title=f"{y_col} over time", markers=True)

    # Single snapshot with multiple institutions → bar chart
    if has_name and len(df) > 1:
        df_sorted = df.sort_values(by=y_col, ascending=False)
        fig = px.bar(df_sorted, x='name', y=y_col, title=f"{y_col} by institution")
        fig.update_xaxes(tickangle=-45)
        return fig

    return None

# ============================================================
# Free-form Q&A tab: the original chat interface
# ============================================================
def render_qa_tab():
    # Example questions
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

    # Sidebar viz controls - store in session state so it's accessible outside this function
    with st.sidebar:
        st.header("Visualization")
        st.session_state.enable_viz = st.checkbox("Auto-generate charts", value=True)

        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()

    enable_viz = st.session_state.enable_viz

    # Render previous exchanges
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
# Process a new question (extracted from render_qa_tab)
# ============================================================
def process_question(question, enable_viz=True):
    """Process a new Q&A question and add to history"""
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
                    st.warning("Query returned no results. Try rephrasing.")

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

# ============================================================
# Main layout: two tabs at the top, content below
# ============================================================
st.title("NSF HERD Research Intelligence")
st.markdown("Explore university R&D funding across 1,004 institutions (2010–2024)")

# Create tabs - we'll use the selection to determine which tab is active
exec_tab, snapshot_tab, qa_tab = st.tabs([
    "Executive Summary",
    "Institution Snapshot",
    "Ask a Question",
])

with exec_tab:
    st.session_state.active_tab = 'executive_summary'
    render_executive_summary_tab()

with snapshot_tab:
    st.session_state.active_tab = 'snapshot'
    render_snapshot_tab()

with qa_tab:
    st.session_state.active_tab = 'qa'
    render_qa_tab()

# ============================================================
# Chat input - OUTSIDE tabs, only shown when Q&A tab is active
# ============================================================
if st.session_state.active_tab == 'qa':
    # Get viz preference from sidebar if it exists, otherwise default to True
    enable_viz = st.session_state.get('enable_viz', True)
    
    question = st.chat_input("Ask a question about university R&D funding...")
    if question:
        process_question(question, enable_viz)