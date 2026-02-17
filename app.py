import streamlit as st
import sys
from pathlib import Path
from dotenv import load_dotenv
import os
import pandas as pd
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
PASSWORD       = get_env('PASSWORD')
GOOGLE_SHEET_ID    = get_env('GOOGLE_SHEET_ID')
GOOGLE_SHEETS_CREDS = get_env('GOOGLE_SHEETS_CREDS')
ALLOWED_INSTITUTIONS = get_env('ALLOWED_INSTITUTIONS', '')  # comma-separated inst_ids, empty = all allowed

if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY is not set. Add it to your environment variables.")
    st.stop()

if not PASSWORD:
    st.error("PASSWORD is not set. Add it to your environment variables.")
    st.stop()

# ============================================================
# Authentication
# Institution-based login: VPR selects their university, enters
# shared password. Tracks who is using the tool and restricts
# access when ALLOWED_INSTITUTIONS is set.
# ============================================================
MAX_QUERIES_PER_HOUR = 50

@st.cache_data(ttl=3600)
def load_login_institution_list():
    """Load institution list for login screen.

    Uses a lightweight direct query so we don't depend on
    the full engine being initialized yet.
    """
    import sqlite3
    conn = sqlite3.connect(f"file:{DATABASE_PATH}?mode=ro", uri=True)
    try:
        df = pd.read_sql("""
            SELECT DISTINCT inst_id, name
            FROM institutions
            WHERE year = (SELECT MAX(year) FROM institutions)
            ORDER BY name
        """, conn)
    finally:
        conn.close()
    return df

def check_login():
    if st.session_state.get('logged_in'):
        return

    st.title("NSF HERD Research Intelligence")
    st.markdown("Select your institution to get started.")

    login_df = load_login_institution_list()

    # If ALLOWED_INSTITUTIONS is set, filter the dropdown
    allowed_ids = [x.strip() for x in ALLOWED_INSTITUTIONS.split(',') if x.strip()]
    if allowed_ids:
        login_df = login_df[login_df['inst_id'].isin(allowed_ids)]
        if login_df.empty:
            st.error("No institutions are currently enrolled. Contact the administrator.")
            st.stop()

    institution = st.selectbox(
        "Your Institution",
        options=login_df['name'].tolist(),
        index=None,
        placeholder="Search for your university..."
    )
    password = st.text_input("Password", type="password")

    if st.button("Login", type="primary"):
        if institution and password == PASSWORD:
            inst_row = login_df[login_df['name'] == institution]
            st.session_state.logged_in = True
            st.session_state.username = institution
            st.session_state.user_inst_id = inst_row['inst_id'].values[0]
            st.session_state.user_institution = institution
            st.session_state.query_count = 0
            st.session_state.query_window_start = datetime.now()
            st.rerun()
        elif not institution:
            st.error("Please select your institution.")
        else:
            st.error("Invalid password.")
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

def log_to_sheets(username, question, sql, viewing=None):
    if not GOOGLE_SHEET_ID:
        return
    try:
        client = get_gsheet_client()
        if not client:
            return
        sheet = client.open_by_key(GOOGLE_SHEET_ID).sheet1
        timestamp = datetime.now(ZoneInfo('America/Chicago')).strftime('%Y-%m-%d %H:%M:%S')
        sheet.append_row([timestamp, username, viewing or '', question, sql])
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
if 'pending_question' not in st.session_state:
    st.session_state.pending_question = None

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
# Labels for funding metric display names
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

# ============================================================
# Snapshot tab: the full institution snapshot experience.
# User picks a school and a time window, we show:
#   - 3 stat cards (current rank, R&D, movement)
#   - Rank trend over the window
#   - Anchor view for context
# ============================================================


def render_executive_summary(metrics, insight, selected_institution, start_year, end_year, all_charts, callouts=None):
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
            "Total R&D ({})".format(end_year),
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
    
    # Second row: field and agency callouts from the new tables
    if callouts:
        c4, c5 = st.columns(2)
        with c4:
            if 'top_field' in callouts:
                st.metric("Largest Research Field", callouts['top_field'],
                          delta=f"{callouts['top_field_pct']}% of portfolio",
                          delta_color="off")
        with c5:
            if 'top_agency' in callouts:
                st.metric("Top Federal Agency", callouts['top_agency'],
                          delta=f"{callouts['top_agency_pct']}% of federal",
                          delta_color="off")

    if metrics['target_growth'] > metrics['peer_avg'] + 5:
        status = "Growing faster than peers"
    elif metrics['target_growth'] < metrics['peer_avg'] - 5:
        status = "Growing slower than peers"
    else:
        status = "Growth aligned with peers"
    
    st.info(f"**Peer Position:** {status}")
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
            title='Federal Share Over Time',
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
    st.markdown(
        f"**Federal Share:** {latest_federal:.1f}% — "
        f"National median: {national_median}%"
    )
    
    st.markdown("---")

def render_state_ranking(state_df, rank, market_share, state_name, selected_institution, end_year):
    if state_df.empty:
        st.warning("No state data available")
        return

    st.subheader(f"{state_name} Competitive Position")

    c1, c2 = st.columns(2)
    with c1:
        st.metric("State Rank", f"#{rank}")
    with c2:
        st.metric("State Market Share", f"{market_share}%")

    # ------------------------------------------------------------------
    # State peer band: the 3 institutions above and 3 below in state rank.
    # This answers "Who am I actually competing with for the next state
    # grant cycle?" -- more actionable than seeing the top 10 when you're
    # ranked #18. Includes federal dependency so VPRs can see which
    # competitors are more/less exposed to federal policy shifts.
    # ------------------------------------------------------------------
    band_above = 3
    band_below = 3
    band_start = max(rank - band_above, 1)
    band_end = rank + band_below

    band_df = state_df[
        (state_df['state_rank'] >= band_start)
        & (state_df['state_rank'] <= band_end)
    ].copy()

    if not band_df.empty and len(band_df) > 1:
        st.markdown("**Your Competitive Band**")
        band_display = band_df.copy()
        band_display['total_rd'] = band_display['total_rd'].apply(fmt_dollars)
        band_display['cagr'] = band_display['cagr'].apply(
            lambda x: f"{x}%" if x == x else "N/A"
        )
        band_display['federal_pct'] = band_display['federal_pct'].apply(
            lambda x: f"{x}%" if x == x else "N/A"
        )
        band_display = band_display[['state_rank', 'name', 'total_rd', 'cagr', 'federal_pct']]
        band_display.columns = ['Rank', 'Institution', f'{end_year} R&D', '5-Yr CAGR', 'Federal %']

        def highlight_target(row):
            if row['Institution'] == selected_institution:
                return ['background-color: #EFF6FF'] * len(row)
            return [''] * len(row)

        st.dataframe(
            band_display.style.apply(highlight_target, axis=1),
            use_container_width=True,
            hide_index=True
        )
        st.caption(
            f"Institutions ranked #{band_start}–#{band_end} in {state_name}. "
            f"Federal % shows each institution's federal funding share."
        )

    # ------------------------------------------------------------------
    # Top 10 state leaders (the existing table, now in an expander so it
    # doesn't dominate the page when the user's institution isn't in it).
    # ------------------------------------------------------------------
    with st.expander(f"Top 10 in {state_name}"):
        display_df = state_df.head(10).copy()
        display_df['total_rd'] = display_df['total_rd'].apply(fmt_dollars)
        display_df['cagr'] = display_df['cagr'].apply(
            lambda x: f"{x}%" if x == x else "N/A"
        )
        display_df = display_df[['state_rank', 'name', 'total_rd', 'cagr']]
        display_df.columns = ['Rank', 'Institution', f'{end_year} R&D', '5-Yr CAGR']
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    
    display_df = state_df.head(10).copy()
    display_df['total_rd'] = display_df['total_rd'].apply(lambda x: fmt_dollars(x))
    display_df['cagr'] = display_df['cagr'].apply(lambda x: f"{x}%" if x == x else "N/A")
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

def render_snapshot_tab(selected_institution, start_year, end_year, inst_id):
    """Snapshot tab — now receives selection from top-level picker."""
    max_year = end_year

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
            title='Federal Share Over Time',
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

    # --- Benchmarker: KNN peer analysis ---
    # The benchmarker fits on the latest year only (681 institutions),
    # while the dropdown filters for 5+ years of data (612 institutions).
    # Some institutions may be in one set but not the other, so we handle
    # the mismatch gracefully rather than crashing.
    bench_gap = None
    bench_peers = None
    bench_trend_df = None
    bench_trend_stats = None
    bench_data = benchmarker.data  # .data returns a copy (safe to use)
    match = bench_data[bench_data["name"] == selected_institution]
    if not match.empty:
        inst_id = match["inst_id"].values[0]
        try:
            bench_gap   = benchmarker.analyze_gap(inst_id)
            bench_peers = benchmarker.get_peers(inst_id)
            bench_trend_df, bench_trend_stats = benchmarker.get_peer_trend(
                inst_id, DATABASE_PATH, start_year, end_year
            )
        except Exception:
            pass  # graceful fallback -- benchmarker sections simply won't render

    # --- Strategic Summary ---
    callouts = engine.get_snapshot_callouts(selected_institution, end_year)
    render_executive_summary(metrics, insight, selected_institution, start_year, end_year, all_charts, callouts)

    # --- Ranking Over Time ---
    st.subheader("Ranking Over Time")
    st.plotly_chart(fig_rank, use_container_width=True)
    st.caption(f"Ranked out of {total_inst:,} institutions nationally")

    # --- Where You Sit Nationally ---
    st.subheader("Where You Sit Nationally")
    st.plotly_chart(fig_anchor, use_container_width=True)

    # --- Unified Peer Analysis ---
    if bench_gap and bench_peers:
        st.subheader("Peer Analysis")
        st.caption(
            f"Compared against your {len(bench_peers)} closest national peers "
            f"(matched across all funding dimensions)"
        )

        # Growth metric cards
        if bench_trend_stats:
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric(
                    "Your Growth (CAGR)",
                    f"{bench_trend_stats['target_cagr']}%",
                )
            with c2:
                st.metric(
                    "Peer Avg Growth",
                    f"{bench_trend_stats['peer_avg_cagr']}%",
                )
            with c3:
                st.metric(
                    "Growth Rank",
                    f"#{bench_trend_stats['growth_rank']} of {bench_trend_stats['total_in_group']}",
                )

        # Two views under sub-tabs: Funding Profile | Growth Over Time
        tab_profile, tab_growth = st.tabs(["Funding Profile", "Growth Over Time"])

        with tab_profile:
            gap_labels = [METRIC_LABELS.get(g["metric"], g["metric"]) for g in bench_gap]
            my_vals    = [g["my_val"] for g in bench_gap]
            avg_vals   = [g["peer_avg"] for g in bench_gap]

            fig_gap = go.Figure()
            fig_gap.add_trace(go.Bar(
                y=gap_labels, x=my_vals, orientation="h",
                name=selected_institution.split(",")[0],
                marker_color="#2563EB",
                text=[fmt_dollars(v) for v in my_vals],
                textposition="outside",
                textfont=dict(size=12),
            ))
            fig_gap.add_trace(go.Bar(
                y=gap_labels, x=avg_vals, orientation="h",
                name="Peer Average",
                marker_color="#D1D5DB",
                text=[fmt_dollars(v) for v in avg_vals],
                textposition="outside",
                textfont=dict(size=12),
            ))
            fig_gap.update_layout(
                barmode="group",
                height=380,
                margin=dict(l=10, r=100, t=10, b=30),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                xaxis=dict(title=None, showgrid=True, gridcolor="#F3F4F6", tickformat="$,.0f"),
                yaxis=dict(title=None, autorange="reversed"),
                plot_bgcolor="white",
                paper_bgcolor="white",
            )
            st.plotly_chart(fig_gap, use_container_width=True)

            with st.expander("Detailed gap numbers"):
                gap_table = []
                for g in bench_gap:
                    label = METRIC_LABELS.get(g["metric"], g["metric"])
                    gap_val = g["gap"]
                    gap_table.append({
                        "Metric": label,
                        "You": fmt_dollars(g["my_val"]),
                        "Peer Avg": fmt_dollars(g["peer_avg"]),
                        "Gap": f"{'+'if gap_val>=0 else ''}{fmt_dollars(abs(gap_val))}",
                    })
                st.dataframe(gap_table, use_container_width=True, hide_index=True)

        with tab_growth:
            if bench_trend_df is not None and not bench_trend_df.empty:
                fig_trend = go.Figure()
                for name in bench_trend_df["name"].unique():
                    inst_data = bench_trend_df[bench_trend_df["name"] == name]
                    is_target = bool(inst_data["is_target"].iloc[0])

                    fig_trend.add_trace(go.Scatter(
                        x=inst_data["year"],
                        y=inst_data["total_rd"],
                        mode="lines+markers",
                        name=name if is_target else None,
                        line=dict(
                            width=3 if is_target else 1,
                            color="#2563EB" if is_target else "#9CA3AF",
                            dash="solid" if is_target else "dot",
                        ),
                        marker=dict(size=8 if is_target else 4),
                        showlegend=is_target,
                        hovertemplate=f"{name}<br>%{{y:$,.0f}}<extra></extra>",
                    ))

                fig_trend.update_layout(
                    xaxis_title="Year",
                    yaxis_title="Total R&D",
                    height=400,
                    hovermode="x unified",
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                    yaxis=dict(tickformat="$,.0s"),
                )
                st.plotly_chart(fig_trend, use_container_width=True)
            else:
                st.info("Historical trend data is not available.")

        # Peer list -- always visible below both sub-tabs
        with st.expander(f"Who are my {len(bench_peers)} peers?"):
            peer_rows = []
            for pname in bench_peers:
                row = bench_data[bench_data["name"] == pname]
                if not row.empty:
                    peer_rows.append({
                        "Institution": pname,
                        "State": row["state"].values[0],
                        "Total R&D": fmt_dollars(float(row["total_rd"].values[0])),
                    })
            st.dataframe(peer_rows, use_container_width=True, hide_index=True)
            st.caption(
                "Peers are identified relative to your institution's funding "
                "profile and may differ when viewed from another institution's "
                "perspective."
            )

        st.markdown("---")
    else:
        # Institution exists in the dropdown but not in the benchmarker,
        # or the benchmarker had an error. Show everything else normally.
        st.info(
            "Peer analysis is not available for this institution. "
            "This can happen when an institution was not included in "
            "the most recent HERD survey year."
        )
        st.markdown("---")

    # --- Funding Source Analysis ---
    if not breakdown_df.empty:
        render_funding_breakdown(breakdown_df, trend_df, national_median, end_year)

    # --- State Competitive Position ---
    state_df, state_rank, market_share, state_name = engine.get_state_ranking(selected_institution, end_year, start_year)
    if not state_df.empty:
        render_state_ranking(state_df, state_rank, market_share, state_name, selected_institution, end_year)


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
# Standalone parents — no sub-field drill-down available
# ============================================================
STANDALONE_PARENTS = {'cs', 'math', 'psychology', 'other_sciences'}

# Short display labels for parent fields
FIELD_SHORT_LABELS = {
    'cs': 'Computer Science',
    'engineering': 'Engineering',
    'geosciences': 'Geosciences',
    'life_sciences': 'Life Sciences',
    'math': 'Math & Statistics',
    'physical_sciences': 'Physical Sciences',
    'psychology': 'Psychology',
    'social_sciences': 'Social Sciences',
    'other_sciences': 'Other Sciences',
    'non_se': 'Non-S&E',
}

# ============================================================
# Research Portfolio tab
# ============================================================
def render_research_portfolio_tab(selected_institution, start_year, end_year, inst_id, peer_inst_ids):
    """Field-level analysis: portfolio composition, momentum, and peer comparison."""

    portfolio = engine.get_field_portfolio(selected_institution, end_year)
    if portfolio.empty:
        st.info("No field-level data available for this institution.")
        return

    # --- Section 1: Portfolio Overview (stacked horizontal bar) ---
    st.subheader(f"Portfolio Overview — FY{end_year}")

    display = portfolio.copy()
    display['label'] = display['field_code'].map(FIELD_SHORT_LABELS).fillna(display['field_name'])

    fig_port = go.Figure()
    fig_port.add_trace(go.Bar(
        y=display['label'],
        x=display['federal'],
        orientation='h',
        name='Federal',
        marker_color='#2563EB',
        hovertemplate='%{y}<br>Federal: $%{x:,.0f}<extra></extra>',
    ))
    fig_port.add_trace(go.Bar(
        y=display['label'],
        x=display['nonfederal'],
        orientation='h',
        name='Nonfederal',
        marker_color='#93C5FD',
        hovertemplate='%{y}<br>Nonfederal: $%{x:,.0f}<extra></extra>',
    ))
    fig_port.update_layout(
        barmode='stack',
        height=max(280, len(display) * 40),
        margin=dict(l=10, r=120, t=10, b=30),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
        xaxis=dict(title='R&D Expenditure', tickformat='$,.0s', showgrid=True, gridcolor='#F3F4F6'),
        yaxis=dict(title=None, autorange='reversed'),
        plot_bgcolor='white',
        paper_bgcolor='white',
    )
    # Add share annotations on the right side
    for i, row in display.iterrows():
        fig_port.add_annotation(
            x=row['total'], y=row['label'],
            text=f"  {fmt_dollars(row['total'])} ({row['portfolio_share']}%)",
            showarrow=False, xanchor='left',
            font=dict(size=11, color='#374151'),
        )
    st.plotly_chart(fig_port, use_container_width=True)

    # --- Section 2: Field Momentum (BCG scatter) ---
    momentum = engine.get_field_momentum(selected_institution, start_year, end_year)
    if not momentum.empty and momentum['cagr'].notna().any():
        st.subheader(f"Field Momentum — {start_year}–{end_year}")
        st.caption(
            f"Each bubble is a research field. **Right** = larger share of portfolio. "
            f"**Up** = faster growth. Fields in the upper-right are your strategic strengths."
        )

        mom_plot = momentum.dropna(subset=['cagr']).copy()
        mom_plot['label'] = mom_plot['field_code'].map(FIELD_SHORT_LABELS).fillna(mom_plot['field_name'])
        mom_plot['size'] = mom_plot['total'].clip(lower=100000)  # min size for tiny fields

        fig_mom = px.scatter(
            mom_plot,
            x='portfolio_share',
            y='cagr',
            size='size',
            text='label',
            size_max=50,
        )
        fig_mom.update_traces(
            marker=dict(color='#2563EB', opacity=0.7, line=dict(width=1, color='white')),
            textposition='top center',
            textfont=dict(size=10),
        )
        # Add quadrant reference lines at medians
        med_x = mom_plot['portfolio_share'].median()
        med_y = mom_plot['cagr'].median()
        fig_mom.add_hline(y=med_y, line_dash='dot', line_color='#D1D5DB')
        fig_mom.add_vline(x=med_x, line_dash='dot', line_color='#D1D5DB')

        # Quadrant labels
        x_range = mom_plot['portfolio_share'].max() - mom_plot['portfolio_share'].min()
        y_range = mom_plot['cagr'].max() - mom_plot['cagr'].min()
        fig_mom.add_annotation(x=med_x + x_range * 0.25, y=med_y + y_range * 0.35,
                               text="Core Strengths", showarrow=False,
                               font=dict(size=10, color='#9CA3AF'), opacity=0.7)
        fig_mom.add_annotation(x=med_x - x_range * 0.25, y=med_y + y_range * 0.35,
                               text="Emerging", showarrow=False,
                               font=dict(size=10, color='#9CA3AF'), opacity=0.7)
        fig_mom.add_annotation(x=med_x + x_range * 0.25, y=med_y - y_range * 0.35,
                               text="Established Base", showarrow=False,
                               font=dict(size=10, color='#9CA3AF'), opacity=0.7)
        fig_mom.add_annotation(x=med_x - x_range * 0.25, y=med_y - y_range * 0.35,
                               text="Low Activity", showarrow=False,
                               font=dict(size=10, color='#9CA3AF'), opacity=0.7)
        fig_mom.update_layout(
            xaxis_title='Portfolio Share (%)',
            yaxis_title=f'{end_year - start_year}-Year CAGR (%)',
            height=420,
            plot_bgcolor='white',
            paper_bgcolor='white',
            showlegend=False,
        )
        st.plotly_chart(fig_mom, use_container_width=True)

        # Growth signal callouts
        fastest = mom_plot.loc[mom_plot['cagr'].idxmax()]
        largest = mom_plot.loc[mom_plot['total'].idxmax()]
        most_fed = portfolio.loc[portfolio['federal_pct'].idxmax()]

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""
            <div style="background: #f8fafc; padding: 15px; border-radius: 6px;">
                <div style="color: #64748b; font-size: 11px; text-transform: uppercase; margin-bottom: 5px;">Fastest Growing</div>
                <div style="color: #0f172a; font-size: 25px; font-weight: 600; margin-bottom: 3px;">{fastest['label']}</div>
                <div style="color: #10b981; font-size: 13px;">↑ {fastest['cagr']}% CAGR</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div style="background: #f8fafc; padding: 15px; border-radius: 6px;">
                <div style="color: #64748b; font-size: 11px; text-transform: uppercase; margin-bottom: 5px;">Largest Field</div>
                <div style="color: #0f172a; font-size: 25px; font-weight: 600; margin-bottom: 3px;">{largest['label']}</div>
                <div style="color: #64748b; font-size: 13px;">↑ {largest['portfolio_share']}% of portfolio</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            fed_label = FIELD_SHORT_LABELS.get(most_fed['field_code'], most_fed['field_name'])
            st.markdown(f"""
            <div style="background: #f8fafc; padding: 15px; border-radius: 6px;">
                <div style="color: #64748b; font-size: 11px; text-transform: uppercase; margin-bottom: 5px;">Most Federal</div>
                <div style="color: #0f172a; font-size: 23px; font-weight: 600; margin-bottom: 3px;">{fed_label}</div>
                <div style="color: #64748b; font-size: 13px;">↑ {most_fed['federal_pct']}% federal</div>
            </div>
            """, unsafe_allow_html=True)

    # --- Section 3: Sub-field Drill-Down ---
    st.subheader(f"Sub-field Drill-Down — FY{end_year}")
    st.caption("Expand a field to see its component disciplines.")

    for _, parent_row in portfolio.iterrows():
        fc = parent_row['field_code']
        if fc in STANDALONE_PARENTS:
            continue  # No sub-fields to show

        drill = engine.get_field_drilldown(selected_institution, end_year, fc)
        if drill.empty:
            continue

        label = FIELD_SHORT_LABELS.get(fc, parent_row['field_name'])
        with st.expander(f"{label} — {fmt_dollars(parent_row['total'])} ({parent_row['portfolio_share']}%)"):
            drill_display = drill.copy()
            # Clean up field names for readability
            parent_prefix = parent_row['field_name'].split(',')[0]
            drill_display['Discipline'] = drill_display['field_name'].str.replace(
                f'{parent_prefix}, ', '', regex=False
            ).str.capitalize()
            drill_display['Total'] = drill_display['total'].apply(fmt_dollars)
            drill_display['Federal %'] = drill_display['federal_pct'].apply(
                lambda x: f"{x}%" if pd.notna(x) else "N/A"
            )
            drill_display['Share of Parent'] = drill_display['share_of_parent'].apply(
                lambda x: f"{x}%" if pd.notna(x) else "N/A"
            )
            st.dataframe(
                drill_display[['Discipline', 'Total', 'Federal %', 'Share of Parent']],
                use_container_width=True,
                hide_index=True
            )

    # --- Section 4: Peer Portfolio Comparison (diverging bar) ---
    if peer_inst_ids:
        field_comp = engine.get_field_peer_comparison(inst_id, peer_inst_ids, end_year)
        if not field_comp.empty:
            st.subheader(f"Portfolio Distinctiveness — FY{end_year}")
            st.caption(
                "How your field mix compares to your 10 nearest peers. "
                "Positive = you invest a larger share than peers."
            )

            fc = field_comp.copy()
            fc['label'] = fc['field_code'].map(FIELD_SHORT_LABELS).fillna(fc['field_name'])
            fc = fc.sort_values('difference')

            colors = ['#2563EB' if d >= 0 else '#93C5FD' for d in fc['difference']]

            fig_div = go.Figure()
            fig_div.add_trace(go.Bar(
                y=fc['label'],
                x=fc['difference'],
                orientation='h',
                marker_color=colors,
                text=[f"{d:+.1f}pp" for d in fc['difference']],
                textposition='outside',
                textfont=dict(size=11),
                hovertemplate='%{y}<br>You: %{customdata[0]:.1f}%<br>Peers: %{customdata[1]:.1f}%<extra></extra>',
                customdata=fc[['your_pct', 'peer_avg_pct']].values,
            ))
            fig_div.add_vline(x=0, line_color='#374151', line_width=1)
            fig_div.update_layout(
                height=max(280, len(fc) * 36),
                margin=dict(l=10, r=80, t=10, b=30),
                xaxis=dict(title='Difference (percentage points)', showgrid=True, gridcolor='#F3F4F6'),
                yaxis=dict(title=None),
                plot_bgcolor='white',
                paper_bgcolor='white',
            )
            st.plotly_chart(fig_div, use_container_width=True)

    st.markdown("---")


# ============================================================
# Federal Landscape tab
# ============================================================
def render_federal_landscape_tab(selected_institution, start_year, end_year, inst_id, peer_inst_ids):
    """Agency-level federal funding analysis: breakdown, trends, positioning."""

    agencies = engine.get_agency_breakdown(selected_institution, end_year)
    if agencies.empty:
        st.info(
            "No federal agency data available for this institution. "
            "This typically means the institution reported zero federal R&D funding."
        )
        return

    # --- Section 1: Agency Breakdown ---
    st.subheader(f"Federal Agency Breakdown — FY{end_year}")

    col_chart, col_table = st.columns([1, 1])

    with col_chart:
        fig_donut = go.Figure(data=[go.Pie(
            labels=agencies['agency_name'],
            values=agencies['amount'],
            hole=0.45,
            marker=dict(colors=[
                '#2563EB', '#3B82F6', '#60A5FA', '#93C5FD',
                '#BFDBFE', '#DBEAFE', '#EFF6FF'
            ]),
            textinfo='percent+label',
            textfont=dict(size=11),
            hovertemplate='%{label}<br>$%{value:,.0f}<br>%{percent}<extra></extra>',
        )])
        fig_donut.update_layout(
            height=350,
            margin=dict(l=10, r=10, t=10, b=10),
            showlegend=False,
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_table:
        table_data = agencies.copy()
        table_data['Amount'] = table_data['amount'].apply(fmt_dollars)
        table_data['Share'] = table_data['pct_of_federal'].apply(lambda x: f"{x}%")
        table_data = table_data.rename(columns={'agency_name': 'Agency'})
        st.dataframe(
            table_data[['Agency', 'Amount', 'Share']],
            use_container_width=True,
            hide_index=True
        )

    # --- Section 2: Diversification Positioning ---
    conc = engine.get_agency_concentration(selected_institution, end_year)
    if conc:
        st.subheader(f"Funding Diversification — FY{end_year}")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Diversification",
                      f"{conc['diversification_score']}%",
                      help="0% = all funding from one agency, 100% = perfectly even across 7 agencies. "
                           "Calculated using the Herfindahl-Hirschman Index (HHI).")
        with c2:
            # Shorten long agency names to prevent truncation
            agency_short = conc['top_agency'].replace('Dept of ', '').replace(' (incl. NIH)', '')
            st.metric("Top Agency",
                      agency_short,
                      delta=f"{conc['top_agency_pct']}% of federal",
                      delta_color="off")
        with c3:
            st.metric("National Position",
                      f"{conc['national_percentile']}th pctl",
                      help=f"Among {conc['total_institutions']} institutions with federal funding. "
                           "Higher percentile = more concentrated top agency.")

    # --- Section 3: Agency Trends ---
    trend = engine.get_agency_trend(selected_institution, start_year, end_year)
    if not trend.empty:
        st.subheader("Agency Funding Trends")
        st.caption(f"Federal funding by agency, {start_year}–{end_year}.")

        # Color palette for agencies
        agency_colors = {
            'DOD': '#1E40AF', 'DOE': '#B45309', 'HHS': '#047857',
            'NASA': '#7C3AED', 'NSF': '#DC2626', 'USDA': '#65A30D',
            'Other agencies': '#6B7280'
        }

        fig_trend = go.Figure()
        for agency_code in trend['agency_code'].unique():
            adf = trend[trend['agency_code'] == agency_code]
            name = adf['agency_name'].iloc[0]
            color = agency_colors.get(agency_code, '#6B7280')
            fig_trend.add_trace(go.Scatter(
                x=adf['year'],
                y=adf['amount'],
                mode='lines+markers',
                name=name,
                line=dict(color=color, width=2),
                marker=dict(size=6),
                hovertemplate=f'{name}<br>%{{x}}: $%{{y:,.0f}}<extra></extra>',
            ))
        fig_trend.update_layout(
            height=400,
            xaxis_title='Year',
            yaxis=dict(title='Funding', tickformat='$,.0s'),
            hovermode='x unified',
            plot_bgcolor='white',
            paper_bgcolor='white',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
        )
        st.plotly_chart(fig_trend, use_container_width=True)

        # Show growth rates per agency
        with st.expander("Agency Growth Summary"):
            growth_rows = []
            for agency_code in trend['agency_code'].unique():
                adf = trend[trend['agency_code'] == agency_code].sort_values('year')
                if len(adf) >= 2:
                    first_val = adf.iloc[0]['amount']
                    last_val = adf.iloc[-1]['amount']
                    years = adf.iloc[-1]['year'] - adf.iloc[0]['year']
                    if first_val > 0 and years > 0:
                        change_pct = round((last_val / first_val - 1) * 100, 0)
                        growth_rows.append({
                            'Agency': adf.iloc[0]['agency_name'],
                            f'{int(adf.iloc[0]["year"])}': fmt_dollars(first_val),
                            f'{int(adf.iloc[-1]["year"])}': fmt_dollars(last_val),
                            'Change': f"{'+' if change_pct >= 0 else ''}{change_pct:.0f}%"
                        })
            if growth_rows:
                st.dataframe(growth_rows, use_container_width=True, hide_index=True)

    # --- Section 4: Peer Agency Comparison (diverging bar) ---
    if peer_inst_ids:
        agency_comp = engine.get_agency_peer_comparison(inst_id, peer_inst_ids, end_year)
        if not agency_comp.empty:
            st.subheader(f"Agency Distinctiveness — FY{end_year}")
            st.caption(
                "How your federal agency mix compares to your 10 nearest peers. "
                "Positive = you rely more on this agency than peers do."
            )

            ac = agency_comp.sort_values('difference')
            colors = ['#2563EB' if d >= 0 else '#93C5FD' for d in ac['difference']]

            fig_adiv = go.Figure()
            fig_adiv.add_trace(go.Bar(
                y=ac['agency_name'],
                x=ac['difference'],
                orientation='h',
                marker_color=colors,
                text=[f"{d:+.1f}pp" for d in ac['difference']],
                textposition='outside',
                textfont=dict(size=11),
                hovertemplate='%{y}<br>You: %{customdata[0]:.1f}%<br>Peers: %{customdata[1]:.1f}%<extra></extra>',
                customdata=ac[['your_pct', 'peer_avg_pct']].values,
            ))
            fig_adiv.add_vline(x=0, line_color='#374151', line_width=1)
            fig_adiv.update_layout(
                height=max(240, len(ac) * 40),
                margin=dict(l=10, r=80, t=10, b=30),
                xaxis=dict(title='Difference (percentage points)', showgrid=True, gridcolor='#F3F4F6'),
                yaxis=dict(title=None),
                plot_bgcolor='white',
                paper_bgcolor='white',
            )
            st.plotly_chart(fig_adiv, use_container_width=True)

    st.markdown("---")


# ============================================================
# Free-form Q&A tab: the original chat interface
# ============================================================
def render_qa_tab(selected_institution=None, state_code=None, start_year=2019, end_year=2024):
    """Enhanced Q&A with contextual suggested questions based on selected institution."""

    # Sidebar viz controls
    with st.sidebar:
        st.header("Visualization")
        st.session_state.enable_viz = st.checkbox("Auto-generate charts", value=True)
        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()

    enable_viz = st.session_state.enable_viz

    # --- Suggested Questions ---
    if selected_institution:
        short_name = selected_institution.split(',')[0]
        state_label = state_code or ''

        st.markdown("**Suggested Questions**")

        q_groups = {
            "How do we compare?": [
                f"Which peer institutions have a larger engineering portfolio share than {short_name} in {end_year}?",
                f"How does {short_name}'s life sciences R&D compare to other {state_label} schools in {end_year}?",
                f"Which institutions in {state_label} grew their total R&D faster than {short_name} from {start_year} to {end_year}?",
            ],
            "Where's the momentum?": [
                f"What are the fastest growing sub-fields at {short_name} from {start_year} to {end_year}?",
                f"Which agencies increased their funding to {short_name} the most from {start_year} to {end_year}?",
                f"How has {short_name}'s federal vs institutional funding balance shifted from {start_year} to {end_year}?",
            ],
            "What's distinctive?": [
                f"What makes {short_name}'s research portfolio different from its peers in {end_year}?",
                f"Which fields does {short_name} invest in more heavily than similar institutions in {end_year}?",
                f"How concentrated is {short_name}'s federal funding compared to other universities in {end_year}?",
            ],
        }

        for group_label, questions in q_groups.items():
            with st.expander(group_label):
                for q in questions:
                    if st.button(q, key=f"sq_{hash(q)}", use_container_width=True):
                        st.session_state['pending_question'] = q
                        st.rerun()
    else:
        with st.expander("Example Questions"):
            generic_questions = [
                "Show top 10 universities by engineering R&D in 2024",
                "Which universities had the fastest R&D growth over the last 5 years?",
                "Compare MIT, Stanford, and Caltech from 2020 to 2024",
                "Which states have the highest total R&D across all institutions?",
                "What are the top 10 universities by NSF funding in 2024?",
                "Which universities get the highest share of DOD funding?",
            ]
            for q in generic_questions:
                if st.button(q, key=f"gq_{hash(q)}", use_container_width=True):
                    st.session_state['pending_question'] = q
                    st.rerun()

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
    # Rate limiting: reset the counter if an hour has passed,
    # otherwise check if the user has exceeded the limit.
    now = datetime.now()
    window_start = st.session_state.get('query_window_start', now)
    if (now - window_start).total_seconds() > 3600:
        st.session_state.query_count = 0
        st.session_state.query_window_start = now

    query_count = st.session_state.get('query_count', 0)
    if query_count >= MAX_QUERIES_PER_HOUR:
        st.error(
            f"You've reached the limit of {MAX_QUERIES_PER_HOUR} queries per hour. "
            "Please wait before asking more questions."
        )
        return

    st.session_state.query_count = query_count + 1

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                sql, results, summary = engine.ask(question)
                viewing = st.session_state.get('current_viewing', '')
                log_to_sheets(st.session_state.username, question, sql, viewing)

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
# Sidebar: Glossary & Methodology
# ============================================================
with st.sidebar:
    st.markdown(f"**Logged in as:** {st.session_state.get('user_institution', 'Unknown')}")
    st.markdown("---")

    with st.expander("How to Read This Dashboard"):
        st.markdown("""
**Glossary**

**CAGR** — Compound Annual Growth Rate. Annualized growth
that smooths year-to-year fluctuations.
Formula: ((end ÷ start)^(1/years) − 1) × 100

**Portfolio Share** — What percentage of total R&D goes to
a given research field.

**Diversification Score** — How evenly federal funding
is spread across 7 agencies. 100% = perfectly even,
0% = all from one agency. Based on the Herfindahl-Hirschman
Index (HHI = sum of squared agency shares), inverted and
scaled to 0–100.

**National Percentile** — Where this institution's top-agency
concentration ranks among all federally-funded institutions.
Higher = more concentrated.

**KNN Peers** — The 10 most similar institutions, identified
using K-Nearest Neighbors on log-transformed funding data
(total R&D + 6 funding sources). Size is intentionally
double-weighted so a \\$300M school is never compared to a \\$3B school.

**Portfolio Distinctiveness** — Your field share minus peer
average share, in percentage points. Positive = you invest
more in this field relative to peers.

**Field Momentum** — Scatter plot showing each field's portfolio
share (x-axis) vs growth rate (y-axis). Fields in the
upper-right are large and growing: your strategic strengths.

---
**Data Source**

NSF Higher Education Research & Development (HERD) Survey,
2010–2024. Covers 1,004 institutions reporting R&D expenditures.
All dollar figures are as-reported (not inflation-adjusted).
""")

# ============================================================
# Main layout: institution picker above all tabs
# ============================================================
st.title("NSF HERD Research Intelligence")
st.markdown("Explore university R&D funding across 1,004 institutions (2010–2024)")

# --- Top-level institution picker + time window ---
institution_list = load_institution_list()
max_year = engine.get_max_year()

# Default to the institution the user logged in as
default_idx = None
user_institution = st.session_state.get('user_institution')
if user_institution and user_institution in institution_list:
    default_idx = institution_list.index(user_institution)

col_pick, col_window = st.columns([2, 1])
with col_pick:
    selected_institution = st.selectbox(
        "Pick an institution",
        options=institution_list,
        index=default_idx,
    )
with col_window:
    time_window = st.selectbox(
        "Time window",
        options=[
            f"5-Year ({max_year - 5}–{max_year})",
            f"10-Year ({max_year - 10}–{max_year})",
        ],
        index=0
    )

# Track what institution is currently being viewed (for logging)
st.session_state['current_viewing'] = selected_institution or ''

start_year = max_year - 5 if "5-Year" in time_window else max_year - 10
end_year = max_year

# Resolve inst_id and peer_inst_ids for the selected institution
inst_id = None
peer_inst_ids = []
state_code = None

if selected_institution:
    bench_data = benchmarker.data
    match = bench_data[bench_data["name"] == selected_institution]
    if not match.empty:
        inst_id = match["inst_id"].values[0]
        state_code = match["state"].values[0]
        try:
            peer_inst_ids = benchmarker.get_peer_inst_ids(inst_id)
        except Exception:
            peer_inst_ids = []

# --- Four tabs ---
tab_snapshot, tab_portfolio, tab_federal, tab_qa = st.tabs([
    "Institution Snapshot",
    "Research Portfolio",
    "Federal Landscape",
    "Ask a Question",
])

with tab_snapshot:
    if selected_institution:
        render_snapshot_tab(selected_institution, start_year, end_year, inst_id)
    else:
        st.info("Select an institution above to view their R&D funding snapshot.")

with tab_portfolio:
    if selected_institution:
        render_research_portfolio_tab(selected_institution, start_year, end_year, inst_id, peer_inst_ids)
    else:
        st.info("Select an institution above to explore their research portfolio.")

with tab_federal:
    if selected_institution:
        render_federal_landscape_tab(selected_institution, start_year, end_year, inst_id, peer_inst_ids)
    else:
        st.info("Select an institution above to analyze their federal funding landscape.")

with tab_qa:
    render_qa_tab(selected_institution, state_code, start_year, end_year)

    # Process pending question from suggested question buttons INSIDE the tab
    # so the response renders in the Q&A tab context.
    enable_viz = st.session_state.get('enable_viz', True)
    pending = st.session_state.pop('pending_question', None)
    if pending:
        process_question(pending, enable_viz)

# ============================================================
# Chat input — OUTSIDE tabs (Streamlit requirement).
# st.chat_input cannot be placed inside st.tabs.
# ============================================================
question = st.chat_input("Ask a question about university R&D funding...")
if question:
    enable_viz = st.session_state.get('enable_viz', True)
    process_question(question, enable_viz)