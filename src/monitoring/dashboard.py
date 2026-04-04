"""Streamlit monitoring dashboard for the Legal Intelligence Engine.

Reads historical request data from the SQLite log written by the API server
and polls /v1/health for live service status.

Environment variables
---------------------
DB_PATH   Path to the SQLite database file   (default: data/requests.db)
API_URL   Base URL of the API service        (default: http://api:8000)
"""

from __future__ import annotations

import json
import os
import sqlite3
import urllib.request
from urllib.error import URLError

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DB_PATH: str = os.getenv("DB_PATH", "data/requests.db")
API_URL: str = os.getenv("API_URL", "http://api:8000")
CACHE_TTL: int = 60  # seconds before st.cache_data refetches

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Legal Intelligence Engine — Monitor",
    page_icon="⚖️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Header row: title + refresh button
# ---------------------------------------------------------------------------

col_title, col_btn = st.columns([5, 1])
col_title.title("⚖️ Legal Intelligence Engine")
col_title.caption(f"DB: `{DB_PATH}`  ·  API: `{API_URL}`  ·  Cache TTL: {CACHE_TTL}s")

if col_btn.button("🔄 Refresh", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

# ---------------------------------------------------------------------------
# Live API health banner
# ---------------------------------------------------------------------------


@st.cache_data(ttl=CACHE_TTL)
def fetch_health(api_url: str) -> dict | None:
    """GET /v1/health from the API container.  Returns None on any failure."""
    try:
        with urllib.request.urlopen(f"{api_url}/v1/health", timeout=3) as resp:
            return json.loads(resp.read())
    except (URLError, Exception):
        return None


health = fetch_health(API_URL)

if health:
    icon = {"healthy": "✅", "degraded": "⚠️", "unhealthy": "❌"}.get(
        health.get("status", ""), "❓"
    )
    uptime_h = health.get("uptime_seconds", 0) / 3600
    st.info(
        f"{icon} **{health.get('status', 'unknown').capitalize()}** "
        f"&nbsp;·&nbsp; "
        f"Pinecone {'✅' if health.get('pinecone_connected') else '❌'} "
        f"&nbsp;·&nbsp; "
        f"Anthropic {'✅' if health.get('anthropic_connected') else '❌'} "
        f"&nbsp;·&nbsp; "
        f"Vectors: **{health.get('index_vector_count', '—'):,}** "
        f"&nbsp;·&nbsp; "
        f"Uptime: **{uptime_h:.1f}h** "
        f"&nbsp;·&nbsp; "
        f"v{health.get('version', '—')}"
    )
else:
    st.warning(f"API unreachable at `{API_URL}` — health check timed out.")

st.divider()

# ---------------------------------------------------------------------------
# Load request log from SQLite
# ---------------------------------------------------------------------------


@st.cache_data(ttl=CACHE_TTL)
def load_requests(db_path: str) -> pd.DataFrame:
    """Read the full requests table from SQLite.

    Opens with ``mode=ro`` URI so the dashboard never acquires a write lock on
    the database file that the API is actively writing.
    """
    if not os.path.exists(db_path):
        return pd.DataFrame()
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        df = pd.read_sql_query(
            "SELECT * FROM requests ORDER BY timestamp DESC",
            conn,
        )
        conn.close()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df
    except Exception:
        return pd.DataFrame()


df = load_requests(DB_PATH)

if df.empty:
    st.info(
        "No requests logged yet.  "
        "Send a query to `POST /v1/query` to populate this dashboard."
    )
    st.stop()

# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------

total = len(df)
errors = int((df["status_code"] >= 500).sum())
error_pct = errors / total * 100 if total else 0.0
p50_s = df["total_time_ms"].quantile(0.50) / 1000
p95_s = df["total_time_ms"].quantile(0.95) / 1000
avg_sim = float(df["avg_similarity"].mean())
halluc_pct = float(df["hallucinations_detected"].mean()) * 100

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Total Queries", f"{total:,}")
k2.metric("Error Rate", f"{error_pct:.1f}%")
k3.metric("p50 Latency", f"{p50_s:.1f}s")
k4.metric("p95 Latency", f"{p95_s:.1f}s")
k5.metric("Avg Similarity", f"{avg_sim:.3f}")
k6.metric("Hallucination Rate", f"{halluc_pct:.1f}%")

st.divider()

# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

chart_l, chart_r = st.columns(2)

with chart_l:
    st.subheader("End-to-End Latency (last 200 requests)")
    latency_df = (
        df.sort_values("timestamp")
        .tail(200)[["timestamp", "retrieval_time_ms", "generation_time_ms", "total_time_ms"]]
        .rename(columns={
            "retrieval_time_ms": "retrieval",
            "generation_time_ms": "generation",
            "total_time_ms": "total",
        })
        .set_index("timestamp")
        / 1000  # ms → seconds
    )
    st.line_chart(latency_df, y_label="seconds")

with chart_r:
    st.subheader("Retrieval Similarity (last 200 requests)")
    sim_df = (
        df.sort_values("timestamp")
        .tail(200)[["timestamp", "avg_similarity"]]
        .set_index("timestamp")
    )
    st.line_chart(sim_df, y_label="cosine similarity")

st.divider()

# ---------------------------------------------------------------------------
# Recent queries table
# ---------------------------------------------------------------------------

st.subheader("Recent Queries (last 50)")

display = (
    df.head(50)[[
        "timestamp",
        "question",
        "total_time_ms",
        "avg_similarity",
        "num_citations",
        "hallucinations_detected",
        "status_code",
    ]]
    .copy()
)

display["latency_s"] = (display["total_time_ms"] / 1000).round(1)
display["avg_similarity"] = display["avg_similarity"].round(3)
display["hallucinated"] = display["hallucinations_detected"].map({0: "", 1: "⚠️"})

display = display.drop(columns=["total_time_ms", "hallucinations_detected"])

st.dataframe(
    display,
    use_container_width=True,
    hide_index=True,
    column_config={
        "timestamp": st.column_config.DatetimeColumn("Time (UTC)", format="YYYY-MM-DD HH:mm:ss"),
        "question": st.column_config.TextColumn("Question", width="large"),
        "latency_s": st.column_config.NumberColumn("Latency (s)", format="%.1f"),
        "avg_similarity": st.column_config.NumberColumn("Similarity", format="%.3f"),
        "num_citations": st.column_config.NumberColumn("Citations"),
        "hallucinated": st.column_config.TextColumn("Halluc."),
        "status_code": st.column_config.NumberColumn("Status"),
    },
)
