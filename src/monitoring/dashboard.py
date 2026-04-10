"""Streamlit monitoring dashboard for the Legal Intelligence Engine.

Four-tab layout
---------------
  Tab 1 — Operations Overview   : request volume, latency, error rate, health
  Tab 2 — Retrieval Quality     : similarity trends, citation distribution
  Tab 3 — Evaluation Dashboard  : RAGAS / custom scores, heatmap, history
  Tab 4 — Live Query Tester     : interactive query → API → rendered result

Usage::

    streamlit run src/monitoring/dashboard.py

Environment variables
---------------------
API_URL      Base URL of the running API  (default: http://localhost:8000)
"""

from __future__ import annotations

import json
import os
import re
import urllib.request

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

API_URL: str = os.getenv("API_URL", "http://localhost:8000")
CACHE_TTL: int = 60  # seconds before st.cache_data refetches

_ARTICLE_RE = re.compile(
    r"Art[ií]culo\s+(\d+(?:\s+(?:BIS|TER|QUATER))?)", re.IGNORECASE
)

# ─────────────────────────────────────────────────────────────────────────────
# Page config  (must be the first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Legal Intelligence Engine — Monitor",
    page_icon="⚖️",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# Header row
# ─────────────────────────────────────────────────────────────────────────────

_col_title, _col_btn = st.columns([5, 1])
_col_title.title("⚖️ Legal Intelligence Engine")
_col_title.caption(
    f"API: `{API_URL}` &nbsp;·&nbsp; Cache TTL: {CACHE_TTL}s"
)

if _col_btn.button("🔄 Refresh", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# Live health banner  (shown at the top of every tab)
# ─────────────────────────────────────────────────────────────────────────────


@st.cache_data(ttl=CACHE_TTL)
def _fetch_health(api_url: str) -> dict | None:
    """GET /v1/health.  Returns None on any failure."""
    try:
        with urllib.request.urlopen(f"{api_url}/v1/health", timeout=3) as resp:
            return json.loads(resp.read())
    except Exception:
        return None


_health = _fetch_health(API_URL)

if _health:
    _icon = {"healthy": "✅", "degraded": "⚠️", "unhealthy": "❌"}.get(
        _health.get("status", ""), "❓"
    )
    _uptime_h = _health.get("uptime_seconds", 0) / 3600
    st.info(
        f"{_icon} **{_health.get('status', 'unknown').capitalize()}**"
        f" &nbsp;·&nbsp; Pinecone {'✅' if _health.get('pinecone_connected') else '❌'}"
        f" &nbsp;·&nbsp; Anthropic {'✅' if _health.get('anthropic_connected') else '❌'}"
        f" &nbsp;·&nbsp; Vectors: **{_health.get('index_vector_count', '—'):,}**"
        f" &nbsp;·&nbsp; Uptime: **{_uptime_h:.1f}h**"
        f" &nbsp;·&nbsp; v{_health.get('version', '—')}"
    )
else:
    st.warning(f"API unreachable at `{API_URL}` — health check timed out.")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Cached data loaders
# ─────────────────────────────────────────────────────────────────────────────


@st.cache_data(ttl=CACHE_TTL)
def _load_requests(api_url: str) -> pd.DataFrame:
    """Fetch the request log from GET /v1/requests."""
    try:
        with urllib.request.urlopen(f"{api_url}/v1/requests", timeout=5) as resp:
            rows = json.loads(resp.read())
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=CACHE_TTL)
def _load_eval_runs(api_url: str) -> list[dict]:
    """Fetch evaluation run results from GET /v1/eval-runs."""
    try:
        with urllib.request.urlopen(f"{api_url}/v1/eval-runs", timeout=5) as resp:
            return json.loads(resp.read())
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Tab layout
# ─────────────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📊 Operations Overview",
        "🔍 Retrieval Quality",
        "📈 Evaluation Dashboard",
        "🧪 Live Query Tester",
    ]
)

_df_all = _load_requests(API_URL)

# ══════════════════════════════════════════════════════════════════════════════
# Tab 1 — Operations Overview
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    if _df_all.empty:
        st.info(
            "No requests logged yet. "
            "Send a query to `POST /v1/query` to populate this dashboard."
        )
    else:
        df = _df_all.copy()
        cutoff_7d = pd.Timestamp.utcnow() - pd.Timedelta(days=7)
        df_7d = df[df["timestamp"] >= cutoff_7d]

        # ── KPI row ───────────────────────────────────────────────────────────
        total_all = len(df)
        total_7d = len(df_7d)
        n_7d = total_7d or 1  # avoid divide-by-zero
        error_cnt = int((df_7d["status_code"] >= 400).sum())
        error_pct = error_cnt / n_7d * 100
        p50_ms = float(df["total_time_ms"].quantile(0.50))
        p95_ms = float(df["total_time_ms"].quantile(0.95))
        p99_ms = float(df["total_time_ms"].quantile(0.99))
        avg_sim = float(df["avg_similarity"].mean())
        halluc_pct = float(df["hallucinations_detected"].mean()) * 100

        k1, k2, k3, k4, k5, k6, k7 = st.columns(7)
        k1.metric("Total Queries", f"{total_all:,}", delta=f"+{total_7d:,} last 7d")
        k2.metric("Error Rate (7d)", f"{error_pct:.1f}%")
        k3.metric("p50 Latency", f"{p50_ms / 1000:.1f}s")
        k4.metric("p95 Latency", f"{p95_ms / 1000:.1f}s")
        k5.metric("p99 Latency", f"{p99_ms / 1000:.1f}s")
        k6.metric("Avg Similarity", f"{avg_sim:.3f}")
        k7.metric("Hallucination Rate", f"{halluc_pct:.1f}%")

        st.divider()

        # ── Request volume — hourly, last 7 days ──────────────────────────────
        st.subheader("Request Volume — Last 7 Days")
        if df_7d.empty:
            st.caption("No traffic in the last 7 days.")
        else:
            vol_df = df_7d.copy()
            vol_df["hour"] = vol_df["timestamp"].dt.floor("h")
            volume = vol_df.groupby("hour").size().reset_index(name="requests")
            fig_vol = px.line(
                volume,
                x="hour",
                y="requests",
                title="Requests per Hour",
                labels={"hour": "Time (UTC)", "requests": "Requests"},
                markers=True,
            )
            fig_vol.update_traces(line_color="#4F8BF9", marker_color="#4F8BF9")
            st.plotly_chart(fig_vol, use_container_width=True)

        # ── Error rate (daily) + Latency breakdown (stacked hourly mean) ──────
        col_l, col_r = st.columns(2)

        with col_l:
            st.subheader("Daily Error Rate")
            if df_7d.empty:
                st.caption("No data.")
            else:
                err_df = df_7d.copy()
                err_df["day"] = err_df["timestamp"].dt.floor("D")
                day_total = err_df.groupby("day").size().rename("total")
                day_errors = (
                    err_df[err_df["status_code"] >= 400]
                    .groupby("day")
                    .size()
                    .rename("errors")
                )
                daily = (
                    pd.concat([day_total, day_errors], axis=1)
                    .fillna(0)
                    .reset_index()
                )
                daily["error_pct"] = (
                    daily["errors"] / daily["total"] * 100
                ).round(1)
                fig_err = px.bar(
                    daily,
                    x="day",
                    y="error_pct",
                    title="Error Rate per Day (%)",
                    labels={"day": "Date", "error_pct": "Error %"},
                    color="error_pct",
                    color_continuous_scale="RdYlGn_r",
                    range_color=[0, 20],
                )
                st.plotly_chart(fig_err, use_container_width=True)

        with col_r:
            st.subheader("Latency Breakdown — Retrieval vs Generation")
            if df_7d.empty:
                st.caption("No data.")
            else:
                lat_df = df_7d.copy()
                lat_df["hour"] = lat_df["timestamp"].dt.floor("h")
                hourly_lat = (
                    lat_df.groupby("hour")[
                        ["retrieval_time_ms", "generation_time_ms"]
                    ]
                    .mean()
                    .reset_index()
                )
                fig_stack = go.Figure()
                fig_stack.add_trace(
                    go.Bar(
                        x=hourly_lat["hour"],
                        y=hourly_lat["retrieval_time_ms"],
                        name="Retrieval",
                        marker_color="#4F8BF9",
                    )
                )
                fig_stack.add_trace(
                    go.Bar(
                        x=hourly_lat["hour"],
                        y=hourly_lat["generation_time_ms"],
                        name="Generation",
                        marker_color="#FF6B6B",
                    )
                )
                fig_stack.update_layout(
                    barmode="stack",
                    title="Hourly Mean Latency (ms)",
                    xaxis_title="Time (UTC)",
                    yaxis_title="ms",
                    legend_title="Stage",
                )
                st.plotly_chart(fig_stack, use_container_width=True)

        st.divider()

        # ── Latency distribution with p50/p95/p99 markers ────────────────────
        st.subheader("Latency Distribution")
        fig_hist = px.histogram(
            df,
            x="total_time_ms",
            nbins=50,
            title="Response Time Distribution (ms)",
            labels={"total_time_ms": "Total Latency (ms)", "count": "Requests"},
            color_discrete_sequence=["#4F8BF9"],
        )
        for q_ms, label, color in [
            (p50_ms, f"p50 ({p50_ms / 1000:.1f}s)", "green"),
            (p95_ms, f"p95 ({p95_ms / 1000:.1f}s)", "orange"),
            (p99_ms, f"p99 ({p99_ms / 1000:.1f}s)", "red"),
        ]:
            fig_hist.add_vline(
                x=q_ms,
                line_dash="dash",
                line_color=color,
                annotation_text=label,
                annotation_position="top right",
            )
        st.plotly_chart(fig_hist, use_container_width=True)

        st.divider()

        # ── Recent queries table ──────────────────────────────────────────────
        st.subheader("Recent Queries (last 50)")
        display = df.head(50)[
            [
                "timestamp",
                "question",
                "total_time_ms",
                "avg_similarity",
                "num_citations",
                "hallucinations_detected",
                "status_code",
            ]
        ].copy()
        display["latency_s"] = (display["total_time_ms"] / 1000).round(1)
        display["avg_similarity"] = display["avg_similarity"].round(3)
        display["hallucinated"] = display["hallucinations_detected"].map(
            {0: "", 1: "⚠️"}
        )
        display = display.drop(columns=["total_time_ms", "hallucinations_detected"])
        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "timestamp": st.column_config.DatetimeColumn(
                    "Time (UTC)", format="YYYY-MM-DD HH:mm:ss"
                ),
                "question": st.column_config.TextColumn("Question", width="large"),
                "latency_s": st.column_config.NumberColumn(
                    "Latency (s)", format="%.1f"
                ),
                "avg_similarity": st.column_config.NumberColumn(
                    "Similarity", format="%.3f"
                ),
                "num_citations": st.column_config.NumberColumn("Citations"),
                "hallucinated": st.column_config.TextColumn("Halluc."),
                "status_code": st.column_config.NumberColumn("Status"),
            },
        )

# ══════════════════════════════════════════════════════════════════════════════
# Tab 2 — Retrieval Quality
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    if _df_all.empty:
        st.info(
            "No requests logged yet. "
            "Send a query to `POST /v1/query` to populate this dashboard."
        )
    else:
        df = _df_all.copy().sort_values("timestamp")

        # ── Similarity trend + Citation count distribution ────────────────────
        col_l, col_r = st.columns(2)

        with col_l:
            st.subheader("Average Similarity Score Over Time")
            sim_df = df.copy()
            sim_df["hour"] = sim_df["timestamp"].dt.floor("h")
            sim_hourly = (
                sim_df.groupby("hour")["avg_similarity"].mean().reset_index()
            )
            overall_mean = float(df["avg_similarity"].mean())
            fig_sim = px.line(
                sim_hourly,
                x="hour",
                y="avg_similarity",
                title="Hourly Mean Cosine Similarity",
                labels={"hour": "Time (UTC)", "avg_similarity": "Cosine Similarity"},
                markers=True,
            )
            fig_sim.update_traces(line_color="#22C55E", marker_color="#22C55E")
            fig_sim.add_hline(
                y=overall_mean,
                line_dash="dot",
                line_color="gray",
                annotation_text=f"Mean {overall_mean:.3f}",
                annotation_position="bottom right",
            )
            fig_sim.update_yaxes(range=[0, 1])
            st.plotly_chart(fig_sim, use_container_width=True)

        with col_r:
            st.subheader("Citations per Response")
            max_cit = int(df["num_citations"].max()) if not df.empty else 0
            fig_cit = px.histogram(
                df,
                x="num_citations",
                nbins=max(max_cit + 1, 1),
                title="Citation Count Distribution",
                labels={
                    "num_citations": "Citations per Response",
                    "count": "Requests",
                },
                color_discrete_sequence=["#A855F7"],
            )
            fig_cit.update_layout(bargap=0.15)
            st.plotly_chart(fig_cit, use_container_width=True)

        # ── Hallucination rate trend + Top referenced articles ────────────────
        col_l2, col_r2 = st.columns(2)

        with col_l2:
            st.subheader("Daily Hallucination Rate")
            hall_df = df.copy()
            hall_df["day"] = hall_df["timestamp"].dt.floor("D")
            hall_daily = (
                hall_df.groupby("day")["hallucinations_detected"]
                .mean()
                .mul(100)
                .reset_index()
                .rename(columns={"hallucinations_detected": "hallucination_pct"})
            )
            overall_hall = float(df["hallucinations_detected"].mean() * 100)
            fig_hall = px.line(
                hall_daily,
                x="day",
                y="hallucination_pct",
                title="Hallucination Rate per Day (%)",
                labels={"day": "Date", "hallucination_pct": "Hallucination %"},
                markers=True,
            )
            fig_hall.update_traces(line_color="#EF4444", marker_color="#EF4444")
            fig_hall.add_hline(
                y=overall_hall,
                line_dash="dot",
                line_color="gray",
                annotation_text=f"Mean {overall_hall:.1f}%",
                annotation_position="bottom right",
            )
            fig_hall.update_yaxes(range=[0, 100])
            st.plotly_chart(fig_hall, use_container_width=True)

        with col_r2:
            st.subheader("Top Referenced Articles (from Answers)")
            article_counts: dict[str, int] = {}
            for answer in df["answer"].dropna():
                for match in _ARTICLE_RE.findall(str(answer)):
                    key = re.sub(r"\s+", " ", match.strip().upper())
                    article_counts[key] = article_counts.get(key, 0) + 1

            if article_counts:
                art_df = (
                    pd.DataFrame.from_dict(
                        article_counts, orient="index", columns=["count"]
                    )
                    .reset_index()
                    .rename(columns={"index": "article"})
                    .sort_values("count", ascending=False)
                    .head(15)
                )
                fig_art = px.bar(
                    art_df,
                    x="article",
                    y="count",
                    title="Top 15 Articles Referenced in Answers",
                    labels={"article": "Article", "count": "Times Referenced"},
                    color="count",
                    color_continuous_scale="Blues",
                )
                fig_art.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_art, use_container_width=True)
            else:
                st.caption(
                    "No article references extracted from answers yet. "
                    "The pattern looks for 'Artículo NNN' in the answer text."
                )

# ══════════════════════════════════════════════════════════════════════════════
# Tab 3 — Evaluation Dashboard
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    _eval_runs = _load_eval_runs(API_URL)

    if not _eval_runs:
        st.info(
            "No evaluation results found. "
            "Run `python scripts/evaluate.py` to generate them."
        )
    else:
        # ── Run selector ─────────────────────────────────────────────────────
        run_labels = [
            f"{r.get('run_id', '?')}  ·  {r.get('timestamp', '')[:10]}  ({r['_file']})"
            for r in _eval_runs
        ]
        sel_idx = st.selectbox(
            "Evaluation run",
            range(len(run_labels)),
            index=len(run_labels) - 1,
            format_func=lambda i: run_labels[i],
        )
        run = _eval_runs[sel_idx]
        agg = run.get("aggregate_scores", {})
        results = run.get("results", [])
        res_df = pd.DataFrame(results) if results else pd.DataFrame()
        cfg = run.get("pipeline_config", {})

        st.caption(
            f"Run `{run.get('run_id')}` &nbsp;·&nbsp; "
            f"{agg.get('n_successful', '?')}/{agg.get('n_total', '?')} successful &nbsp;·&nbsp; "
            f"Model: {cfg.get('model', '—')} &nbsp;·&nbsp; "
            f"top_k: {cfg.get('top_k_final', '—')}"
        )

        # ── Aggregate score cards ─────────────────────────────────────────────
        st.subheader("Aggregate Scores")

        def _fmt(v: float | None) -> str:
            return f"{v:.3f}" if v is not None else "—"

        sc1, sc2, sc3, sc4, sc5, sc6 = st.columns(6)
        sc1.metric("Faithfulness", _fmt(agg.get("faithfulness")))
        sc2.metric("Answer Relevance", _fmt(agg.get("answer_relevance")))
        sc3.metric("Context Relevance", _fmt(agg.get("context_relevance")))
        sc4.metric("Context Recall", _fmt(agg.get("context_recall")))
        sc5.metric("Citation Accuracy", _fmt(agg.get("citation_accuracy")))
        sc6.metric("Hit Rate", _fmt(agg.get("retrieval_hit_rate")))

        # Secondary row: latency + error counts
        la1, la2, la3, la4, la5 = st.columns(5)
        la1.metric("Mean Total Latency", f"{agg.get('mean_total_time_ms', 0):.0f} ms")
        la2.metric("Mean Retrieval", f"{agg.get('mean_retrieval_time_ms', 0):.0f} ms")
        la3.metric("Mean Generation", f"{agg.get('mean_generation_time_ms', 0):.0f} ms")
        la4.metric("Hallucination Rate", _fmt(agg.get("hallucination_rate")))
        la5.metric("Errors", str(agg.get("n_errors", 0)))

        st.divider()

        if not res_df.empty:
            col_heat, col_type = st.columns(2)

            # ── Heatmap: category × difficulty → citation_accuracy ────────────
            with col_heat:
                st.subheader("Score Heatmap: Category × Difficulty")
                needed = {"category", "difficulty", "citation_accuracy"}
                if needed.issubset(res_df.columns):
                    pivot = (
                        res_df.dropna(subset=list(needed))
                        .groupby(["category", "difficulty"])["citation_accuracy"]
                        .mean()
                        .unstack()
                    )
                    # Keep canonical column order
                    ordered = [c for c in ["easy", "medium", "hard"] if c in pivot.columns]
                    if ordered:
                        pivot = pivot[ordered]
                    try:
                        fig_hm = px.imshow(
                            pivot,
                            color_continuous_scale="RdYlGn",
                            zmin=0,
                            zmax=1,
                            title="Mean Citation Accuracy",
                            text_auto=".2f",
                            aspect="auto",
                        )
                        fig_hm.update_layout(
                            xaxis_title="Difficulty",
                            yaxis_title="Category",
                            coloraxis_colorbar_title="Score",
                        )
                        st.plotly_chart(fig_hm, use_container_width=True)
                    except Exception as e:
                        st.caption(f"Heatmap error: {e}")
                else:
                    st.caption(
                        "Heatmap requires `category`, `difficulty`, and "
                        "`citation_accuracy` columns in the eval results."
                    )

            # ── Scores by question type ───────────────────────────────────────
            with col_type:
                st.subheader("Scores by Question Type")
                qt_cols = {"question_type", "citation_accuracy", "retrieval_hit_rate"}
                if qt_cols.issubset(res_df.columns):
                    type_agg = (
                        res_df.groupby("question_type")[
                            ["citation_accuracy", "retrieval_hit_rate"]
                        ]
                        .mean()
                        .reset_index()
                    )
                    type_long = type_agg.melt(
                        id_vars="question_type",
                        value_vars=["citation_accuracy", "retrieval_hit_rate"],
                        var_name="metric",
                        value_name="score",
                    )
                    fig_type = px.bar(
                        type_long,
                        x="question_type",
                        y="score",
                        color="metric",
                        barmode="group",
                        title="Citation Accuracy & Hit Rate by Question Type",
                        labels={
                            "question_type": "Question Type",
                            "score": "Score",
                            "metric": "Metric",
                        },
                        color_discrete_map={
                            "citation_accuracy": "#22C55E",
                            "retrieval_hit_rate": "#4F8BF9",
                        },
                    )
                    fig_type.update_yaxes(range=[0, 1])
                    st.plotly_chart(fig_type, use_container_width=True)

        st.divider()

        # ── Historical trend (visible only when ≥2 runs exist) ───────────────
        if len(_eval_runs) >= 2:
            st.subheader("Score Trends Across Evaluation Runs")
            trend_rows = []
            for r in _eval_runs:
                a = r.get("aggregate_scores", {})
                trend_rows.append(
                    {
                        "run_id": r.get("run_id", "?"),
                        "date": r.get("timestamp", "")[:10],
                        "Faithfulness": a.get("faithfulness"),
                        "Answer Relevance": a.get("answer_relevance"),
                        "Citation Accuracy": a.get("citation_accuracy"),
                        "Hit Rate": a.get("retrieval_hit_rate"),
                    }
                )
            trend_df = pd.DataFrame(trend_rows)
            # Use run_id as x-axis label so runs with the same date are distinct
            trend_long = trend_df.melt(
                id_vars=["run_id", "date"],
                value_vars=[
                    "Faithfulness",
                    "Answer Relevance",
                    "Citation Accuracy",
                    "Hit Rate",
                ],
                var_name="Metric",
                value_name="Score",
            ).dropna(subset=["Score"])
            if not trend_long.empty:
                fig_trend = px.line(
                    trend_long,
                    x="run_id",
                    y="Score",
                    color="Metric",
                    markers=True,
                    title="Eval Metrics Over Runs",
                    labels={"run_id": "Run ID", "Score": "Score"},
                    range_y=[0, 1],
                )
                st.plotly_chart(fig_trend, use_container_width=True)

        # ── Failure explorer ──────────────────────────────────────────────────
        if not res_df.empty and "citation_accuracy" in res_df.columns:
            st.subheader("Failure Explorer")
            threshold = st.slider(
                "Show questions with citation accuracy below:",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.05,
            )
            fail_mask = res_df["citation_accuracy"].fillna(0) < threshold
            if "error" in res_df.columns:
                fail_mask &= res_df["error"].isna()
            fail_df = res_df[fail_mask].copy()

            st.caption(
                f"{len(fail_df)} question(s) below threshold "
                f"(citation_accuracy < {threshold:.2f})"
            )

            if not fail_df.empty:
                show_cols = [
                    c
                    for c in [
                        "question_id",
                        "question",
                        "category",
                        "difficulty",
                        "question_type",
                        "citation_accuracy",
                        "retrieval_hit_rate",
                        "hallucination_detected",
                        "total_time_ms",
                    ]
                    if c in fail_df.columns
                ]
                st.dataframe(
                    fail_df[show_cols].sort_values("citation_accuracy"),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "question": st.column_config.TextColumn(
                            "Question", width="large"
                        ),
                        "citation_accuracy": st.column_config.NumberColumn(
                            "Cite Acc", format="%.3f"
                        ),
                        "retrieval_hit_rate": st.column_config.NumberColumn(
                            "Hit Rate", format="%.3f"
                        ),
                        "total_time_ms": st.column_config.NumberColumn(
                            "Latency (ms)", format="%.0f"
                        ),
                    },
                )

                # Full-context drill-down for a selected failing question
                st.subheader("Drill-Down")
                if "question_id" in fail_df.columns:
                    sel_qid = st.selectbox(
                        "Question ID", fail_df["question_id"].tolist()
                    )
                    row = fail_df[fail_df["question_id"] == sel_qid].iloc[0]
                    st.markdown(f"**Question:** {row.get('question', '—')}")
                    exp_arts = row.get("expected_articles", [])
                    if isinstance(exp_arts, list):
                        st.markdown(
                            f"**Expected articles:** {', '.join(map(str, exp_arts))}"
                        )
                    with st.expander("Model answer"):
                        st.write(row.get("actual_answer", "—"))
                    with st.expander("Retrieved chunks (top 5)"):
                        chunks = row.get("retrieved_chunks", [])
                        if not isinstance(chunks, list):
                            chunks = []
                        for i, chunk in enumerate(chunks[:5], 1):
                            meta = chunk.get("metadata", {}) if isinstance(chunk, dict) else {}
                            st.markdown(
                                f"**Chunk {i}** — Art. {meta.get('article_number', '?')} "
                                f"· similarity: `{chunk.get('similarity_score', 0):.3f}` "
                                f"· rerank: `{chunk.get('rerank_score', 0):.2f}`"
                            )
                            st.text(str(chunk.get("text", ""))[:500])
                        if not chunks:
                            st.caption("No chunks stored for this result.")

# ══════════════════════════════════════════════════════════════════════════════
# Tab 4 — Live Query Tester
# ══════════════════════════════════════════════════════════════════════════════

with tab4:
    st.subheader("Live Query Tester")
    st.caption(
        f"Sends `POST {API_URL}/v1/query` — the API must be running locally "
        f"(`uvicorn src.api.main:app --reload`)."
    )

    question = st.text_input(
        "Question about the Mexican Penal Code (CDMX)",
        placeholder="¿Cuál es la pena por homicidio doloso en el CDMX?",
    )

    if st.button("▶ Run Query", disabled=not question.strip(), type="primary"):
        with st.spinner("Querying the RAG pipeline…"):
            _result: dict | None = None
            _err: str | None = None
            try:
                payload = json.dumps({"question": question.strip()}).encode()
                req = urllib.request.Request(
                    f"{API_URL}/v1/query",
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=60) as resp:
                    _result = json.loads(resp.read())
            except Exception as exc:
                _err = str(exc)

        if _err:
            st.error(f"Request failed: {_err}")
        elif _result:
            # ── Answer ────────────────────────────────────────────────────────
            st.markdown("### Answer")
            st.markdown(_result.get("answer", "—"))

            # ── Timing breakdown ──────────────────────────────────────────────
            st.markdown("### Timing")
            t1, t2, t3 = st.columns(3)
            retrieval_meta = _result.get("retrieval", {})
            t1.metric(
                "Retrieval", f"{retrieval_meta.get('retrieval_time_ms', 0):.0f} ms"
            )
            t2.metric("Generation", f"{_result.get('generation_time_ms', 0):.0f} ms")
            t3.metric("Total", f"{_result.get('total_time_ms', 0):.0f} ms")

            # ── Citations ──────────────────────────────────────────────────────
            citations = _result.get("citations", [])
            halluc = _result.get("hallucination_flags", [])
            st.markdown(f"### Citations ({len(citations)} grounded)")
            if citations:
                for cit in citations:
                    with st.expander(
                        f"Artículo {cit['article_number']} "
                        f"— confidence {cit['confidence']:.2f}"
                    ):
                        st.write(cit.get("source_text", "—"))
            else:
                st.caption("No grounded citations in this response.")

            if halluc:
                st.warning(
                    "Potential hallucinations — articles cited but absent from "
                    f"retrieved context: **{', '.join(halluc)}**"
                )

            # ── Query expansion ───────────────────────────────────────────────
            expanded = retrieval_meta.get("expanded_queries", [])
            if expanded:
                with st.expander(f"Expanded queries ({len(expanded)} variants)"):
                    for i, q in enumerate(expanded, 1):
                        st.markdown(f"{i}. {q}")

            # ── Retrieval metadata ────────────────────────────────────────────
            st.markdown("### Retrieval Metadata")
            st.json(
                {
                    "chunks_retrieved": retrieval_meta.get("chunks_retrieved"),
                    "chunks_after_rerank": retrieval_meta.get("chunks_after_rerank"),
                    "avg_similarity_score": retrieval_meta.get("avg_similarity_score"),
                    "model": _result.get("model"),
                    "prompt_version": _result.get("prompt_version"),
                }
            )
