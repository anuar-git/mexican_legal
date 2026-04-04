"""Markdown report generator for the Mexican Penal Code RAG evaluation system.

Produces a human-readable evaluation report from an ``EvalRun`` object and
saves it to ``results/reports/eval_{run_id}.md``.

Report structure:
    1. Header — run metadata, pipeline config summary.
    2. Aggregate scores table — all RAGAS + custom metrics + latency.
    3. Scores by question type — per-stratum citation accuracy, hit rate,
       faithfulness, context recall, and latency.
    4. Scores by difficulty — same metrics grouped by easy / medium / hard.
    5. Failure analysis — worst 5 performers with per-question diagnosis,
       followed by common failure patterns grouped by failure type.

Usage::

    from src.evaluation.reporter import Reporter

    reporter = Reporter()
    report_path = reporter.generate(eval_run)  # saves + returns Path
    print(open(report_path).read())

    # Or get the markdown string without saving:
    md = reporter.render(eval_run)
"""

from __future__ import annotations

import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.evaluation.evaluator import EvalResult, EvalRun
from src.evaluation.metrics import (
    aggregate,
    article_coverage_rate,
    citation_accuracy,
    detect_hallucination,
    oos_correctly_refused,
    retrieval_hit_rate,
    score_result,
    stratified_breakdown,
    worst_performers,
)

_DEFAULT_REPORT_DIR = Path("results/reports")


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _fmt(value: Optional[float], decimals: int = 3) -> str:
    """Format a float for table display; show '—' for None."""
    if value is None:
        return "—"
    return f"{value:.{decimals}f}"


def _pct(value: Optional[float]) -> str:
    """Format a [0,1] float as a percentage string."""
    if value is None:
        return "—"
    return f"{value * 100:.1f}%"


def _ms(value: Optional[float]) -> str:
    """Format a millisecond value."""
    if value is None:
        return "—"
    return f"{value:.0f} ms"


def _bar(value: Optional[float], width: int = 10) -> str:
    """Simple ASCII bar chart for [0,1] values."""
    if value is None:
        return " " * width
    filled = round(value * width)
    return "█" * filled + "░" * (width - filled)


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------


def _header_section(run: EvalRun) -> str:
    ts = run.timestamp[:10] if run.timestamp else "unknown"
    cfg = run.pipeline_config
    model = cfg.get("model", "unknown")
    top_k_candidates = cfg.get("top_k_candidates", "?")
    top_k_final = cfg.get("top_k_final", "?")
    prompt_version = cfg.get("prompt_version", "?")

    lines = [
        f"# Evaluation Report — Run `{run.run_id}`",
        "",
        f"| Field | Value |",
        f"|-------|-------|",
        f"| **Date** | {ts} |",
        f"| **Test set** | {run.test_set_version} |",
        f"| **Model** | {model} |",
        f"| **Prompt version** | {prompt_version} |",
        f"| **top_k_candidates** | {top_k_candidates} |",
        f"| **top_k_final** | {top_k_final} |",
        f"| **Total questions** | {len(run.results)} |",
        "",
    ]
    return "\n".join(lines)


def _aggregate_section(run: EvalRun) -> str:
    agg = aggregate(run.results)

    # Compute p50 (median) total latency inline — not stored on AggregateMetrics
    ok = [r for r in run.results if r.error is None]
    total_ms_vals = [r.total_time_ms for r in ok]
    p50_ms: Optional[float] = (
        round(statistics.median(total_ms_vals), 1) if total_ms_vals else None
    )

    lines = [
        "## Aggregate Scores",
        "",
        "### Run Summary",
        "",
        f"- **Successful:** {agg.n_successful} / {agg.n_total}",
        f"- **Errors:** {agg.n_errors}",
        f"- **Hallucinations detected:** {agg.n_hallucinated}"
        + (
            f" ({_pct(agg.hallucination_rate)} of successful)"
            if agg.hallucination_rate is not None
            else ""
        ),
        f"- **OOS correctly refused:** {agg.n_oos_refused} / {agg.n_oos_questions}"
        + (f" ({_pct(agg.oos_refusal_rate)})" if agg.oos_refusal_rate is not None else ""),
        "",
        "### Custom Metrics",
        "",
        "| Metric | Score | Bar |",
        "|--------|-------|-----|",
        f"| Citation accuracy | {_fmt(agg.mean_citation_accuracy)} | {_bar(agg.mean_citation_accuracy)} |",
        f"| Retrieval hit rate | {_fmt(agg.mean_retrieval_hit_rate)} | {_bar(agg.mean_retrieval_hit_rate)} |",
        f"| Article coverage rate | {_fmt(agg.mean_article_coverage_rate)} | {_bar(agg.mean_article_coverage_rate)} |",
        f"| Hallucination rate | {_fmt(agg.hallucination_rate)} | {_bar(agg.hallucination_rate)} |",
        "",
        "### RAGAS Metrics",
        "",
        "| Metric | Score | Bar |",
        "|--------|-------|-----|",
        f"| Faithfulness | {_fmt(agg.mean_faithfulness)} | {_bar(agg.mean_faithfulness)} |",
        f"| Answer relevance | {_fmt(agg.mean_answer_relevance)} | {_bar(agg.mean_answer_relevance)} |",
        f"| Context relevance | {_fmt(agg.mean_context_relevance)} | {_bar(agg.mean_context_relevance)} |",
        f"| Context recall | {_fmt(agg.mean_context_recall)} | {_bar(agg.mean_context_recall)} |",
        "",
        "### Latency",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Mean retrieval | {_ms(agg.mean_retrieval_ms)} |",
        f"| Mean generation | {_ms(agg.mean_generation_ms)} |",
        f"| Mean total (p50) | {_ms(p50_ms)} |",
        f"| p95 total | {_ms(agg.p95_total_ms)} |",
        "",
    ]
    return "\n".join(lines)


def _stratum_table(breakdown: dict[str, dict], label: str) -> str:
    """Render a stratum dict as a markdown table."""
    header = (
        f"| {label} | n | Cite Acc | Hit Rate | Art Cov | Halluc | "
        f"Faithfulness | Ctx Recall | Avg ms |"
    )
    sep = "|" + "|".join(["-" * max(len(c), 3) for c in header.split("|")[1:-1]]) + "|"

    rows = [header, sep]
    for name, stats in breakdown.items():
        n = stats.get("n", 0)
        rows.append(
            f"| {name} | {n} "
            f"| {_fmt(stats.get('citation_accuracy'))} "
            f"| {_fmt(stats.get('retrieval_hit_rate'))} "
            f"| {_fmt(stats.get('article_coverage_rate'))} "
            f"| {_pct(stats.get('hallucination_rate'))} "
            f"| {_fmt(stats.get('faithfulness'))} "
            f"| {_fmt(stats.get('context_recall'))} "
            f"| {_ms(stats.get('mean_total_ms'))} |"
        )
    return "\n".join(rows)


def _stratified_section(run: EvalRun) -> str:
    bd = stratified_breakdown(run.results)

    lines = [
        "## Scores by Question Type",
        "",
        _stratum_table(bd["by_question_type"], "Question Type"),
        "",
        "## Scores by Difficulty",
        "",
        _stratum_table(bd["by_difficulty"], "Difficulty"),
        "",
        "## Scores by Category",
        "",
        _stratum_table(bd["by_category"], "Category"),
        "",
    ]
    return "\n".join(lines)


def _diagnose(result: EvalResult) -> str:
    """Produce a short one-line failure diagnosis for a single result."""
    metrics = score_result(result)

    parts: list[str] = []

    if result.error:
        return f"Pipeline error: {result.error}"

    if metrics.hallucination_detected:
        bad = ", ".join(metrics.hallucinated_articles[:3])
        suffix = "…" if len(metrics.hallucinated_articles) > 3 else ""
        parts.append(f"hallucinated article(s): {bad}{suffix}")

    if metrics.retrieval_hit_rate < 0.5:
        missing = ", ".join(metrics.missing_articles[:3])
        suffix = "…" if len(metrics.missing_articles) > 3 else ""
        parts.append(f"retriever missed: {missing}{suffix}")
    elif metrics.retrieval_hit_rate >= 0.5 and metrics.article_coverage_rate < 0.5:
        parts.append("generator failed to cite retrieved articles")

    if result.question_type == "adversarial_oos":
        refused = oos_correctly_refused(result)
        if refused is False:
            parts.append("OOS question not refused")

    if not parts:
        # Low score but no specific signal — generic fallback
        ca = citation_accuracy(result)
        hr = retrieval_hit_rate(result)
        if ca < 0.5:
            parts.append(f"low citation accuracy ({ca:.2f})")
        elif hr < 0.5:
            parts.append(f"low retrieval hit rate ({hr:.2f})")
        else:
            parts.append("marginal performance across metrics")

    return "; ".join(parts)


def _failure_analysis_section(run: EvalRun) -> str:
    """Worst 5 performers + common failure pattern groups."""
    ok = [r for r in run.results if r.error is None]
    if not ok:
        return "## Failure Analysis\n\n_No successful results to analyse._\n"

    worst = worst_performers(run.results, metric="citation_accuracy", n=5)

    # ---- Worst 5 performers table ----------------------------------------
    worst_lines = [
        "## Failure Analysis",
        "",
        "### Worst 5 Performers (by citation accuracy)",
        "",
        "| ID | Question (truncated) | Type | Cite Acc | Hit Rate | Diagnosis |",
        "|----|----------------------|------|----------|----------|-----------|",
    ]
    for r in worst:
        q_short = (r.question[:55] + "…") if len(r.question) > 55 else r.question
        q_short = q_short.replace("|", "\\|")
        ca = _fmt(citation_accuracy(r))
        hr = _fmt(retrieval_hit_rate(r))
        diag = _diagnose(r).replace("|", "\\|")
        worst_lines.append(
            f"| {r.question_id} | {q_short} | {r.question_type} | {ca} | {hr} | {diag} |"
        )

    # ---- Common failure patterns ------------------------------------------
    all_metrics = [score_result(r) for r in ok]

    # Group into four buckets
    retrieval_misses = [
        m for m in all_metrics if m.retrieval_hit_rate < 0.5
    ]
    gen_failures = [
        m for m in all_metrics
        if m.retrieval_hit_rate >= 0.5 and m.article_coverage_rate < 0.5
    ]
    hallucinations = [m for m in all_metrics if m.hallucination_detected]
    oos_failures = [
        m for m in all_metrics
        if m.oos_correctly_refused is False
    ]

    pattern_lines = [
        "",
        "### Common Failure Patterns",
        "",
        f"| Pattern | Count | % of successful |",
        f"|---------|-------|-----------------|",
        f"| Retrieval miss (hit rate < 50%) | {len(retrieval_misses)} | {_pct(len(retrieval_misses)/len(ok) if ok else None)} |",
        f"| Generator coverage gap (retrieved but not cited) | {len(gen_failures)} | {_pct(len(gen_failures)/len(ok) if ok else None)} |",
        f"| Hallucinated citations | {len(hallucinations)} | {_pct(len(hallucinations)/len(ok) if ok else None)} |",
        f"| OOS question not refused | {len(oos_failures)} | {_pct(len(oos_failures)/len(ok) if ok else None)} |",
        "",
    ]

    # Detail lists for non-empty buckets
    def _detail_list(label: str, items: list, id_getter) -> list[str]:
        if not items:
            return []
        ids = ", ".join(id_getter(m) for m in items[:10])
        suffix = f" (+{len(items)-10} more)" if len(items) > 10 else ""
        return [f"**{label}:** {ids}{suffix}", ""]

    detail_lines: list[str] = []
    detail_lines += _detail_list(
        "Retrieval misses", retrieval_misses, lambda m: m.question_id
    )
    detail_lines += _detail_list(
        "Coverage gaps", gen_failures, lambda m: m.question_id
    )
    detail_lines += _detail_list(
        "Hallucinations", hallucinations, lambda m: m.question_id
    )
    detail_lines += _detail_list(
        "OOS not refused", oos_failures, lambda m: m.question_id
    )

    return "\n".join(worst_lines + pattern_lines + detail_lines)


def _footer_section(run: EvalRun) -> str:
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines = [
        "---",
        "",
        f"_Report generated at {generated_at} from run `{run.run_id}`._",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class Reporter:
    """Generates a markdown evaluation report from an ``EvalRun``.

    Args:
        report_dir: Directory where reports are saved.  Created automatically
                    if it does not exist.  Defaults to ``results/reports/``.
    """

    def __init__(self, report_dir: Path = _DEFAULT_REPORT_DIR) -> None:
        self._report_dir = Path(report_dir)

    def render(self, eval_run: EvalRun) -> str:
        """Render the full markdown report as a string (does not save).

        Args:
            eval_run: A completed EvalRun (RAGAS scores optional — shown as
                      '—' when absent).

        Returns:
            Markdown string of the complete report.
        """
        sections = [
            _header_section(eval_run),
            _aggregate_section(eval_run),
            _stratified_section(eval_run),
            _failure_analysis_section(eval_run),
            _footer_section(eval_run),
        ]
        return "\n".join(sections)

    def generate(self, eval_run: EvalRun) -> Path:
        """Render the report and save it to ``results/reports/eval_{run_id}.md``.

        The report directory is created automatically if it does not exist.

        Args:
            eval_run: A completed EvalRun.

        Returns:
            Path to the saved report file.
        """
        self._report_dir.mkdir(parents=True, exist_ok=True)
        path = self._report_dir / f"eval_{eval_run.run_id}.md"
        md = self.render(eval_run)
        path.write_text(md, encoding="utf-8")
        return path


def generate_report(
    eval_run: EvalRun,
    report_dir: Path = _DEFAULT_REPORT_DIR,
) -> Path:
    """Convenience function — generate and save a report, return its path.

    Equivalent to::

        Reporter(report_dir=report_dir).generate(eval_run)

    Args:
        eval_run:   A completed EvalRun.
        report_dir: Target directory (default: ``results/reports/``).

    Returns:
        Path to the saved ``.md`` file.
    """
    return Reporter(report_dir=report_dir).generate(eval_run)
