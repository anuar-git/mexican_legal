#!/usr/bin/env python3
"""Evaluation pipeline CLI — run the golden test set through the RAG pipeline.

Run from the project root:

    # Full evaluation (all 75 questions)
    python scripts/evaluate.py

    # Custom test set path
    python scripts/evaluate.py --test-set data/evaluation/golden_test_set.json

    # Filter by question type
    python scripts/evaluate.py --question-type factual_lookup
    python scripts/evaluate.py --question-type factual_lookup comparison

    # Filter by category
    python scripts/evaluate.py --category homicidio robo

    # Filter by difficulty
    python scripts/evaluate.py --difficulty easy medium

    # Limit to N questions (quick smoke-test)
    python scripts/evaluate.py --limit 5

    # Debug a single question by ID
    python scripts/evaluate.py --question-id q001

    # Skip RAGAS scoring (faster, no Claude judge calls)
    python scripts/evaluate.py --no-ragas

    # Skip markdown report generation
    python scripts/evaluate.py --no-report

    # Save run JSON to a custom path
    python scripts/evaluate.py --output results/my_run.json

    # Compare two previously saved runs
    python scripts/evaluate.py --compare results/eval_abc123.json results/eval_def456.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

# When executed as `python scripts/evaluate.py`, the project root is not
# automatically on sys.path.  Insert it so `src.*` imports resolve correctly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Load .env before any src imports so API keys are available
from dotenv import load_dotenv  # type: ignore[import-untyped]

load_dotenv()

from src.evaluation.evaluator import EvalResult, EvalRun, Evaluator
from src.evaluation.metrics import aggregate, stratified_breakdown
from src.evaluation.ragas_scorer import RAGASScorer
from src.evaluation.reporter import Reporter

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_LOG_FORMAT = "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s"
_LOG_DATE = "%Y-%m-%d %H:%M:%S"


def _configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format=_LOG_FORMAT,
        datefmt=_LOG_DATE,
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    # Suppress chatty third-party loggers unless verbose
    if not verbose:
        for noisy in ("httpx", "httpcore", "urllib3", "sentence_transformers"):
            logging.getLogger(noisy).setLevel(logging.WARNING)


logger = logging.getLogger("evaluate")

# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="evaluate.py",
        description="Run the golden test set through the RAG pipeline and report results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # --- Source / filter arguments -------------------------------------------
    src_group = parser.add_argument_group("Test set & filtering")
    src_group.add_argument(
        "--test-set",
        metavar="PATH",
        default="data/evaluation/golden_test_set.json",
        help="Path to golden_test_set.json (default: %(default)s)",
    )
    src_group.add_argument(
        "--question-id",
        metavar="ID",
        help="Evaluate a single question by its ID (e.g. q001). "
             "Useful for debugging failures.",
    )
    src_group.add_argument(
        "--question-type",
        metavar="TYPE",
        nargs="+",
        dest="question_types",
        help="Only evaluate questions of these type(s). "
             "Choices: factual_lookup comparison multi_article conditional "
             "adversarial_oos adversarial_tricky",
    )
    src_group.add_argument(
        "--difficulty",
        metavar="LEVEL",
        nargs="+",
        dest="difficulties",
        help="Only evaluate questions at these difficulty level(s). "
             "Choices: easy medium hard adversarial",
    )
    src_group.add_argument(
        "--category",
        metavar="CAT",
        nargs="+",
        dest="categories",
        help="Only evaluate questions in these legal category(ies). "
             "E.g. homicidio robo lesiones",
    )
    src_group.add_argument(
        "--limit",
        metavar="N",
        type=int,
        default=None,
        help="Cap the number of questions after filtering (smoke-test mode).",
    )

    # --- Pipeline tuning -----------------------------------------------------
    pipe_group = parser.add_argument_group("Pipeline config")
    pipe_group.add_argument(
        "--top-k-candidates",
        metavar="N",
        type=int,
        default=20,
        help="Pinecone candidates per query variant (default: %(default)s)",
    )
    pipe_group.add_argument(
        "--top-k-final",
        metavar="N",
        type=int,
        default=5,
        help="Chunks kept after reranking (default: %(default)s)",
    )

    # --- Output / scoring options --------------------------------------------
    out_group = parser.add_argument_group("Output & scoring")
    out_group.add_argument(
        "--output",
        metavar="PATH",
        default=None,
        help="Save the EvalRun JSON to this path. "
             "Default: results/eval_{run_id}.json",
    )
    out_group.add_argument(
        "--no-ragas",
        action="store_true",
        help="Skip RAGAS scoring (no Claude judge calls — faster).",
    )
    out_group.add_argument(
        "--no-report",
        action="store_true",
        help="Skip markdown report generation.",
    )
    out_group.add_argument(
        "--report-dir",
        metavar="DIR",
        default="results/reports",
        help="Directory for markdown reports (default: %(default)s)",
    )
    out_group.add_argument(
        "--checkpoint-dir",
        metavar="DIR",
        default="data/processed/eval_checkpoints",
        help="Directory for per-question checkpoints (default: %(default)s)",
    )

    # --- Compare mode ---------------------------------------------------------
    parser.add_argument(
        "--compare",
        metavar="JSON",
        nargs=2,
        help="Compare two saved EvalRun JSON files and print a diff table. "
             "All other arguments are ignored when --compare is used.",
    )

    # --- Verbosity -----------------------------------------------------------
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )

    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Compare mode
# ---------------------------------------------------------------------------


def _fmt_delta(a: Optional[float], b: Optional[float]) -> str:
    """Format the change from run A to run B."""
    if a is None or b is None:
        return "—"
    delta = b - a
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:+.3f}"


def _compare_runs(path_a: str, path_b: str) -> None:
    """Load two EvalRun JSON files and print a metric comparison table."""
    run_a = EvalRun.load(path_a)
    run_b = EvalRun.load(path_b)

    agg_a = aggregate(run_a.results)
    agg_b = aggregate(run_b.results)

    label_a = f"{run_a.run_id} ({run_a.timestamp[:10]})"
    label_b = f"{run_b.run_id} ({run_b.timestamp[:10]})"

    metrics = [
        ("citation_accuracy",       "Citation accuracy",       agg_a.mean_citation_accuracy,    agg_b.mean_citation_accuracy),
        ("retrieval_hit_rate",      "Retrieval hit rate",      agg_a.mean_retrieval_hit_rate,   agg_b.mean_retrieval_hit_rate),
        ("article_coverage_rate",   "Article coverage rate",   agg_a.mean_article_coverage_rate,agg_b.mean_article_coverage_rate),
        ("hallucination_rate",      "Hallucination rate",      agg_a.hallucination_rate,        agg_b.hallucination_rate),
        ("oos_refusal_rate",        "OOS refusal rate",        agg_a.oos_refusal_rate,          agg_b.oos_refusal_rate),
        ("faithfulness",            "Faithfulness (RAGAS)",    agg_a.mean_faithfulness,         agg_b.mean_faithfulness),
        ("answer_relevance",        "Answer relevance (RAGAS)",agg_a.mean_answer_relevance,     agg_b.mean_answer_relevance),
        ("context_relevance",       "Ctx relevance (RAGAS)",   agg_a.mean_context_relevance,    agg_b.mean_context_relevance),
        ("context_recall",          "Ctx recall (RAGAS)",      agg_a.mean_context_recall,       agg_b.mean_context_recall),
        ("mean_total_ms",           "Mean total latency",      agg_a.mean_total_ms,             agg_b.mean_total_ms),
        ("p95_total_ms",            "p95 total latency",       agg_a.p95_total_ms,              agg_b.p95_total_ms),
    ]

    def _f(v: Optional[float]) -> str:
        return "—" if v is None else f"{v:.3f}"

    col_w = max(len(label_a), len(label_b), 24) + 2
    print()
    print("=" * (col_w * 2 + 30))
    print(f"  Run comparison")
    print(f"  A: {label_a}")
    print(f"  B: {label_b}")
    print("=" * (col_w * 2 + 30))
    print(f"  {'Metric':<28}  {'Run A':>10}  {'Run B':>10}  {'Δ (B−A)':>10}")
    print(f"  {'-'*28}  {'-'*10}  {'-'*10}  {'-'*10}")
    for _, label, va, vb in metrics:
        delta = _fmt_delta(va, vb)
        print(f"  {label:<28}  {_f(va):>10}  {_f(vb):>10}  {delta:>10}")
    print()
    print(f"  Questions:  A={agg_a.n_successful}/{agg_a.n_total}  B={agg_b.n_successful}/{agg_b.n_total}")
    print("=" * (col_w * 2 + 30))
    print()


# ---------------------------------------------------------------------------
# Console summary helpers
# ---------------------------------------------------------------------------


def _print_run_summary(run: EvalRun) -> None:
    """Print a concise score table to stdout."""
    agg = aggregate(run.results)
    bd = stratified_breakdown(run.results)

    def _f(v: Optional[float]) -> str:
        return "—" if v is None else f"{v:.3f}"

    def _pct(v: Optional[float]) -> str:
        return "—" if v is None else f"{v * 100:.1f}%"

    print()
    print("=" * 58)
    print(f"  Run: {run.run_id}  |  {run.timestamp[:19]}  |  {agg.n_successful}/{agg.n_total} OK")
    print("=" * 58)
    print(f"  {'Metric':<30}  {'Score':>8}")
    print(f"  {'-'*30}  {'-'*8}")
    print(f"  {'Citation accuracy':<30}  {_f(agg.mean_citation_accuracy):>8}")
    print(f"  {'Retrieval hit rate':<30}  {_f(agg.mean_retrieval_hit_rate):>8}")
    print(f"  {'Article coverage rate':<30}  {_f(agg.mean_article_coverage_rate):>8}")
    print(f"  {'Hallucination rate':<30}  {_pct(agg.hallucination_rate):>8}")
    print(f"  {'OOS correctly refused':<30}  {_pct(agg.oos_refusal_rate):>8}")
    print()
    print(f"  {'Faithfulness (RAGAS)':<30}  {_f(agg.mean_faithfulness):>8}")
    print(f"  {'Answer relevance (RAGAS)':<30}  {_f(agg.mean_answer_relevance):>8}")
    print(f"  {'Context relevance (RAGAS)':<30}  {_f(agg.mean_context_relevance):>8}")
    print(f"  {'Context recall (RAGAS)':<30}  {_f(agg.mean_context_recall):>8}")
    print()
    print(f"  {'Mean total latency':<30}  {agg.mean_total_ms:.0f} ms" if agg.mean_total_ms else f"  {'Mean total latency':<30}  —")
    print(f"  {'p95 total latency':<30}  {agg.p95_total_ms:.0f} ms" if agg.p95_total_ms else f"  {'p95 total latency':<30}  —")
    print("=" * 58)

    # Per question-type breakdown (compact)
    print()
    print("  By question type:")
    print(f"  {'Type':<22}  {'n':>3}  {'CiteAcc':>7}  {'HitRate':>7}  {'Halluc%':>7}")
    print(f"  {'-'*22}  {'-'*3}  {'-'*7}  {'-'*7}  {'-'*7}")
    for qtype, stats in bd["by_question_type"].items():
        n = stats.get("n", 0)
        ca = _f(stats.get("citation_accuracy"))
        hr = _f(stats.get("retrieval_hit_rate"))
        hl = _pct(stats.get("hallucination_rate"))
        print(f"  {qtype:<22}  {n:>3}  {ca:>7}  {hr:>7}  {hl:>7}")
    print()


def _print_question_result(result: EvalResult) -> None:
    """Print the full details of a single question result."""
    ok = result.error is None

    def _f(v) -> str:
        return "—" if v is None else f"{v:.3f}"

    print()
    print("=" * 62)
    print(f"  Question : {result.question_id}  ({result.question_type} / {result.difficulty})")
    print(f"  Category : {result.category}")
    print("=" * 62)
    print()
    print(f"  Q: {result.question}")
    print()
    if result.error:
        print(f"  ERROR: {result.error}")
    else:
        print("  Answer:")
        # Wrap long answer text at ~70 chars
        for line in result.actual_answer.splitlines():
            if not line.strip():
                print()
                continue
            while len(line) > 68:
                print(f"    {line[:68]}")
                line = line[68:]
            print(f"    {line}")
        print()
        print(f"  Expected articles : {', '.join(result.expected_articles) or '(none)'}")
        print(f"  Cited articles    : {', '.join(result.actual_citations) or '(none)'}")
        print()
        print(f"  Citation accuracy : {_f(result.citation_accuracy)}")
        print(f"  Retrieval hit rate: {_f(result.retrieval_hit_rate)}")
        print(f"  Hallucination     : {result.hallucination_detected}")
        print()
        print(f"  Retrieved chunks  : {len(result.retrieved_chunks)}")
        print(f"  Retrieval time    : {result.retrieval_time_ms:.0f} ms")
        print(f"  Generation time   : {result.generation_time_ms:.0f} ms")
        print(f"  Total time        : {result.total_time_ms:.0f} ms")
    print("=" * 62)
    print()


# ---------------------------------------------------------------------------
# Evaluation run orchestration
# ---------------------------------------------------------------------------


def _resolve_question_id_filter(
    test_set_path: Path,
    question_id: str,
) -> tuple[list[str] | None, list[str] | None, list[str] | None, int | None]:
    """Resolve a --question-id flag into an (categories, types, diff, limit) tuple.

    The test set is scanned to find the matching question.  The category,
    question_type, and difficulty of that one question are returned as
    single-element whitelist filters, plus limit=1.

    Returns:
        (categories, question_types, difficulties, limit) to pass to Evaluator.run()
    """
    with test_set_path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    for q in data["questions"]:
        if q["id"] == question_id:
            return (
                [q["category"]],
                [q["question_type"]],
                [q["difficulty"]],
                1,
            )
    raise ValueError(
        f"Question ID {question_id!r} not found in {test_set_path}. "
        "Check the ID and re-run."
    )


def _run_evaluation(args: argparse.Namespace) -> int:
    """Execute the full evaluation flow and return the process exit code."""
    test_set_path = Path(args.test_set)

    # Resolve --question-id into standard filter args
    categories = args.categories
    question_types = args.question_types
    difficulties = args.difficulties
    limit = args.limit

    if args.question_id:
        try:
            categories, question_types, difficulties, limit = _resolve_question_id_filter(
                test_set_path, args.question_id
            )
        except ValueError as exc:
            logger.error("%s", exc)
            return 1

    # ---- Step 1: Run the pipeline ------------------------------------------
    logger.info("Initialising Evaluator …")
    evaluator = Evaluator(
        test_set_path=test_set_path,
        checkpoint_dir=args.checkpoint_dir,
        top_k_candidates=args.top_k_candidates,
        top_k_final=args.top_k_final,
    )

    t0 = time.monotonic()
    try:
        eval_run = evaluator.run(
            limit=limit,
            question_types=question_types,
            difficulties=difficulties,
            categories=categories,
        )
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        logger.warning("Interrupted — partial results may be in the checkpoint dir.")
        return 130

    # ---- Step 2: RAGAS scoring (optional) ----------------------------------
    if not args.no_ragas:
        import os
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            logger.warning(
                "ANTHROPIC_API_KEY not set — skipping RAGAS scoring. "
                "Set the variable or pass --no-ragas to suppress this warning."
            )
        else:
            logger.info("Running RAGAS scoring with Claude Haiku judge …")
            try:
                scorer = RAGASScorer(anthropic_api_key=api_key)
                scorer.score_run(eval_run)
            except ImportError as exc:
                logger.warning(
                    "RAGAS dependencies not installed — skipping scoring. "
                    "Run: pip install 'ragas>=0.2' 'langchain-anthropic>=0.3'\n"
                    "Original error: %s", exc,
                )
            except Exception as exc:
                logger.error("RAGAS scoring failed: %s", exc)
                logger.info("Continuing without RAGAS scores.")

    # ---- Step 3: Save EvalRun JSON -----------------------------------------
    output_path = Path(args.output) if args.output else Path(f"results/eval_{eval_run.run_id}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    eval_run.save(output_path)
    logger.info("EvalRun saved → %s", output_path)

    # ---- Step 4: Generate markdown report (optional) -----------------------
    report_path: Optional[Path] = None
    if not args.no_report:
        try:
            reporter = Reporter(report_dir=Path(args.report_dir))
            report_path = reporter.generate(eval_run)
            logger.info("Report saved → %s", report_path)
        except Exception as exc:
            logger.warning("Report generation failed: %s", exc)

    # ---- Step 5: Console summary -------------------------------------------
    if args.question_id and eval_run.results:
        _print_question_result(eval_run.results[0])
    else:
        _print_run_summary(eval_run)

    elapsed = time.monotonic() - t0
    logger.info("Total wall time: %.1f s", elapsed)

    if report_path:
        print(f"  Report: {report_path}")
    print(f"  JSON:   {output_path}")
    print()

    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    _configure_logging(args.verbose)

    if args.compare:
        path_a, path_b = args.compare
        try:
            _compare_runs(path_a, path_b)
        except FileNotFoundError as exc:
            logger.error("File not found: %s", exc)
            sys.exit(1)
        except Exception as exc:
            logger.error("Compare failed: %s", exc)
            sys.exit(1)
        sys.exit(0)

    sys.exit(_run_evaluation(args))


if __name__ == "__main__":
    main()
