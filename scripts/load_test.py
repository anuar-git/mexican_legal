"""Load test for the Legal Intelligence Engine API.

Runs two test phases against the configured base URL:

  1. **Sequential** — sends each question one at a time, measuring the
     per-request latency baseline without any server-side concurrency.

  2. **Concurrent** — fires all questions simultaneously (up to
     ``--concurrency`` in-flight at once) to find the throughput ceiling
     and surface latency degradation under load.

After each phase, per-request results and aggregate statistics
(min / avg / p50 / p95 / p99 / max) are printed.  An optional JSON
report can be written to disk with ``--output``.

Usage
-----
Install the extra dependency first::

    pip install -e ".[loadtest]"

Run against a local dev server::

    python scripts/load_test.py

Run against Railway with custom concurrency::

    python scripts/load_test.py \\
        --base-url https://your-app.railway.app \\
        --concurrency 20 \\
        --timeout 90

Skip the sequential warm-up::

    python scripts/load_test.py --no-sequential

Save results to JSON::

    python scripts/load_test.py --output results/load_test.json
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Question bank — 20 queries across distinct legal domains of the CDMX
# Penal Code so that each one exercises a different area of the index.
# ---------------------------------------------------------------------------

QUESTIONS: list[str] = [
    # Delitos contra la vida
    "¿Cuál es la pena por homicidio doloso en el Código Penal del CDMX?",
    "¿Qué diferencia hay entre homicidio culposo y homicidio doloso?",
    # Lesiones
    "¿Cómo se clasifican las lesiones según el Código Penal de la Ciudad de México?",
    # Delitos contra el patrimonio
    "¿Qué artículo regula el robo con violencia en el CDMX?",
    "¿Cuál es la pena por fraude en el Código Penal del CDMX?",
    "¿Qué es el abuso de confianza y cuál es su sanción?",
    "¿Cuáles son los elementos del delito de extorsión?",
    # Delitos sexuales
    "¿Cómo define el código penal la violación y cuál es su pena?",
    "¿Qué artículos regulan el acoso sexual en el CDMX?",
    # Privación de libertad
    "¿Qué es el secuestro y cuál es la pena máxima establecida?",
    "¿Cuándo se configura el delito de privación ilegal de la libertad?",
    # Delitos contra la salud
    "¿Qué penas establece el código penal para el tráfico de drogas?",
    # Violencia familiar
    "¿Cómo define la ley la violencia familiar y qué sanciones aplica?",
    # Delitos informáticos
    "¿Qué delitos informáticos contempla el Código Penal del CDMX?",
    # Responsabilidad y participación
    "¿Cuándo procede la tentativa punible según el código penal?",
    "¿Cómo se determina la participación criminal como cómplice?",
    # Penas y ejecución
    "¿Cuáles son los sustitutivos de la pena de prisión en el CDMX?",
    "¿Cuándo procede la libertad provisional bajo caución?",
    # Prescripción y extinción
    "¿En qué casos se extingue la acción penal por prescripción?",
    # Servidores públicos
    "¿Qué delitos pueden cometer los servidores públicos según el código penal?",
]


# ---------------------------------------------------------------------------
# Data model for a single request outcome
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class RequestResult:
    """Outcome of a single POST /v1/query call."""

    question: str
    latency_s: float
    status_code: int
    ok: bool  # True if status_code == 200
    error: Optional[str] = None  # network or decode error message


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------


async def send_request(
    session,  # aiohttp.ClientSession
    base_url: str,
    question: str,
    top_k: int,
    timeout: float,
) -> RequestResult:
    """POST one question to /v1/query and return the result."""
    url = f"{base_url.rstrip('/')}/v1/query"
    t0 = time.monotonic()
    try:
        import aiohttp  # local import so the rest of the module loads without it

        async with session.post(
            url,
            json={"question": question, "top_k": top_k},
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as resp:
            latency = time.monotonic() - t0
            try:
                await resp.json(content_type=None)
            except Exception:
                pass  # we only need the status code for timing purposes
            return RequestResult(
                question=question,
                latency_s=latency,
                status_code=resp.status,
                ok=resp.status == 200,
            )
    except Exception as exc:
        return RequestResult(
            question=question,
            latency_s=time.monotonic() - t0,
            status_code=0,
            ok=False,
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def compute_stats(latencies: list[float]) -> dict[str, float]:
    """Return summary statistics for a list of latency values (seconds)."""
    if not latencies:
        return {}
    s = sorted(latencies)
    n = len(s)
    return {
        "count": n,
        "min": s[0],
        "avg": statistics.mean(s),
        "p50": s[int(n * 0.50)],
        "p95": s[min(int(n * 0.95), n - 1)],
        "p99": s[min(int(n * 0.99), n - 1)],
        "max": s[-1],
    }


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

_RESET = "\033[0m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_BOLD = "\033[1m"
_CYAN = "\033[36m"


def _status_colour(result: RequestResult) -> str:
    if result.ok:
        return _GREEN
    if result.status_code == 429:
        return _YELLOW
    return _RED


def _print_result(idx: int, result: RequestResult) -> None:
    colour = _status_colour(result)
    status = result.status_code if result.status_code else "ERR"
    question = result.question[:55] + ("…" if len(result.question) > 55 else "")
    suffix = f" [{result.error}]" if result.error else ""
    print(
        f"  {idx:>2}. {colour}{status}{_RESET}  {result.latency_s:6.2f}s  {question}{suffix}"
    )


def _print_stats(stats: dict[str, float]) -> None:
    if not stats:
        print("  (no data)")
        return
    print(
        f"  {_BOLD}count{_RESET}={int(stats['count'])}  "
        f"{_BOLD}min{_RESET}={stats['min']:.2f}s  "
        f"{_BOLD}avg{_RESET}={stats['avg']:.2f}s  "
        f"{_BOLD}p50{_RESET}={stats['p50']:.2f}s  "
        f"{_BOLD}p95{_RESET}={stats['p95']:.2f}s  "
        f"{_BOLD}p99{_RESET}={stats['p99']:.2f}s  "
        f"{_BOLD}max{_RESET}={stats['max']:.2f}s"
    )


def _print_header(title: str) -> None:
    print(f"\n{_BOLD}{_CYAN}{'─' * 65}{_RESET}")
    print(f"{_BOLD}{_CYAN}  {title}{_RESET}")
    print(f"{_BOLD}{_CYAN}{'─' * 65}{_RESET}")


def _error_summary(results: list[RequestResult]) -> None:
    errors = [r for r in results if not r.ok]
    if not errors:
        print(f"  {_GREEN}All requests succeeded.{_RESET}")
        return
    by_code: dict[int, int] = {}
    for r in errors:
        by_code[r.status_code] = by_code.get(r.status_code, 0) + 1
    parts = [f"  {_RED}Errors:{_RESET}"]
    for code, count in sorted(by_code.items()):
        label = {0: "network", 429: "rate_limited", 500: "server_error"}.get(
            code, f"http_{code}"
        )
        parts.append(f"    {code} ({label}): {count}")
    print("\n".join(parts))


# ---------------------------------------------------------------------------
# Test phases
# ---------------------------------------------------------------------------


async def run_sequential(
    session,
    base_url: str,
    questions: list[str],
    top_k: int,
    timeout: float,
) -> list[RequestResult]:
    """Send questions one at a time; print each result as it arrives."""
    _print_header("Sequential (baseline)")
    print(f"  Sending {len(questions)} requests one at a time …\n")
    results: list[RequestResult] = []
    for i, q in enumerate(questions, 1):
        result = await send_request(session, base_url, q, top_k, timeout)
        _print_result(i, result)
        results.append(result)
    return results


async def run_concurrent(
    session,
    base_url: str,
    questions: list[str],
    concurrency: int,
    top_k: int,
    timeout: float,
) -> list[RequestResult]:
    """Send all questions with up to *concurrency* in-flight simultaneously."""
    _print_header(f"Concurrent (concurrency={concurrency})")
    print(
        f"  Sending {len(questions)} requests with up to {concurrency} in-flight …\n"
    )
    semaphore = asyncio.Semaphore(concurrency)

    async def _bounded(q: str) -> RequestResult:
        async with semaphore:
            return await send_request(session, base_url, q, top_k, timeout)

    wall_t0 = time.monotonic()
    results = await asyncio.gather(*[_bounded(q) for q in questions])
    wall_s = time.monotonic() - wall_t0

    for i, r in enumerate(results, 1):
        _print_result(i, r)

    print(f"\n  {_BOLD}Total wall time:{_RESET} {wall_s:.2f}s")
    return list(results)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main(args: argparse.Namespace) -> int:
    """Run load test phases and optionally write results to JSON.

    Returns an exit code (0 = all requests succeeded, 1 = some failures).
    """
    try:
        import aiohttp
    except ImportError:
        print(
            "aiohttp is required for the load test.\n"
            "Install it with:  pip install -e '.[loadtest]'",
            file=sys.stderr,
        )
        return 1

    print(f"\n{_BOLD}Legal Intelligence Engine — Load Test{_RESET}")
    print(f"  base_url    : {args.base_url}")
    print(f"  questions   : {len(QUESTIONS)}")
    print(f"  top_k       : {args.top_k}")
    print(f"  timeout     : {args.timeout}s per request")
    print(f"  concurrency : {args.concurrency}")

    report: dict = {
        "config": {
            "base_url": args.base_url,
            "top_k": args.top_k,
            "timeout": args.timeout,
            "concurrency": args.concurrency,
            "questions": len(QUESTIONS),
        },
        "sequential": None,
        "concurrent": None,
    }

    async with aiohttp.ClientSession() as session:
        # ── Phase 1: sequential ─────────────────────────────────────────────
        if not args.no_sequential:
            seq_results = await run_sequential(
                session, args.base_url, QUESTIONS, args.top_k, args.timeout
            )
            ok_seq = [r for r in seq_results if r.ok]
            seq_latencies = [r.latency_s for r in ok_seq]
            seq_stats = compute_stats(seq_latencies)

            print(f"\n  {_BOLD}Stats (successful requests only):{_RESET}")
            _print_stats(seq_stats)
            _error_summary(seq_results)

            report["sequential"] = {
                "results": [dataclasses.asdict(r) for r in seq_results],
                "stats": seq_stats,
            }

        # ── Phase 2: concurrent ─────────────────────────────────────────────
        if not args.no_concurrent:
            con_results = await run_concurrent(
                session,
                args.base_url,
                QUESTIONS,
                args.concurrency,
                args.top_k,
                args.timeout,
            )
            ok_con = [r for r in con_results if r.ok]
            con_latencies = [r.latency_s for r in ok_con]
            con_stats = compute_stats(con_latencies)

            print(f"\n  {_BOLD}Stats (successful requests only):{_RESET}")
            _print_stats(con_stats)
            _error_summary(con_results)

            report["concurrent"] = {
                "results": [dataclasses.asdict(r) for r in con_results],
                "stats": con_stats,
            }

    # ── Final summary ───────────────────────────────────────────────────────
    _print_header("Summary")
    if report["sequential"]:
        s = report["sequential"]["stats"]
        print(f"  Sequential  avg={s.get('avg', 0):.2f}s  p95={s.get('p95', 0):.2f}s")
    if report["concurrent"]:
        c = report["concurrent"]["stats"]
        print(f"  Concurrent  avg={c.get('avg', 0):.2f}s  p95={c.get('p95', 0):.2f}s")
    print()

    # ── Optional JSON output ────────────────────────────────────────────────
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2))
        print(f"  Results written to {out_path}\n")

    # Exit 1 if any request failed
    all_results: list[RequestResult] = []
    if report["sequential"]:
        # dataclasses.asdict was already called; reconstruct for the check
        all_results += [r for r in seq_results]  # type: ignore[possibly-undefined]
    if report["concurrent"]:
        all_results += [r for r in con_results]  # type: ignore[possibly-undefined]

    return 0 if all(r.ok for r in all_results) else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load test for the Legal Intelligence Engine API",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Base URL of the running API server",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Maximum number of simultaneous in-flight requests in concurrent phase",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Per-request timeout in seconds",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        dest="top_k",
        help="top_k value sent in every query request",
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="PATH",
        help="Write full JSON report to this file path",
    )
    parser.add_argument(
        "--no-sequential",
        action="store_true",
        help="Skip the sequential phase",
    )
    parser.add_argument(
        "--no-concurrent",
        action="store_true",
        help="Skip the concurrent phase",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sys.exit(asyncio.run(main(args)))
