#!/usr/bin/env python3
"""
prompt-eval: Prompt Evaluation Harness for Claude

Runs a YAML-defined test suite against a Claude prompt, measures quality and cost,
and shows a diff when the prompt changes.

Key Claude API features:
- Prompt caching: system prompt is cached and shared across all test cases (~90% cheaper)
- Async execution: all test cases run concurrently (faster wall-clock time)
- LLM-as-judge: uses Claude to score open-ended outputs
- Structured output: JSON field matching for deterministic responses
"""

import argparse
import asyncio
import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import anthropic
import yaml
from rich.console import Console
from rich.table import Table

console = Console()

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class Case:
    id: str
    input: str
    expected: dict[str, Any]
    tags: list[str] = field(default_factory=list)


@dataclass
class Result:
    case_id: str
    passed: bool
    actual_output: str
    actual_parsed: dict | None
    expected: dict[str, Any]
    score: float
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cache_creation_tokens: int
    cache_read_tokens: int
    error: str | None = None


@dataclass
class RunSummary:
    suite_name: str
    model: str
    prompt_hash: str
    results: list[Result]
    total_ms: float


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def strip_markdown_json(text: str) -> str:
    """Strip ```json ... ``` fences if present. Claude sometimes wraps JSON despite instructions."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        # Remove first line (```json or ```) and last line (```)
        inner = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
        return "\n".join(inner).strip()
    return text


def score_json_fields(actual: dict, expected: dict) -> tuple[bool, float]:
    """All expected fields must match in actual output."""
    if not expected:
        return True, 1.0
    matches = sum(1 for k, v in expected.items() if actual.get(k) == v)
    score = matches / len(expected)
    return score == 1.0, score


async def llm_judge(
    client: anthropic.AsyncAnthropic,
    input_text: str,
    actual_output: str,
    expected: dict,
) -> tuple[bool, float]:
    """
    Use Claude to score open-ended outputs against evaluation criteria.

    Using a separate, cheaper model (Haiku) for the judge keeps costs low.
    The judge's system prompt is also cached — it's identical for every case.
    """
    judge_system = """You are an objective evaluator assessing AI responses against criteria.
Respond with JSON only (no markdown): {"score": 0.0-1.0, "passed": true/false, "reason": "one line"}
Scoring: 1.0 = fully meets all criteria. 0.5 = partially meets. 0.0 = does not meet.
passed = true when score >= 0.7"""

    eval_input = f"""Input: {input_text}

Response to evaluate:
{actual_output}

Criteria to check:
{json.dumps(expected, indent=2)}"""

    response = await client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=256,
        system=[
            {
                "type": "text",
                "text": judge_system,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": eval_input}],
    )

    try:
        result = json.loads(response.content[0].text)
        return bool(result.get("passed", False)), float(result.get("score", 0.0))
    except (json.JSONDecodeError, KeyError, TypeError):
        return False, 0.0


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------


async def run_case(
    client: anthropic.AsyncAnthropic,
    prompt: str,
    case: Case,
    model: str,
    scorer: str,
) -> Result:
    """
    Run a single test case and return a scored Result.

    The system prompt is marked with cache_control so it's cached after the
    first call. Every subsequent case that shares the same prompt pays only
    the cache-read price (~10% of normal input cost).
    """
    start = time.monotonic()
    error = None
    actual_output = ""
    actual_parsed = None
    passed = False
    score = 0.0
    usage = None

    try:
        response = await client.messages.create(
            model=model,
            max_tokens=512,
            system=[
                {
                    "type": "text",
                    "text": prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": case.input}],
        )
        actual_output = response.content[0].text
        usage = response.usage

        if scorer == "json_field_match":
            try:
                actual_parsed = json.loads(strip_markdown_json(actual_output))
                passed, score = score_json_fields(actual_parsed, case.expected)
            except json.JSONDecodeError:
                passed = False
                score = 0.0
                error = "Response was not valid JSON"

        elif scorer == "exact":
            expected_str = str(case.expected.get("output", "")).strip()
            passed = actual_output.strip() == expected_str
            score = 1.0 if passed else 0.0

        elif scorer == "llm_judge":
            passed, score = await llm_judge(client, case.input, actual_output, case.expected)

        else:
            error = f"Unknown scorer: {scorer}"

    except anthropic.RateLimitError:
        error = "Rate limited"
    except anthropic.APIError as e:
        error = f"API error: {e.status_code}"
    except Exception as e:
        error = str(e)

    latency_ms = (time.monotonic() - start) * 1000

    return Result(
        case_id=case.id,
        passed=passed,
        actual_output=actual_output,
        actual_parsed=actual_parsed,
        expected=case.expected,
        score=score,
        latency_ms=latency_ms,
        input_tokens=usage.input_tokens if usage else 0,
        output_tokens=usage.output_tokens if usage else 0,
        cache_creation_tokens=usage.cache_creation_input_tokens if usage else 0,
        cache_read_tokens=usage.cache_read_input_tokens if usage else 0,
        error=error,
    )


async def run_suite(suite_path: Path, model_override: str | None = None) -> RunSummary:
    """Load a YAML eval suite and run all cases concurrently."""
    with open(suite_path) as f:
        suite = yaml.safe_load(f)

    name = suite["name"]
    prompt = suite["prompt"]
    model = model_override or suite.get("model", "claude-sonnet-4-6")
    scorer = suite.get("scorer", "json_field_match")
    cases = [
        Case(
            id=c["id"],
            input=c["input"],
            expected=c.get("expected", {}),
            tags=c.get("tags", []),
        )
        for c in suite["cases"]
    ]

    prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:8]

    console.print(f"\n[bold]{name}[/bold]")
    console.print(f"[dim]model: {model}  |  cases: {len(cases)}  |  scorer: {scorer}  |  prompt: {prompt_hash}[/dim]")
    console.print()

    async_client = anthropic.AsyncAnthropic()
    start = time.monotonic()

    # All cases run concurrently. The system prompt is cached after the first
    # response arrives, so most parallel requests still share the cache —
    # they're writing to the same cache key and reads start immediately.
    tasks = [run_case(async_client, prompt, case, model, scorer) for case in cases]
    results = await asyncio.gather(*tasks)

    total_ms = (time.monotonic() - start) * 1000

    return RunSummary(
        suite_name=name,
        model=model,
        prompt_hash=prompt_hash,
        results=list(results),
        total_ms=total_ms,
    )


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def display_results(summary: RunSummary, show_failures: bool = False) -> None:
    results = summary.results
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    pct = passed / total * 100 if total else 0

    color = "green" if pct == 100 else "yellow" if pct >= 70 else "red"
    title = f"{summary.suite_name}  [{color}]{passed}/{total} passed ({pct:.0f}%)[/{color}]"

    table = Table(title=title, show_header=True, header_style="bold")
    table.add_column("Case ID", style="dim", min_width=16)
    table.add_column("", justify="center", width=3)   # pass/fail emoji
    table.add_column("Score", justify="right", width=6)
    table.add_column("Cache", justify="center", width=6)
    table.add_column("Latency", justify="right", width=8)
    table.add_column("Error", style="red dim")

    for r in results:
        status = "✅" if r.passed else "❌"
        # 🎯 = cache read (cheap), 💾 = cache write (first call), — = no cache
        if r.cache_read_tokens > 0:
            cache = "🎯"
        elif r.cache_creation_tokens > 0:
            cache = "💾"
        else:
            cache = "[dim]—[/dim]"
        table.add_row(
            r.case_id,
            status,
            f"{r.score:.2f}",
            cache,
            f"{r.latency_ms:.0f}ms",
            r.error or "",
        )

    console.print(table)

    # Token / cost summary
    total_input = sum(r.input_tokens for r in results)
    total_output = sum(r.output_tokens for r in results)
    total_writes = sum(r.cache_creation_tokens for r in results)
    total_reads = sum(r.cache_read_tokens for r in results)
    total_billable = total_input + total_writes + total_reads

    cache_pct = total_reads / total_billable * 100 if total_billable else 0

    console.print()
    console.print("[bold]Tokens[/bold]")
    console.print(f"  Uncached input : {total_input:>8,}")
    console.print(f"  Cache writes   : {total_writes:>8,}  [dim](1.25× cost)[/dim]")
    console.print(f"  Cache reads    : {total_reads:>8,}  [green](0.1× cost)[/green]")
    console.print(f"  Output         : {total_output:>8,}")
    console.print(f"  Cache hit rate : {cache_pct:>7.1f}%")
    console.print(f"  Wall-clock     : {summary.total_ms:>7.0f}ms")

    if show_failures:
        failures = [r for r in results if not r.passed]
        if failures:
            console.print()
            console.print("[bold]Failures[/bold]")
            for r in failures:
                console.print(f"\n  [red]{r.case_id}[/red]")
                console.print(f"  Expected : {r.expected}")
                console.print(f"  Got      : {r.actual_output[:300]!r}")
                if r.error:
                    console.print(f"  Error    : {r.error}")


def display_diff(current: RunSummary, baseline_path: Path) -> None:
    """Show what changed compared to a saved baseline."""
    with open(baseline_path) as f:
        baseline = json.load(f)

    b_by_id = {r["case_id"]: r for r in baseline["results"]}
    c_by_id = {r.case_id: r for r in current.results}

    regressions = []
    improvements = []
    unchanged_pass = 0
    unchanged_fail = 0

    for case_id, result in c_by_id.items():
        prev = b_by_id.get(case_id)
        if not prev:
            continue
        if prev["passed"] and not result.passed:
            regressions.append(case_id)
        elif not prev["passed"] and result.passed:
            improvements.append(case_id)
        elif result.passed:
            unchanged_pass += 1
        else:
            unchanged_fail += 1

    console.print()
    console.print(f"[bold]Diff vs[/bold] [dim]{baseline_path.name}[/dim] [dim](prompt {baseline['prompt_hash']})[/dim]")

    if not regressions and not improvements:
        console.print("  [dim]No changes in pass/fail[/dim]")
    else:
        for cid in regressions:
            console.print(f"  [red]REGRESSION [/red] {cid}  (PASS → FAIL)")
        for cid in improvements:
            console.print(f"  [green]IMPROVED   [/green] {cid}  (FAIL → PASS)")

    net = len(improvements) - len(regressions)
    sign = "+" if net > 0 else ""
    color = "green" if net > 0 else "red" if net < 0 else "dim"
    console.print(f"\n  Regressions : {len(regressions)}")
    console.print(f"  Improvements: {len(improvements)}")
    console.print(f"  Net         : [{color}]{sign}{net}[/{color}]")

    # Prompt change detection
    if baseline.get("prompt_hash") != current.prompt_hash:
        console.print(f"\n  [yellow]Prompt changed[/yellow]: {baseline['prompt_hash']} → {current.prompt_hash}")
    else:
        console.print(f"\n  [dim]Prompt unchanged ({current.prompt_hash})[/dim]")


def save_results(summary: RunSummary, output_path: Path) -> None:
    data = {
        "suite_name": summary.suite_name,
        "model": summary.model,
        "prompt_hash": summary.prompt_hash,
        "total_ms": summary.total_ms,
        "total_input_tokens": sum(r.input_tokens for r in summary.results),
        "total_cache_creation_tokens": sum(r.cache_creation_tokens for r in summary.results),
        "total_cache_read_tokens": sum(r.cache_read_tokens for r in summary.results),
        "total_output_tokens": sum(r.output_tokens for r in summary.results),
        "results": [asdict(r) for r in summary.results],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2))
    console.print(f"\n[dim]Saved → {output_path}[/dim]")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="harness",
        description="Prompt evaluation harness for Claude",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python harness.py run evals/support-classifier.yml
  python harness.py run evals/support-classifier.yml --save results/v1.json
  python harness.py run evals/support-classifier.yml --compare results/v1.json
  python harness.py run evals/support-classifier.yml --model claude-haiku-4-5
  python harness.py run evals/support-classifier.yml --show-failures
""",
    )

    sub = parser.add_subparsers(dest="command", required=True)
    run_p = sub.add_parser("run", help="Run an eval suite")
    run_p.add_argument("suite", type=Path, help="Path to YAML eval suite")
    run_p.add_argument("--model", help="Override the model defined in the suite")
    run_p.add_argument("--save", type=Path, metavar="FILE", help="Save results to JSON")
    run_p.add_argument("--compare", type=Path, metavar="FILE", help="Diff against saved results")
    run_p.add_argument("--show-failures", action="store_true", help="Print full output for failed cases")

    args = parser.parse_args()

    if args.command == "run":
        summary = asyncio.run(run_suite(args.suite, args.model))
        display_results(summary, show_failures=args.show_failures)

        if args.compare:
            if args.compare.exists():
                display_diff(summary, args.compare)
            else:
                console.print(f"\n[yellow]--compare: file not found: {args.compare}[/yellow]")

        if args.save:
            save_results(summary, args.save)

        # Exit with non-zero if any cases failed (useful in CI)
        failed = sum(1 for r in summary.results if not r.passed)
        if failed:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
