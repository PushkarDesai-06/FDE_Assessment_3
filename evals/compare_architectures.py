"""Run the same case set on both architectures and write a comparison.

    python evals/compare_architectures.py                 # public + extended cases, single + staged
    python evals/compare_architectures.py --cases public  # the six public cases only
    python evals/compare_architectures.py --reps 2        # repeat each case to measure variance

Starts the mock vendor-risk API if it is not already running. Adds a no-LLM
"rules_only" ablation so the model's marginal contribution is visible.
Outputs: evals/results/comparison_runs.csv and evals/results/comparison_summary.md
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evals.run_public_evals import evaluate, group_present  # noqa: E402
from src.copilot import run  # noqa: E402
from src.data_access import get_request  # noqa: E402

RESULTS = ROOT / "evals" / "results"


def ensure_mock_api() -> subprocess.Popen | None:
    base = os.getenv("VENDOR_RISK_BASE_URL", "http://127.0.0.1:8001").rstrip("/")
    try:
        if requests.get(f"{base}/health", timeout=1).ok:
            return None
    except requests.RequestException:
        pass
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "mock_api.app:app", "--host", "127.0.0.1", "--port", "8001"],
                            cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(40):
        try:
            if requests.get(f"{base}/health", timeout=0.5).ok:
                return proc
        except requests.RequestException:
            time.sleep(0.25)
    raise RuntimeError("Mock vendor-risk API did not start")


def extra_checks(result, expectations: dict) -> list[str]:
    failures = []
    d = result.decision
    for token in expectations.get("required_approvals_must_not_contain", []):
        if group_present(d.required_approvals, [token]):
            failures.append(f"unexpected approval containing: {token}")
    routes = expectations.get("route_any")
    if routes and result.final_route not in routes:
        failures.append(f"route {result.final_route} not in {routes}")
    return failures


def load_cases(which: str) -> list[dict]:
    cases = []
    if which in ("public", "all"):
        cases += [dict(c, suite="public") for c in json.loads((ROOT / "evals" / "public_cases.json").read_text())]
    if which in ("extended", "all"):
        cases += [dict(c, suite="extended") for c in json.loads((ROOT / "evals" / "extended_cases.json").read_text())]
    return cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", choices=["public", "extended", "all"], default="all")
    parser.add_argument("--architectures", default="rules_only,single,staged")
    parser.add_argument("--reps", type=int, default=1)
    args = parser.parse_args()

    api = ensure_mock_api()
    cases = load_cases(args.cases)
    rows = []
    try:
        for arch in args.architectures.split(","):
            print(f"\n=== architecture={arch} ===")
            for rep in range(args.reps):
                for case in cases:
                    request = case.get("request") or get_request(case["request_id"])
                    result = run(request, arch)
                    t = result.telemetry
                    failures = evaluate(result.decision, case["expectations"]) + extra_checks(result, case["expectations"])
                    model_route = (result.model_output or {}).get("route")
                    row = {
                        "suite": case["suite"], "case_id": case["case_id"], "architecture": arch, "rep": rep,
                        "passed": not failures,
                        "latency_s": round(result.elapsed_s, 2),
                        "active_latency_s": round(result.elapsed_s - t.rate_limit_wait_s, 2),
                        "rate_limit_wait_s": round(t.rate_limit_wait_s, 2),
                        "llm_calls": t.llm_calls, "tool_calls": t.tool_calls,
                        "agent_tool_calls": sum(1 for n in t.tool_names if "(" not in n),
                        "guardrail_tool_fills": sum(1 for n in t.tool_names if "(guardrail)" in n),
                        "prompt_tokens": t.prompt_tokens, "completion_tokens": t.completion_tokens,
                        "policy_route": result.policy["route"], "model_route": model_route or "",
                        "final_route": result.final_route,
                        "model_route_relaxed": bool(model_route) and model_route != result.final_route,
                        "llm_error": result.llm_error or "",
                        "approvals": "; ".join(result.decision.required_approvals),
                        "risk_flags": "; ".join(result.decision.risk_flags),
                        "failures": " | ".join(failures),
                    }
                    rows.append(row)
                    print(f"{'PASS' if row['passed'] else 'FAIL'}  {case['case_id']:7} rep{rep} {row['latency_s']:6.1f}s "
                          f"(active {row['active_latency_s']:5.1f}s) llm={t.llm_calls} tools={t.tool_calls} "
                          f"route={result.final_route}" + (f"  ERR={result.llm_error}" if result.llm_error else ""))
                    for f in failures:
                        print(f"        - {f}")
    finally:
        if api:
            api.terminate()

    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / "comparison_runs.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = summarise(rows)
    (RESULTS / "comparison_summary.md").write_text(summary, encoding="utf-8")
    print("\n" + summary)


def summarise(rows: list[dict]) -> str:
    archs = list(dict.fromkeys(r["architecture"] for r in rows))
    mean = lambda xs: statistics.mean(xs) if xs else 0.0  # noqa: E731

    def stats(arch: str) -> dict:
        rs = [r for r in rows if r["architecture"] == arch]
        pub = [r for r in rs if r["suite"] == "public"]
        ext = [r for r in rs if r["suite"] == "extended"]
        return {
            "Public cases passed": f"{sum(r['passed'] for r in pub)}/{len(pub)}",
            "Extended cases passed": f"{sum(r['passed'] for r in ext)}/{len(ext)}",
            "Avg end-to-end latency (s)": f"{mean([r['latency_s'] for r in rs]):.1f}",
            "Avg active latency excl. rate-limit waits (s)": f"{mean([r['active_latency_s'] for r in rs]):.1f}",
            "Avg LLM calls": f"{mean([r['llm_calls'] for r in rs]):.1f}",
            "Avg tool calls": f"{mean([r['tool_calls'] for r in rs]):.1f}",
            "Avg tokens (prompt+completion)": f"{mean([r['prompt_tokens'] + r['completion_tokens'] for r in rs]):,.0f}",
            "Runs where guardrail filled skipped tools": str(sum(r["guardrail_tool_fills"] > 0 for r in rs)),
            "Runs where model route was relaxed and overridden": str(sum(bool(r["model_route_relaxed"]) for r in rs)),
            "LLM errors (fell back to rules)": str(sum(bool(r["llm_error"]) for r in rs)),
        }

    table = {a: stats(a) for a in archs}
    metrics = list(next(iter(table.values())).keys())
    lines = ["# Architecture comparison", "",
             f"Cases: {len({r['case_id'] for r in rows})} ({sum(1 for r in rows if r['architecture'] == archs[0])} runs per architecture). "
             "`rules_only` is a no-LLM ablation, not a shipped option.", "",
             "| Metric | " + " | ".join(archs) + " |", "|---|" + "---:|" * len(archs)]
    lines += [f"| {m} | " + " | ".join(table[a][m] for a in archs) + " |" for m in metrics]
    failed = [r for r in rows if not r["passed"]]
    lines += ["", "## Failures", ""]
    lines += [f"- **{r['architecture']} / {r['case_id']}** (rep {r['rep']}): {r['failures']}" for r in failed] or ["- none"]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
