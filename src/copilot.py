"""Orchestration + guardrail merge.

AI interprets context and recommends; CODE owns thresholds and mandatory checks;
HUMANS own approvals. The merge below enforces that split:
  * approvals / risk flags / missing info = policy engine  UNION  model additions
  * route = the stricter of the policy-engine route and the model's route
  * evidence = produced by code from tool outputs only
  * human_review_required is always True (policy section 11)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from src.agents import RISK_FLAGS, run_single, run_staged
from src.contracts import Architecture, EvidenceItem, ProcurementDecision, RunTelemetry
from src.llm import LLM, LLMError
from src.policy_engine import APPROVAL_ORDER, ROUTES, detect_injection
from src.tools import ToolSession, Telemetry


@dataclass
class CopilotResult:
    decision: ProcurementDecision
    telemetry: Telemetry
    policy: dict
    model_output: dict | None
    analyst_output: dict | None = None
    llm_error: str | None = None
    approval_reasons: dict = field(default_factory=dict)
    final_route: str = ""
    elapsed_s: float = 0.0


def _stricter(a: str, b: str | None) -> str:
    if b not in ROUTES:
        return a
    return a if ROUTES.index(a) <= ROUTES.index(b) else b


def _join(items: list[str]) -> str:
    return ", ".join(items) if items else "none"


def _recommendation(route: str, approvals: list[str], overlapping: list[str]) -> str:
    specialists = [a for a in approvals if a in ("Security", "Privacy", "Legal")]
    business = [a for a in approvals if a not in specialists]
    return {
        "REQUEST_CLARIFICATION": "Request clarification - not ready for approval routing",
        "ESCALATE_MANUAL_REVIEW": "Escalate to manual review - material evidence could not be verified",
        "CONSIDER_EXISTING_TOOL": f"Review existing option ({_join(overlapping)}) before any new purchase",
        "ROUTE_FOR_SPECIALIST_REVIEW": f"Route to {_join(specialists) if specialists else 'Finance budget-exception'} "
                                       f"review before {_join(business)} approval",
        "ROUTE_FOR_BUSINESS_APPROVAL": f"Proceed to {_join(business)} approval - no specialist review triggered",
    }[route]


def _next_step(route: str, request: dict, approvals: list[str], missing: list[str], questions: list[str],
               manual: list[str], overlapping: list[str]) -> str:
    specialists = [a for a in approvals if a in ("Security", "Privacy", "Legal")]
    business = [a for a in approvals if a not in specialists]
    then = f" Required approvers once ready: {_join(approvals)}."
    if route == "REQUEST_CLARIFICATION":
        q = (" Ask: " + " ".join(questions[:3])) if questions else ""
        return f"Procurement returns the request to the requester for: {_join(missing)}.{q} Re-run the review when complete.{then}"
    if route == "ESCALATE_MANUAL_REVIEW":
        return (f"Procurement opens a manual review ({_join(manual)}); Security verifies vendor status directly "
                f"before any approval.{then}")
    if route == "CONSIDER_EXISTING_TOOL":
        return (f"Procurement confirms with the requester whether {_join(overlapping)} (existing seats/scope) meets the need; "
                f"only if a gap is confirmed, continue to review.{then}")
    if route == "ROUTE_FOR_SPECIALIST_REVIEW":
        first = _join(specialists) if specialists else "Finance (budget exception)"
        return f"Send the evidence pack to {first} for review; after sign-off, route to {_join(business)} for approval."
    return f"Send the evidence pack to {_join(business)} for approval."


def _merge(request: dict, session: ToolSession, policy: dict, model: dict | None, llm_error: str | None,
           telemetry: Telemetry) -> tuple[ProcurementDecision, dict, str]:
    approvals = {a: list(r) for a, r in policy["approval_reasons"].items()}
    flags = list(policy["risk_flags"])
    missing = list(policy["missing_information"])
    route = policy["route"]
    overlapping: list[str] = []
    questions: list[str] = []
    rationale = ""

    if model:
        for f in model.get("additional_risk_flags") or []:
            if f in RISK_FLAGS and f not in flags:
                flags.append(f)
        for item in model.get("additional_approvals") or []:
            if isinstance(item, dict) and item.get("approver") in APPROVAL_ORDER and item["approver"] not in approvals:
                approvals.setdefault(item["approver"], []).append(f"AI-suggested: {item.get('reason', '')}")
        if model.get("overlap_assessment") in ("overlap_with_credible_gap", "overlap_without_credible_gap"):
            if "existing_tool_overlap" not in flags:
                flags.append("existing_tool_overlap")
            overlapping = [str(t) for t in model.get("overlapping_tools") or []]
        if model.get("prompt_injection_suspected") and "prompt_injection_detected" not in flags:
            flags.append("prompt_injection_detected")
        if model.get("justification_adequate") is False:
            missing.append("clear business purpose / justification")
            if "missing_information" not in flags:
                flags.append("missing_information")
            route = "REQUEST_CLARIFICATION"
        route = _stricter(route, model.get("route"))
        questions = [str(q) for q in model.get("clarifying_questions") or []]
        rationale = str(model.get("rationale") or "").strip()
        if detect_injection({"rationale": rationale}):  # never echo injected instructions as advice
            rationale = ""
    else:
        flags.append("ai_assessment_unavailable")

    if missing and route != "REQUEST_CLARIFICATION":
        route = "REQUEST_CLARIFICATION"
    if route == "CONSIDER_EXISTING_TOOL" and not overlapping:
        overlapping = ["the existing tool(s) listed in the evidence"]

    ordered = [a for a in APPROVAL_ORDER if a in approvals]
    recommendation = _recommendation(route, ordered, overlapping)
    if rationale:
        recommendation += f". {rationale}"
    if llm_error:
        recommendation += " (Deterministic fallback: AI assessment unavailable.)"

    next_step = _next_step(route, request, ordered, missing, questions, policy["manual_review_reasons"], overlapping)
    next_step += " Final approval stays with the named humans; the copilot has not approved or purchased anything."

    decision = ProcurementDecision(
        request_id=request["request_id"],
        recommendation=recommendation,
        evidence=[EvidenceItem(**e) for e in session.evidence],
        required_approvals=ordered,
        missing_information=missing,
        risk_flags=flags,
        next_step=next_step,
        human_review_required=True,
        telemetry=RunTelemetry(llm_calls=telemetry.llm_calls, tool_calls=telemetry.tool_calls,
                               tool_names=list(telemetry.tool_names)),
    )
    return decision, {a: approvals[a] for a in ordered}, route


def run(request: dict, architecture: Architecture = "single") -> CopilotResult:
    start = time.perf_counter()
    telemetry = Telemetry()
    session = ToolSession(request, telemetry)
    model: dict | None = None
    analysis: dict | None = None
    error: str | None = None

    try:
        if architecture == "rules_only":  # evaluation ablation: no model at all
            raise LLMError("rules_only ablation (no LLM)")
        llm = LLM(telemetry)
        if architecture == "single":
            model = run_single(llm, request, session)
        elif architecture == "staged":
            model, analysis = run_staged(llm, request, session)
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
    except LLMError as exc:
        error = str(exc)

    # Guardrails: complete evidence + authoritative policy, whatever the model did.
    session.ensure_core_evidence()
    policy = session.policy()
    decision, reasons, route = _merge(request, session, policy, model, error, telemetry)
    return CopilotResult(decision=decision, telemetry=telemetry, policy=policy, model_output=model,
                         analyst_output=analysis, llm_error=error, approval_reasons=reasons,
                         final_route=route, elapsed_s=time.perf_counter() - start)
