"""Deterministic procurement policy engine (no LLM).

Encodes data/procurement_policy.md. Given the request and the facts gathered by
src/facts.py it returns the minimum required approvals, mandatory risk flags,
missing information and a route floor. The LLM may add to these results but the
final decision can never contain fewer approvals/flags than this engine returns.
"""
from __future__ import annotations

import re
from typing import Any

APPROVAL_ORDER = ["Manager", "Department Head", "Finance", "CFO", "Procurement", "Security", "Privacy", "Legal"]

# Routes, from most to least restrictive. The LLM may only move a decision to a
# stricter route (or to CONSIDER_EXISTING_TOOL when the code route allows it).
ROUTES = [
    "REQUEST_CLARIFICATION",
    "ESCALATE_MANUAL_REVIEW",
    "CONSIDER_EXISTING_TOOL",
    "ROUTE_FOR_SPECIALIST_REVIEW",
    "ROUTE_FOR_BUSINESS_APPROVAL",
]

UNKNOWN_VALUES = {"", "unknown", "tbd", "n/a", "none provided", "null"}
PII_LEVELS = {"employee_pii", "customer_pii"}
SECURITY_DATA_KEYWORDS = ["source_code", "source code", "production", "cloud", "confidential",
                          "pii", "credential", "secret"]
SECURITY_INTEGRATION_KEYWORDS = ["production", "cloud", "git", "source code", "credential", "secret", "vault"]

INJECTION_PATTERNS = [
    r"\bignore\b.{0,40}\b(rules?|instructions?|polic(y|ies)|controls?|procurement)\b",
    r"\bdisregard\b.{0,40}\b(rules?|instructions?|polic(y|ies)|controls?)\b",
    r"\b(treat|mark|consider|record)\b.{0,40}\bas\b.{0,20}\bapproved\b",
    r"\b(cfo|ceo|vp|finance|security|legal|privacy)[- ]approved\b",
    r"\bapprove (it|this|the request)?\s*(immediately|now|right away|automatically)\b",
    r"\b(bypass|override|skip)\b.{0,30}\b(polic(y|ies)|controls?|review|approvals?|security|legal|privacy)\b",
    r"\b(system|developer) (prompt|message|instructions?)\b",
    r"\byou are now\b",
    r"\b(reveal|print|expose)\b.{0,30}\b(secrets?|api keys?|credentials?|prompt)\b",
]


def _norm(text: object) -> str:
    return str(text or "").strip().lower()


def _missing(value: Any) -> bool:
    return value is None or (isinstance(value, str) and _norm(value) in UNKNOWN_VALUES)


def detect_injection(texts: dict[str, str | None]) -> list[dict]:
    """Return [{field, excerpt}] for text that tries to change rules or fabricate approval."""
    hits = []
    for field, text in texts.items():
        if not text:
            continue
        for pattern in INJECTION_PATTERNS:
            m = re.search(pattern, str(text), flags=re.IGNORECASE)
            if m:
                hits.append({"field": field, "excerpt": m.group(0)[:120]})
                break
    return hits


def financial_approvals(cost: float) -> tuple[str, list[str]]:
    if cost <= 1000:
        return "Up to $1,000", ["Manager"]
    if cost <= 10000:
        return "$1,000.01 - $10,000", ["Department Head", "Procurement"]
    if cost <= 25000:
        return "$10,000.01 - $25,000", ["Department Head", "Finance", "Procurement"]
    return "Above $25,000", ["Department Head", "Finance", "CFO", "Procurement"]


def evaluate(request: dict, requester: dict, budget: dict, catalog: dict, vendor: dict) -> dict:
    approvals: dict[str, list[str]] = {}
    flags: list[str] = []
    missing: list[str] = []
    findings: list[dict] = []
    manual_review: list[str] = []

    def need(approver: str, reason: str) -> None:
        approvals.setdefault(approver, []).append(reason)

    def flag(name: str) -> None:
        if name not in flags:
            flags.append(name)

    def note(finding: str, ref: str) -> None:
        findings.append({"finding": finding, "reference": ref})

    cost = request.get("annual_cost_usd")
    cost_known = isinstance(cost, (int, float)) and not isinstance(cost, bool) and cost >= 0
    data_level = _norm(request.get("data_access_level"))
    data_unknown = _missing(request.get("data_access_level"))
    integrations = request.get("requested_integrations")
    integrations_text = " ".join(_norm(i) for i in (integrations or []))
    registry = vendor.get("registry") or {}
    service = vendor.get("vendor_risk_service", {})
    service_record = service.get("record") or {}

    # §1 Required information --------------------------------------------------
    if not requester.get("found"):
        missing.append("requester and department (requester not found in employee directory)")
    if _missing(request.get("product_name")) or _missing(request.get("vendor_name")):
        missing.append("product/vendor name")
    if not cost_known:
        missing.append("annual cost (or a reasonable annual estimate)")
    if _missing(request.get("user_count")):
        missing.append("number of users/licenses")
    if _missing(request.get("business_justification")):
        missing.append("business purpose")
    if data_unknown:
        missing.append("intended data-access level")
    if integrations is None:
        missing.append("required integrations (state 'none' if not needed)")
    if missing:
        flag("missing_information")
        note(f"Request is not ready for approval; missing: {', '.join(missing)}", "Policy §1")

    # §4 Financial thresholds --------------------------------------------------
    if cost_known:
        tier, tier_approvals = financial_approvals(float(cost))
        for a in tier_approvals:
            need(a, f"Annual amount ${cost:,.0f} falls in tier '{tier}'")
        note(f"Annual amount ${cost:,.0f} -> tier '{tier}' requires {', '.join(tier_approvals)}", "Policy §4")
    else:
        note("Financial approval tier cannot be determined until annual cost is provided", "Policy §4")

    # §2 Budget ----------------------------------------------------------------
    if budget.get("found") and cost_known:
        if budget["within_budget"]:
            note(f"${cost:,.0f} is within {budget['department']} available budget ${budget['available_usd']:,.0f}", "Policy §2")
        else:
            flag("budget_insufficient")
            need("Finance", f"Cost exceeds {budget['department']} available budget (${budget['available_usd']:,.0f})")
            note(f"${cost:,.0f} exceeds {budget['department']} available budget ${budget['available_usd']:,.0f}; "
                 "Finance budget-exception review required", "Policy §2")
    elif not budget.get("found"):
        note("Department budget could not be found; budget check not performed", "Policy §2")
        manual_review.append("budget could not be verified")

    # §3 Existing software / overlap ------------------------------------------
    same_cat = catalog.get("same_category_other_vendors", [])
    same_vendor = catalog.get("same_product", []) + catalog.get("same_vendor_other_products", [])
    if same_cat:
        flag("existing_tool_overlap")
        names = ", ".join(f"{i['product_name']} ({i['scope']}, {i['licensed_seats']} seats)" for i in same_cat)
        note(f"Approved catalog already has {request.get('category')} tools from other vendors: {names}", "Policy §3")
    if same_vendor:
        names = ", ".join(f"{i['product_name']} [{i['status']}] ({i['scope']}, {i['licensed_seats']} seats)" for i in same_vendor)
        note(f"Existing contract(s) with the same vendor: {names}; check existing capacity/scope before new purchase", "Policy §3")

    # §5 Security --------------------------------------------------------------
    security_reasons: list[str] = []
    matched = [k for k in SECURITY_DATA_KEYWORDS if k in data_level]
    if matched:
        security_reasons.append(f"data access level '{request.get('data_access_level')}' is security-sensitive")
    matched_int = [k for k in SECURITY_INTEGRATION_KEYWORDS if k in integrations_text]
    if matched_int:
        security_reasons.append(f"integration(s) {integrations} touch production/cloud/code/credentials")
    if data_unknown:
        security_reasons.append("data access level unknown; sensitive data cannot be ruled out")
    if service.get("status") == "unavailable":
        flag("vendor_risk_unavailable")
        security_reasons.append("vendor-risk service unavailable; vendor security status cannot be verified")
        manual_review.append("vendor-risk evidence unavailable")
        note(f"Vendor-risk service unavailable ({service.get('detail')}); no favourable status inferred", "Policy §10")
    elif service.get("status") == "not_found":
        security_reasons.append("vendor has no record in the vendor-risk service")
        note("Vendor-risk service has no record for this vendor", "Policy §5")
    if vendor.get("security_review_expired"):
        flag("vendor_review_expired")
        security_reasons.append("vendor security assessment is older than 365 days or marked expired")
        note(f"Vendor security review expired (registry age {vendor.get('registry_review_age_days')} days, "
             f"service status '{service_record.get('security_review_status')}')", "Policy §5")
    elif not vendor.get("security_assessment_current") and service.get("status") == "ok":
        security_reasons.append("vendor security assessment is missing or not completed")
    if vendor.get("conflicts"):
        flag("conflicting_vendor_evidence")
        security_reasons.append("vendor registry and vendor-risk service disagree")
        manual_review.append("conflicting vendor evidence")
        note("Conflicting vendor evidence: " + "; ".join(vendor["conflicts"]), "Policy §5")
    limited = [i for i in same_vendor if "limited" in _norm(i.get("status"))]
    if limited and (matched or data_unknown):
        security_reasons.append(f"existing approval for {limited[0]['product_name']} is limited-use and does not cover this data class")
        note(f"{limited[0]['product_name']} is '{limited[0]['status']}'; prior approval does not extend to new data classes", "Policy §8")
    if security_reasons:
        flag("security_review_required")
        for r in security_reasons:
            need("Security", r)
        note("Security review required: " + "; ".join(security_reasons), "Policy §5")

    # §6 Privacy ---------------------------------------------------------------
    privacy_reasons: list[str] = []
    if data_level in PII_LEVELS or "pii" in data_level:
        privacy_reasons.append(f"tool will process {request.get('data_access_level')}")
    outside_region = service_record.get("stores_data_outside_region") is True
    if outside_region and (matched or data_unknown):
        privacy_reasons.append("vendor stores data outside the operating region and data may be sensitive")
    if privacy_reasons:
        flag("privacy_review_required")
        for r in privacy_reasons:
            need("Privacy", r)
        note("Privacy review required: " + "; ".join(privacy_reasons), "Policy §6")

    # §7 Legal -----------------------------------------------------------------
    legal_reasons: list[str] = []
    vendor_new = not vendor.get("in_registry") or _norm(registry.get("procurement_status")) != "approved"
    if vendor_new and cost_known and cost >= 10000:
        legal_reasons.append(f"new vendor with annual spend ${cost:,.0f} (>= $10,000)")
    if vendor_new and not cost_known:
        legal_reasons.append("new vendor and spend unknown; $10,000 legal threshold cannot be ruled out")
    terms = registry.get("legal_terms_status")
    if _norm(terms) != "approved":
        legal_reasons.append(f"legal terms status is '{terms or 'not on file'}'")
    if outside_region and (data_level in PII_LEVELS or "pii" in data_level):
        legal_reasons.append("cross-region processing of personal data")
    if legal_reasons:
        flag("legal_review_required")
        for r in legal_reasons:
            need("Legal", r)
        note("Legal review required: " + "; ".join(legal_reasons), "Policy §7")

    # §9 Untrusted content -----------------------------------------------------
    injection = detect_injection({
        "request.business_justification": request.get("business_justification"),
        "request.product_name": request.get("product_name"),
        "request.urgency": request.get("urgency"),
        "vendor_registry.notes": registry.get("notes"),
        "vendor_risk_service.notes": service_record.get("notes"),
    })
    if injection:
        flag("prompt_injection_detected")
        manual_review.append("embedded instructions in business data")
        note("Embedded instruction ignored as untrusted data: "
             + "; ".join(f"{h['field']}: \"{h['excerpt']}\"" for h in injection), "Policy §9")

    # Route floor ----------------------------------------------------------------
    specialist = {"Security", "Privacy", "Legal"} & approvals.keys()
    if missing:
        route = "REQUEST_CLARIFICATION"
    elif {"vendor_risk_unavailable", "conflicting_vendor_evidence"} & set(flags) or not budget.get("found"):
        route = "ESCALATE_MANUAL_REVIEW"
    elif specialist or "budget_insufficient" in flags:
        route = "ROUTE_FOR_SPECIALIST_REVIEW"
    else:
        route = "ROUTE_FOR_BUSINESS_APPROVAL"

    ordered = [a for a in APPROVAL_ORDER if a in approvals]
    return {
        "required_approvals": ordered,
        "approval_reasons": {a: approvals[a] for a in ordered},
        "risk_flags": flags,
        "missing_information": missing,
        "findings": findings,
        "manual_review_reasons": manual_review,
        "injection_matches": injection,
        "route": route,
    }
