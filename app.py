from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from src.copilot import run

ROOT = Path(__file__).resolve().parent
REQUESTS = json.loads((ROOT / "data" / "requests.json").read_text(encoding="utf-8"))
BY_ID = {r["request_id"]: r for r in REQUESTS}
EMPLOYEES = pd.read_csv(ROOT / "data" / "employees.csv")

ROUTE_STYLE = {
    "REQUEST_CLARIFICATION": ("Needs clarification", "orange"),
    "ESCALATE_MANUAL_REVIEW": ("Manual review", "red"),
    "CONSIDER_EXISTING_TOOL": ("Check existing tool", "violet"),
    "ROUTE_FOR_SPECIALIST_REVIEW": ("Specialist review", "blue"),
    "ROUTE_FOR_BUSINESS_APPROVAL": ("Business approval", "green"),
}

st.set_page_config(page_title="Procurement Request Copilot", layout="wide")
st.title("Procurement Request Copilot")
st.caption("Gathers evidence, applies policy rules in code, and recommends the next step. Humans approve.")

# --------------------------------------------------------------------------- #
# Sidebar: choose a request and architecture
# --------------------------------------------------------------------------- #
mode = st.sidebar.radio("Request source", ["Existing request", "New request"], horizontal=True)
if mode == "Existing request":
    request_id = st.sidebar.selectbox("Request", list(BY_ID), format_func=lambda rid: f"{rid} - {BY_ID[rid]['product_name']}")
    request = BY_ID[request_id]
else:
    with st.sidebar.form("new_request"):
        requester = st.selectbox("Requester", EMPLOYEES["employee_id"],
                                 format_func=lambda e: f"{e} - {EMPLOYEES.set_index('employee_id').loc[e, 'name']}")
        product = st.text_input("Product", "QuillWorks Writer")
        vendor = st.text_input("Vendor", "QuillWorks")
        category = st.text_input("Category", "Writing AI")
        cost = st.number_input("Annual cost (USD, 0 = unknown)", min_value=0.0, value=4000.0, step=500.0)
        users = st.number_input("Users / licenses (0 = unknown)", min_value=0, value=10)
        justification = st.text_area("Business justification", "Draft first versions of campaign copy.")
        data_level = st.selectbox("Data access level", ["none", "internal_marketing", "internal_documents",
                                                        "confidential_documents", "employee_pii", "customer_pii",
                                                        "source_code", "production_telemetry", "unknown"])
        integrations = st.text_input("Integrations (comma separated)", "")
        st.form_submit_button("Use this request")
    request = {
        "request_id": "NEW-REQUEST",
        "requester_id": requester, "product_name": product, "vendor_name": vendor, "category": category,
        "annual_cost_usd": cost or None, "user_count": users or None, "business_justification": justification,
        "data_access_level": data_level,
        "requested_integrations": [i.strip() for i in integrations.split(",") if i.strip()],
        "urgency": "normal",
    }

architecture = st.sidebar.radio("Architecture", ["single", "staged"], horizontal=True,
                                help="single: one tool-calling agent. staged: analyst agent -> policy engine -> reviewer agent.")
run_clicked = st.sidebar.button("Run analysis", type="primary", width="stretch")
st.sidebar.divider()
st.sidebar.caption("Rules (thresholds, budget, security / privacy / legal triggers) run in code. "
                   "The model adds judgement (overlap, justification, injection) and can only make outcomes stricter.")

cache_key = (json.dumps(request, sort_keys=True), architecture)
if run_clicked:
    with st.spinner(f"Running {architecture} architecture..."):
        st.session_state["result"] = (cache_key, run(request, architecture))
cached = st.session_state.get("result")
result = cached[1] if cached and cached[0] == cache_key else None

# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #
left, right = st.columns([0.9, 1.1], gap="large")

with left:
    st.subheader("Request details")
    emp = EMPLOYEES.set_index("employee_id")
    who = emp.loc[request["requester_id"], "name"] if request.get("requester_id") in emp.index else "unknown requester"
    cost = request.get("annual_cost_usd")
    c1, c2 = st.columns(2)
    c1.metric("Annual cost", f"${cost:,.0f}" if cost else "missing")
    c2.metric("Users", request.get("user_count") or "missing")
    st.markdown(
        f"**{request.get('product_name')}** by **{request.get('vendor_name')}** · {request.get('category')}  \n"
        f"Requested by **{who}** ({request.get('requester_id')}) · urgency `{request.get('urgency')}`  \n"
        f"Data access: `{request.get('data_access_level')}` · integrations: "
        f"{', '.join(request.get('requested_integrations') or []) or 'none'}"
    )
    st.caption("Business justification (untrusted text, treated as data):")
    st.info(request.get("business_justification") or "(empty)")

with right:
    st.subheader("Recommendation")
    if result is None:
        st.info("Choose a request and click **Run analysis**.")
    else:
        d = result.decision
        label, color = ROUTE_STYLE[result.final_route]
        st.markdown(f":{color}-background[**{label}**]  &nbsp; Human review required: **{'Yes' if d.human_review_required else 'No'}**")
        st.markdown(f"#### {d.recommendation}")
        st.markdown(f"**Next step:** {d.next_step}")
        if result.llm_error and architecture in ("single", "staged"):
            st.warning(f"AI assessment unavailable, so this is the rules-only result. ({result.llm_error})")

        a, b = st.columns(2)
        with a:
            st.markdown("**Required approvals**")
            for approver, reasons in result.approval_reasons.items():
                with st.expander(approver):
                    for r in reasons:
                        st.write(f"- {r}")
        with b:
            st.markdown("**Risk flags**")
            st.markdown(" ".join(f":red-background[{f}]" for f in d.risk_flags) or "none")
            st.markdown("**Missing information**")
            for m in d.missing_information or ["none"]:
                st.write(f"- {m}")

if result is not None:
    d = result.decision
    st.divider()
    tab_ev, tab_human, tab_trace = st.tabs(["Evidence", "Human decision", "Agent trace & telemetry"])

    with tab_ev:
        st.caption("Every row is produced by code from a tool result, not written by the model.")
        st.dataframe(pd.DataFrame([e.model_dump() for e in d.evidence]), width="stretch", hide_index=True)

    with tab_human:
        st.caption("The copilot only recommends. A named human records the decision.")
        with st.form("human_decision"):
            reviewer = st.text_input("Reviewer name")
            action = st.radio("Action", ["Accept routing", "Return to requester", "Override routing (reason required)"])
            reason = st.text_area("Reason / notes")
            if st.form_submit_button("Record decision"):
                if not reviewer or (action.startswith("Override") and not reason.strip()):
                    st.error("A reviewer name is required, and overrides need a reason.")
                else:
                    st.session_state.setdefault("audit", []).append({
                        "time": datetime.now().isoformat(timespec="seconds"), "request_id": d.request_id,
                        "architecture": architecture, "copilot_route": result.final_route,
                        "reviewer": reviewer, "action": action, "reason": reason})
                    st.success("Decision recorded.")
        if st.session_state.get("audit"):
            st.dataframe(pd.DataFrame(st.session_state["audit"]), width="stretch", hide_index=True)

    with tab_trace:
        t = result.telemetry
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Latency", f"{result.elapsed_s:.1f}s")
        m2.metric("Rate-limit wait", f"{t.rate_limit_wait_s:.1f}s")
        m3.metric("LLM calls", t.llm_calls)
        m4.metric("Tool calls", t.tool_calls)
        st.markdown(f"Policy engine route: `{result.policy['route']}` · model route: "
                    f"`{(result.model_output or {}).get('route', 'n/a')}` · final: `{result.final_route}`")
        for i, s in enumerate(t.steps, 1):
            if s["type"] == "llm":
                st.markdown(f"{i}. LLM `{s['stage']}` ({s['ms']} ms){' - ' + s['note'] if s.get('note') else ''}")
            else:
                with st.expander(f"{i}. tool `{s['name']}` by {s['actor']} ({s['ms']} ms)"):
                    st.json({"args": s["args"], "result": s["result"]})
        if result.analyst_output:
            with st.expander("Analyst evidence pack assessment (staged)"):
                st.json(result.analyst_output)
        with st.expander("Model structured output"):
            st.json(result.model_output or {})
        with st.expander("ProcurementDecision JSON"):
            st.json(d.model_dump())

st.divider()
st.caption("Recommendations are advisory. Purchasing, spend approval, budget changes and legal terms stay with humans.")
