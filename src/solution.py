from __future__ import annotations

from src.contracts import Architecture, ProcurementDecision
from src.copilot import run
from src.data_access import get_request


def handle_request(request_id: str, architecture: Architecture = "single") -> ProcurementDecision:
    """Assessment adapter.

    architecture="single": one tool-calling agent (Architecture A).
    architecture="staged": analyst agent -> deterministic policy -> reviewer agent (Architecture B).
    Both share the same tools, deterministic policy engine and guardrail merge (src/copilot.py).
    """
    return run(get_request(request_id), architecture).decision
