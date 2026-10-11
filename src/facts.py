"""Deterministic evidence lookups (no LLM).

Every function here reads the business data or the vendor-risk API and returns a
compact, JSON-serialisable dict. Free-text fields (notes) are returned as data;
they are never interpreted as instructions.
"""
from __future__ import annotations

import math
import re
from datetime import date

import requests

from src import data_access as da
from src.vendor_client import get_vendor_risk

REVIEW_VALIDITY_DAYS = 365


def _parse_reference_date() -> date:
    """Policy reference date comes from the policy document, not the system clock."""
    match = re.search(r"reference date:\**\s*(\d{4}-\d{2}-\d{2})", da.load_policy_text())
    return date.fromisoformat(match.group(1)) if match else date(2026, 9, 30)


REFERENCE_DATE = _parse_reference_date()


def _clean(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if hasattr(value, "item"):  # numpy scalar -> python
        return value.item()
    return value


def _row(record: dict) -> dict:
    return {k: _clean(v) for k, v in record.items()}


def _norm(text: object) -> str:
    return str(text or "").strip().lower()


def review_age_days(review_date: str | None) -> int | None:
    if not review_date:
        return None
    try:
        return (REFERENCE_DATE - date.fromisoformat(str(review_date))).days
    except ValueError:
        return None


# --------------------------------------------------------------------------- #
# Requester
# --------------------------------------------------------------------------- #
def lookup_requester(employee_id: str | None) -> dict:
    employees = da.load_employees()
    match = employees[employees["employee_id"] == employee_id]
    if not employee_id or match.empty:
        return {"found": False, "employee_id": employee_id}
    rec = _row(match.iloc[0].to_dict())
    manager = employees[employees["employee_id"] == rec.get("manager_id")]
    return {
        "found": True,
        "employee_id": rec["employee_id"],
        "name": rec["name"],
        "department": rec["department"],
        "level": rec["level"],
        "country": rec["country"],
        "manager_id": rec.get("manager_id"),
        "manager_name": None if manager.empty else manager.iloc[0]["name"],
    }


# --------------------------------------------------------------------------- #
# Budget
# --------------------------------------------------------------------------- #
def check_budget(department: str | None, annual_cost_usd: float | None) -> dict:
    budgets = da.load_budgets()
    match = budgets[budgets["department"].str.lower() == _norm(department)]
    if not department or match.empty:
        return {"found": False, "department": department}
    rec = _row(match.iloc[0].to_dict())
    out = {
        "found": True,
        "department": rec["department"],
        "annual_software_budget_usd": rec["annual_software_budget_usd"],
        "committed_usd": rec["committed_usd"],
        "available_usd": rec["available_usd"],
        "requested_annual_cost_usd": annual_cost_usd,
    }
    if annual_cost_usd is None:
        out.update(cost_known=False, within_budget=None)
    else:
        out.update(
            cost_known=True,
            within_budget=annual_cost_usd <= rec["available_usd"],
            remaining_after_purchase_usd=rec["available_usd"] - annual_cost_usd,
        )
    return out


# --------------------------------------------------------------------------- #
# Existing software / purchase history
# --------------------------------------------------------------------------- #
_CATALOG_FIELDS = ["software_id", "product_name", "vendor_name", "category", "status",
                   "annual_cost_usd", "licensed_seats", "scope", "notes"]


def search_catalog(product_name: str | None, vendor_name: str | None, category: str | None) -> dict:
    catalog = da.load_software_catalog()
    items = [_row({k: r[k] for k in _CATALOG_FIELDS}) for r in catalog.to_dict("records")]
    p, v, c = _norm(product_name), _norm(vendor_name), _norm(category)

    same_product = [i for i in items if p and _norm(i["product_name"]) == p]
    same_vendor = [i for i in items if v and _norm(i["vendor_name"]) == v and i not in same_product]
    same_category = [i for i in items if c and _norm(i["category"]) == c and _norm(i["vendor_name"]) != v]

    history = da.load_purchase_history()
    vendor_history = [_row(r) for r in history.to_dict("records") if v and _norm(r["vendor_name"]) == v]
    return {
        "same_product": same_product,
        "same_vendor_other_products": same_vendor,
        "same_category_other_vendors": same_category,
        "vendor_purchase_history": vendor_history,
    }


# --------------------------------------------------------------------------- #
# Vendor registry + external vendor-risk service
# --------------------------------------------------------------------------- #
_REGISTRY_TO_API_STATUS = {"approved": "approved", "pending": "not_completed"}


def get_vendor_profile(vendor_name: str | None) -> dict:
    vendors = da.load_vendors()
    match = vendors[vendors["vendor_name"].str.lower() == _norm(vendor_name)]
    registry = None if match.empty else _row(match.iloc[0].to_dict())

    api: dict = {"status": "not_called"}
    if vendor_name:
        try:
            api = {"status": "ok", "record": get_vendor_risk(vendor_name)}
        except requests.HTTPError as exc:
            code = exc.response.status_code if exc.response is not None else None
            try:
                detail = exc.response.json().get("detail")
            except Exception:
                detail = str(exc)
            api = {"status": "not_found" if code == 404 else "unavailable", "http_status": code, "detail": detail}
        except requests.RequestException as exc:
            api = {"status": "unavailable", "detail": f"{type(exc).__name__}: {exc}"}

    record = api.get("record") or {}
    reg_status = _norm(registry.get("security_status")) if registry else None
    reg_date = registry.get("security_review_date") if registry else None
    api_status = _norm(record.get("security_review_status")) or None
    api_date = record.get("last_review_date")

    reg_age, api_age = review_age_days(reg_date), review_age_days(api_date)

    conflicts: list[str] = []
    mapped = _REGISTRY_TO_API_STATUS.get(reg_status or "")
    if mapped and api_status and mapped != api_status:
        conflicts.append(f"registry security_status '{registry['security_status']}' vs vendor-risk service '{api_status}'")
    if reg_date and api_date and reg_date != api_date:
        conflicts.append(f"registry review date {reg_date} vs vendor-risk service {api_date}")

    def _current(status: str | None, age: int | None) -> bool:
        return status == "approved" and age is not None and 0 <= age <= REVIEW_VALIDITY_DAYS

    expired = any(age is not None and age > REVIEW_VALIDITY_DAYS for age in (reg_age, api_age)) or api_status == "expired"
    assessment_current = (
        api["status"] == "ok"
        and not conflicts
        and _current(api_status, api_age)
        and (registry is None or _current(reg_status, reg_age))
    )

    return {
        "vendor_name": vendor_name,
        "in_registry": registry is not None,
        "registry": registry,
        "vendor_risk_service": api,
        "reference_date": REFERENCE_DATE.isoformat(),
        "registry_review_age_days": reg_age,
        "service_review_age_days": api_age,
        "security_assessment_current": assessment_current,
        "security_review_expired": expired,
        "conflicts": conflicts,
    }
