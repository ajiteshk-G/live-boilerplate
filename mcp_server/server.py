"""Custom MCP Server for Gemini Live Boilerplate (Streamable HTTP on Cloud Run).

Provides a set of CRM, financial calculation, ticketing, and cloud operations
tools designed to be called during real-time bidirectional voice conversations.
Uses `stateless_http=True` so requests work across Cloud Run instances and cold
starts without requiring sticky server-side session state.
"""

from __future__ import annotations

import math
import os
import time
from datetime import UTC, datetime
from typing import Any

import uvicorn
from mcp.server.mcpserver import MCPServer
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

mcp = MCPServer(
    "live-ops-crm",
    instructions=(
        "Enterprise CRM, financial EMI calculator, support ticket desk, and "
        "platform health monitor. Use these tools whenever the user asks about "
        "a customer account, loan/EMI calculation, filing a support ticket, or "
        "checking cloud service status."
    ),
)

# ----------------------------------------------------------------- mock dataset

_CUSTOMERS: list[dict[str, Any]] = [
    {
        "customer_id": "CUST-1001",
        "name": "Rajesh Sharma",
        "company": "IndTech Solutions Pvt Ltd",
        "email": "rajesh@indtech.in",
        "phone": "+91-98201-44102",
        "city": "Bengaluru",
        "plan_tier": "Enterprise Plus",
        "annual_contract_inr": 48_00_000,
        "status": "Active",
        "renewal_date": "2027-03-31",
        "csm_owner": "Vikram Desai",
        "open_tickets": [
            {
                "ticket_id": "TKT-8821",
                "summary": "Latency spike on Mumbai asia-south1 voice gateway",
                "priority": "high",
                "status": "In Progress",
            }
        ],
    },
    {
        "customer_id": "CUST-1002",
        "name": "Priya Nair",
        "company": "BharatCloud AI",
        "email": "priya@bharatcloud.in",
        "phone": "+91-98450-77190",
        "city": "Hyderabad",
        "plan_tier": "Growth",
        "annual_contract_inr": 18_50_000,
        "status": "Active",
        "renewal_date": "2026-11-15",
        "csm_owner": "Neha Kapoor",
        "open_tickets": [],
    },
    {
        "customer_id": "CUST-1003",
        "name": "Arjun Mehta",
        "company": "FinSetu Payments",
        "email": "arjun@finsetu.co.in",
        "phone": "+91-98111-65230",
        "city": "Mumbai",
        "plan_tier": "Enterprise",
        "annual_contract_inr": 35_00_000,
        "status": "Active",
        "renewal_date": "2027-01-20",
        "csm_owner": "Vikram Desai",
        "open_tickets": [
            {
                "ticket_id": "TKT-8790",
                "summary": "Webhook signature verification failed on UPI callback",
                "priority": "normal",
                "status": "Waiting on Customer",
            }
        ],
    },
    {
        "customer_id": "CUST-1004",
        "name": "Ananya Iyer",
        "company": "Prana Digital Health",
        "email": "ananya@pranahealth.in",
        "phone": "+91-94440-11820",
        "city": "Chennai",
        "plan_tier": "Scale",
        "annual_contract_inr": 24_00_000,
        "status": "Renewal Due",
        "renewal_date": "2026-10-01",
        "csm_owner": "Neha Kapoor",
        "open_tickets": [],
    },
]

_SERVICES: dict[str, dict[str, Any]] = {
    "voice-gateway": {
        "service": "voice-gateway",
        "status": "operational",
        "primary_region": "asia-south1 (Mumbai)",
        "p99_latency_ms": 42,
        "uptime_30d_percent": 99.98,
        "notes": "All bidirectional audio relays healthy.",
    },
    "payments-upi": {
        "service": "payments-upi",
        "status": "operational",
        "primary_region": "asia-south1 (Mumbai)",
        "p99_latency_ms": 118,
        "uptime_30d_percent": 99.95,
        "notes": "NPCI settlement pipeline operating within normal parameters.",
    },
    "kyc-verifier": {
        "service": "kyc-verifier",
        "status": "degraded",
        "primary_region": "asia-south2 (Delhi)",
        "p99_latency_ms": 620,
        "uptime_30d_percent": 99.40,
        "notes": "Elevated OCR latency on document verification; mitigation in progress.",
    },
    "llm-router": {
        "service": "llm-router",
        "status": "operational",
        "primary_region": "us-central1 (Iowa)",
        "p99_latency_ms": 85,
        "uptime_30d_percent": 99.99,
        "notes": "Gemini 3.8 Live native audio routing active.",
    },
}

_TICKET_COUNTER = 9000


def _format_inr(amount: float) -> str:
    """Return a human-friendly Indian numbering string (lakhs/crores)."""
    if amount >= 1_00_00_000:
        return f"INR {amount / 1_00_00_000:.2f} Crore (₹{amount:,.0f})"
    if amount >= 1_00_000:
        return f"INR {amount / 1_00_000:.2f} Lakh (₹{amount:,.0f})"
    return f"INR ₹{amount:,.2f}"


# ------------------------------------------------------------------------ tools


@mcp.tool()
def lookup_customer_account(query: str) -> dict[str, Any]:
    """Look up an enterprise customer account by customer ID, name, company, or email.

    **Invocation Condition:** Invoke this tool ONLY when the user asks to look up,
    verify, or list customer CRM accounts, contract values, CSM owners, or open tickets.

    Examples of queries: 'CUST-1001', 'Rajesh Sharma', 'IndTech', 'priya@bharatcloud.in',
    or 'list all' to list all customers.
    """
    q = (query or "").strip().lower()
    if not q or q in ("all", "list", "list all"):
        return {
            "status": "ok",
            "retryable": False,
            "count": len(_CUSTOMERS),
            "customers": [
                {
                    "customer_id": c["customer_id"],
                    "name": c["name"],
                    "company": c["company"],
                    "plan_tier": c["plan_tier"],
                    "status": c["status"],
                }
                for c in _CUSTOMERS
            ],
        }

    matches = [
        c
        for c in _CUSTOMERS
        if q in c["customer_id"].lower()
        or q in c["name"].lower()
        or q in c["company"].lower()
        or q in c["email"].lower()
        or q in c["city"].lower()
    ]
    if not matches:
        return {
            "status": "no_results",
            "retryable": False,
            "found": False,
            "query": query,
            "valid_options": [c["customer_id"] for c in _CUSTOMERS],
            "message": (
                f"No customer matching '{query}' exists in the CRM database. "
                "Report this result to the user before calling the tool again. "
                "Valid customer IDs are CUST-1001 (Rajesh Sharma), CUST-1002 (Priya Nair), "
                "CUST-1003 (Arjun Mehta), and CUST-1004 (Ananya Iyer)."
            ),
        }
    customer = dict(matches[0])
    customer["annual_contract_formatted"] = _format_inr(customer["annual_contract_inr"])
    return {"status": "ok", "retryable": False, "found": True, "customer": customer}


@mcp.tool()
def calculate_loan_emi(
    principal_inr: float,
    annual_interest_rate_percent: float,
    tenure_months: int,
) -> dict[str, Any]:
    """Calculate monthly loan EMI, total interest, and repayment summary in Indian Rupees.

    **Invocation Condition:** Invoke this tool ONLY after the user has specified or
    confirmed the loan principal amount, interest rate, and repayment tenure.

    Args:
        principal_inr: Loan principal amount in INR (e.g. 2500000 for 25 Lakhs).
        annual_interest_rate_percent: Annual interest rate as a percentage (e.g. 8.5).
        tenure_months: Total repayment tenure in months (e.g. 60 for 5 years).
    """
    if principal_inr <= 0 or tenure_months <= 0:
        return {
            "status": "invalid_argument",
            "retryable": False,
            "error": (
                "principal_inr and tenure_months must be positive numbers. "
                "Ask the user for valid positive loan parameters before retrying."
            ),
        }

    monthly_rate = (annual_interest_rate_percent / 100.0) / 12.0
    if monthly_rate == 0:
        emi = principal_inr / tenure_months
    else:
        factor = math.pow(1.0 + monthly_rate, tenure_months)
        emi = principal_inr * monthly_rate * factor / (factor - 1.0)

    total_payment = emi * tenure_months
    total_interest = total_payment - principal_inr

    return {
        "status": "ok",
        "retryable": False,
        "principal_formatted": _format_inr(principal_inr),
        "annual_interest_rate_percent": annual_interest_rate_percent,
        "tenure_months": tenure_months,
        "tenure_years": round(tenure_months / 12.0, 2),
        "monthly_emi_inr": round(emi, 2),
        "monthly_emi_formatted": f"₹{round(emi):,}/month",
        "total_interest_inr": round(total_interest, 2),
        "total_interest_formatted": _format_inr(total_interest),
        "total_payment_inr": round(total_payment, 2),
        "total_payment_formatted": _format_inr(total_payment),
    }


@mcp.tool()
def create_support_ticket(
    customer_id_or_name: str,
    issue_summary: str,
    priority: str = "normal",
) -> dict[str, Any]:
    """Create a new support ticket for a customer account and return the ticket ID and SLA.

    **Invocation Condition:** Invoke this tool ONLY when the user explicitly asks to
    file, open, or create a support ticket for a customer issue.

    Args:
        customer_id_or_name: Customer ID (e.g. CUST-1001) or customer name.
        issue_summary: Brief description of the issue reported by the user.
        priority: One of 'low', 'normal', 'high', or 'urgent'.
    """
    global _TICKET_COUNTER
    _TICKET_COUNTER += 1
    ticket_id = f"TKT-{_TICKET_COUNTER}"
    prio = (priority or "normal").lower()
    sla_map = {
        "urgent": "15 minutes (24x7 PagerDuty escalation)",
        "high": "1 hour",
        "normal": "4 business hours",
        "low": "24 business hours",
    }
    sla = sla_map.get(prio, "4 business hours")

    ticket = {
        "status_code": "ok",
        "retryable": False,
        "ticket_id": ticket_id,
        "customer": customer_id_or_name,
        "summary": issue_summary,
        "priority": prio,
        "status": "Open",
        "sla_response_target": sla,
        "created_at": datetime.now(UTC).isoformat(),
    }

    # Attach to matching customer if present
    q = customer_id_or_name.strip().lower()
    for c in _CUSTOMERS:
        if q in c["customer_id"].lower() or q in c["name"].lower():
            c["open_tickets"].append(
                {
                    "ticket_id": ticket_id,
                    "summary": issue_summary,
                    "priority": prio,
                    "status": "Open",
                }
            )
            ticket["linked_customer_id"] = c["customer_id"]
            ticket["linked_company"] = c["company"]
            break

    return ticket


@mcp.tool()
def get_platform_service_status(service_name: str = "all") -> dict[str, Any]:
    """Check real-time operational health and latency of cloud platform microservices.

    **Invocation Condition:** Invoke this tool when the user asks about platform
    status, service health, uptime, or microservice latency.

    Args:
        service_name: Specific service ('voice-gateway', 'payments-upi',
            'kyc-verifier', 'llm-router') or 'all' for full status board.
    """
    key = (service_name or "all").strip().lower()
    if key in ("all", "", "*"):
        degraded = [s["service"] for s in _SERVICES.values() if s["status"] != "operational"]
        return {
            "status": "ok",
            "retryable": False,
            "overall_status": "degraded" if degraded else "operational",
            "degraded_services": degraded,
            "checked_at": datetime.now(UTC).isoformat(),
            "services": list(_SERVICES.values()),
        }

    for name, info in _SERVICES.items():
        if key in name:
            return {"found": True, "retryable": False, **info}

    return {
        "status": "no_results",
        "retryable": False,
        "found": False,
        "requested": service_name,
        "available_services": list(_SERVICES.keys()),
        "message": (
            f"Service '{service_name}' does not exist. Do not retry with guessed names; "
            f"choose from available_services: {', '.join(_SERVICES.keys())}."
        ),
    }


# ------------------------------------------------------------------- HTTP app


async def _healthz(_request: Request) -> JSONResponse:
    return JSONResponse(
        {
            "status": "ok",
            "service": "live-ops-crm-mcp",
            "transport": "streamable_http",
            "endpoint": "/mcp",
            "timestamp": int(time.time()),
        }
    )


def create_app():
    app = mcp.streamable_http_app(
        host="0.0.0.0",
        streamable_http_path="/mcp",
        stateless_http=True,
    )
    app.routes.append(Route("/healthz", _healthz))
    app.routes.append(Route("/", _healthz))
    return app


app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)
