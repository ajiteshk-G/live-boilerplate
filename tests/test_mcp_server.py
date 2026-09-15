"""Tests for the custom Cloud Run MCP toolset (mcp_server/server.py)."""

from __future__ import annotations

import asyncio

import uvicorn

from gemini_live.settings.schema import McpServerConfig
from gemini_live.tools.mcp_toolset import McpToolsetManager
from mcp_server.server import (
    app,
    calculate_loan_emi,
    create_support_ticket,
    get_platform_service_status,
    lookup_customer_account,
)


def test_lookup_customer_account_by_name_and_id():
    res = lookup_customer_account("Rajesh")
    assert res["found"] is True
    assert res["customer"]["customer_id"] == "CUST-1001"
    assert "48.00 Lakh" in res["customer"]["annual_contract_formatted"]

    all_res = lookup_customer_account("all")
    assert all_res["count"] == 4


def test_calculate_loan_emi_matches_standard_amortization():
    res = calculate_loan_emi(
        principal_inr=25_00_000,
        annual_interest_rate_percent=8.5,
        tenure_months=60,
    )
    assert res["monthly_emi_inr"] == 51291.33
    assert res["monthly_emi_formatted"] == "₹51,291/month"


def test_create_support_ticket_attaches_to_customer():
    ticket = create_support_ticket(
        customer_id_or_name="CUST-1002",
        issue_summary="Need dedicated VPC peering for Hyderabad cluster",
        priority="high",
    )
    assert ticket["ticket_id"].startswith("TKT-")
    assert ticket["linked_customer_id"] == "CUST-1002"
    assert ticket["sla_response_target"] == "1 hour"


def test_get_platform_service_status_board_and_single():
    board = get_platform_service_status("all")
    assert board["overall_status"] == "degraded"
    assert "kyc-verifier" in board["degraded_services"]

    single = get_platform_service_status("voice-gateway")
    assert single["found"] is True
    assert single["status"] == "operational"


async def test_mcp_toolset_manager_connects_over_streamable_http():
    """Verify McpToolsetManager discovers and invokes all 4 tools over HTTP."""
    config = uvicorn.Config(app, host="127.0.0.1", port=18995, log_level="error")
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    await asyncio.sleep(0.4)
    try:
        srv_cfg = McpServerConfig(
            name="crm_ops",
            transport="streamable_http",
            url="http://127.0.0.1:18995/mcp",
        )
        async with McpToolsetManager([srv_cfg]) as mgr:
            names = sorted(c.exposed_name for c in mgr.candidates)
            assert names == [
                "crm_ops__calculate_loan_emi",
                "crm_ops__create_support_ticket",
                "crm_ops__get_platform_service_status",
                "crm_ops__lookup_customer_account",
            ]
            by_name = {c.exposed_name: c for c in mgr.candidates}
            out = await by_name["crm_ops__calculate_loan_emi"].invoke(
                {
                    "principal_inr": 10_00_000,
                    "annual_interest_rate_percent": 9.0,
                    "tenure_months": 12,
                }
            )
            assert "87,451" in out["result"]
    finally:
        server.should_exit = True
        await task
