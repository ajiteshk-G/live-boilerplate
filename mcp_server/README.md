# Custom MCP Server for Gemini Live (Cloud Run)

This directory contains a standalone, Cloud Run-ready **Model Context Protocol (MCP)** server (`live-ops-crm`) built with `mcp.server.mcpserver.MCPServer` and `stateless_http=True`.

## Tools Exposed (`/mcp` via `streamable_http`)

1. **`lookup_customer_account(query: str)`**
   Searches enterprise accounts (`CUST-1001` Rajesh Sharma / IndTech Solutions, `CUST-1002` Priya Nair / BharatCloud AI, `CUST-1003` Arjun Mehta / FinSetu Payments, `CUST-1004` Ananya Iyer / Prana Health) by ID, name, company, or email.
2. **`calculate_loan_emi(principal_inr: float, annual_interest_rate_percent: float, tenure_months: int)`**
   Computes monthly EMI, total interest, and total repayment formatted in INR (Lakhs / Crores) for natural voice readout.
3. **`create_support_ticket(customer_id_or_name: str, issue_summary: str, priority: str = "normal")`**
   Creates a new support ticket (`TKT-xxxx`), attaches it to the customer record, and returns SLA response targets.
4. **`get_platform_service_status(service_name: str = "all")`**
   Reports real-time health, latency, and region status (`asia-south1 Mumbai`, `asia-south2 Delhi`, `us-central1 Iowa`) for platform microservices.

## Local Run

```bash
PORT=8081 uv run python mcp_server/server.py
```

## Deploy to Cloud Run

```bash
gcloud run deploy gemini-live-mcp-tools \
  --source ./mcp_server \
  --region us-central1 \
  --project mb-poc-352009 \
  --allow-unauthenticated
```
