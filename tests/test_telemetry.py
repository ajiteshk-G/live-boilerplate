"""Unit tests for `gemini-live-telemetry` integration and per-turn latency tracking."""

from __future__ import annotations

from pathlib import Path

from gemini_live.settings.schema import AppConfig, TelemetrySection, VertexSection
from gemini_live.telemetry import (
    _percentile,
    activate_telemetry,
    get_telemetry_snapshot,
    sync_turn_latency_to_telemetry,
)


def _make_cfg(tmp_path: Path, *, enabled: bool = True) -> AppConfig:
    return AppConfig(
        vertex=VertexSection(project="mb-poc-352009", location="us-central1"),
        telemetry=TelemetrySection(
            enabled=enabled,
            enable_gcp_export=False,
            enable_dashboard=False,
            enable_json_export=True,
            metrics_dir=str(tmp_path / "metrics"),
            log_dir=str(tmp_path / "metrics" / "logs"),
        ),
    )


def test_percentile_computation() -> None:
    assert _percentile([], 50) is None
    assert _percentile([420.0], 95) == 420.0
    vals = [200.0, 300.0, 400.0, 500.0, 1000.0]
    assert _percentile(vals, 50) == 400.0
    p95 = _percentile(vals, 95)
    assert p95 is not None and 850.0 <= p95 <= 1000.0


def test_telemetry_disabled(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path, enabled=False)
    res = activate_telemetry(cfg)
    assert res["enabled"] is False
    snap = get_telemetry_snapshot(cfg)
    assert snap["enabled"] is False
    assert snap["aggregates"]["total_turns"] == 0


def test_telemetry_activation_and_turn_latency_sync(tmp_path: Path) -> None:
    import gemini_live_telemetry
    from gemini_live_telemetry.models import (
        SessionMetrics,
        ToolCallMetrics,
        TurnMetrics,
        UsageMetrics,
    )

    cfg = _make_cfg(tmp_path, enabled=True)
    status = activate_telemetry(cfg)
    assert status["enabled"] is True
    assert status["activated"] is True

    store = gemini_live_telemetry.get_metrics_store()
    assert store is not None

    sid = "test-session-latency-1"
    sm = SessionMetrics(session_id=sid, setup_latency_ms=312.5)
    sm.turns.append(
        TurnMetrics(
            turn_number=1,
            duration_ms=1450.0,
            ttfb_ms=8.0,  # simulated <25ms continuous-mic artifact
            was_interrupted=False,
            usage=UsageMetrics(
                prompt_token_count=120,
                response_token_count=45,
                total_token_count=165,
            ),
        )
    )
    sm.ttfb_values.append(8.0)
    sm.tool_calls.append(
        ToolCallMetrics(
            tool_id="call-1",
            tool_name="crm_ops__lookup_customer",
            round_trip_ms=218.4,
        )
    )
    store._sessions[sid] = sm

    # Backfill true user-perceived TTFB (e.g. 465.2 ms)
    sync_turn_latency_to_telemetry(
        None,
        turn_index=0,
        ttfb_ms=465.2,
        turn_duration_ms=1450.0,
        was_interrupted=False,
    )

    snap = get_telemetry_snapshot(cfg)
    assert snap["enabled"] is True
    agg = snap["aggregates"]
    assert agg["total_turns"] >= 1
    assert agg["avg_ttfb_ms"] is not None and agg["avg_ttfb_ms"] >= 100.0
    assert agg["p95_ttfb_ms"] is not None
    assert agg["avg_turn_duration_ms"] is not None
    assert agg["avg_setup_latency_ms"] is not None
    assert agg["avg_tool_round_trip_ms"] is not None
    assert any(t["ttfb_ms"] == 465.2 for t in snap["turns"])
