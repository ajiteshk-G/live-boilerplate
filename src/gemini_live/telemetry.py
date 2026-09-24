"""Integration with `gemini-live-telemetry` for per-turn latency & Cloud Monitoring dashboards.

Wraps `gemini-live-telemetry` (https://github.com/kkrishnan90/gemini-live-telemetry)
to provide:
1. Automatic instrumentation of `google-genai` `AsyncLive.connect` and `AsyncSession`
   methods (`send_realtime_input`, `send_client_content`, `send_tool_response`, `receive`).
2. Auto-provisioning of the Google Cloud Monitoring dashboard (`"Gemini Live API Metrics"`).
3. OpenTelemetry metric export (`gemini_live.turn.ttfb_ms`, `gemini_live.turn.duration_ms`,
   `gemini_live.session.setup_latency_ms`, `gemini_live.tool.round_trip_ms`, etc.) and local
   JSONL snapshots (`./metrics/`).
4. Real-time telemetry aggregation (`get_telemetry_snapshot`) for the Web Console UI and CLI.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any

import google.auth

from .live.client import _resolve_credentials
from .settings.schema import AppConfig

log = logging.getLogger(__name__)

_ACTIVATED = False
_DASHBOARD_RESOURCE: str | None = None


def _percentile(sorted_vals: list[float], p: float) -> float | None:
    """Compute the p-th percentile (0..100) from a pre-sorted list of floats."""
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return round(sorted_vals[0], 2)
    idx = (p / 100.0) * (len(sorted_vals) - 1)
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return round(sorted_vals[lo], 2)
    frac = idx - lo
    return round(sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac, 2)


def _bridge_gcp_credentials(project: str | None) -> None:
    """Ensure `google.auth.default()` uses the resolved GCP credentials for `project`.

    `gemini-live-telemetry` calls `google.auth.default()` inside its OTel
    `CloudMonitoringMetricsExporter` and `DashboardsServiceClient`. On multi-account
    dev workstations, `_resolve_credentials` selects the account with access to
    `vertex.project`.
    """
    if not project:
        return
    creds = _resolve_credentials(project)
    if creds is None:
        return
    orig_default = google.auth.default

    def _patched_default(*args: Any, **kwargs: Any) -> tuple[Any, str | None]:
        try:
            return creds, project
        except Exception:
            return orig_default(*args, **kwargs)  # type: ignore[no-any-return]

    google.auth.default = _patched_default  # type: ignore[assignment]


_GCP_PROJECT: str | None = None
_ENABLE_GCP_EXPORT: bool = False


def _full_metric_type(metric_prefix: str, suffix: str) -> str:
    """Build a canonical Cloud Monitoring metric type string.

    `gemini-live-telemetry` uses `metric_prefix="workload.googleapis.com"` and
    names its metrics `workload.googleapis.com/gemini_live.<suffix>`.
    """
    prefix = (metric_prefix or "workload.googleapis.com").rstrip("/")
    if prefix.endswith("gemini_live"):
        return f"{prefix}.{suffix}"
    if "googleapis.com" in prefix:
        return f"{prefix}/gemini_live.{suffix}"
    return f"workload.googleapis.com/{prefix}.{suffix}"


def _export_turn_gauge_to_cloud_monitoring(
    *,
    project_id: str,
    metric_prefix: str,
    session_id: str,
    turn_number: int,
    ttfb_ms: float | None,
    turn_duration_ms: float | None,
) -> None:
    """Write per-turn GAUGE metrics (`turn.ttfb_ms` and `turn.duration_ms`) with
    `session_id` and `turn_number` labels directly to Google Cloud Monitoring."""
    try:
        from google.api import metric_pb2, monitored_resource_pb2
        from google.cloud import monitoring_v3
        from google.protobuf.timestamp_pb2 import Timestamp

        client = monitoring_v3.MetricServiceClient()
        project_name = f"projects/{project_id}"

        now = time.time()
        seconds = int(now)
        nanos = int((now - seconds) * 10**9)
        interval = monitoring_v3.TimeInterval(
            end_time=Timestamp(seconds=seconds, nanos=nanos)
        )

        series_list: list[Any] = []
        turn_label = f"turn_{turn_number}"

        resource_msg = monitored_resource_pb2.MonitoredResource(
            type="generic_task",
            labels={
                "project_id": project_id,
                "location": "global",
                "namespace": "",
                "job": "gemini-live-boilerplate",
                "task_id": session_id[:32] or "live-session",
            },
        )

        if ttfb_ms is not None:
            series_list.append(
                monitoring_v3.TimeSeries(
                    metric=metric_pb2.Metric(
                        type=_full_metric_type(metric_prefix, "turn.ttfb_ms"),
                        labels={
                            "session_id": session_id,
                            "turn_number": turn_label,
                        },
                    ),
                    resource=resource_msg,
                    metric_kind=metric_pb2.MetricDescriptor.MetricKind.GAUGE,
                    value_type=metric_pb2.MetricDescriptor.ValueType.DOUBLE,
                    points=[
                        monitoring_v3.Point(
                            interval=interval,
                            value=monitoring_v3.TypedValue(double_value=round(ttfb_ms, 2)),
                        )
                    ],
                )
            )

        if turn_duration_ms is not None:
            series_list.append(
                monitoring_v3.TimeSeries(
                    metric=metric_pb2.Metric(
                        type=_full_metric_type(metric_prefix, "turn.duration_ms"),
                        labels={
                            "session_id": session_id,
                            "turn_number": turn_label,
                        },
                    ),
                    resource=resource_msg,
                    metric_kind=metric_pb2.MetricDescriptor.MetricKind.GAUGE,
                    value_type=metric_pb2.MetricDescriptor.ValueType.DOUBLE,
                    points=[
                        monitoring_v3.Point(
                            interval=interval,
                            value=monitoring_v3.TypedValue(
                                double_value=round(turn_duration_ms, 2)
                            ),
                        )
                    ],
                )
            )

        if series_list:
            client.create_time_series(name=project_name, time_series=series_list)
    except Exception as exc:
        log.debug("Cloud Monitoring per-turn gauge export skipped: %s", exc)


def _ensure_per_turn_dashboard_widgets(
    project_id: str, dashboard_name: str, metric_prefix: str
) -> str | None:
    """Ensure the Cloud Monitoring dashboard retains ALL 12 original
    `gemini-live-telemetry` widgets in `grid_layout` AND adds Per-Turn TTFB &
    Duration widgets (`gemini_live.turn.ttfb_ms` & `gemini_live.turn.duration_ms`)."""
    if not project_id:
        return None
    try:
        import contextlib

        from gemini_live_telemetry._dashboard import _build_dashboard
        from google.api import label_pb2, metric_pb2
        from google.cloud import monitoring_v3
        from google.cloud.monitoring_dashboard_v1 import DashboardsServiceClient
        from google.cloud.monitoring_dashboard_v1 import types as dtypes
        from google.cloud.monitoring_dashboard_v1.types import UpdateDashboardRequest

        mclient = monitoring_v3.MetricServiceClient()
        for m_suffix, m_disp in (
            ("turn.ttfb_ms", "Per-Turn TTFB Latency (ms)"),
            ("turn.duration_ms", "Per-Turn Duration (ms)"),
        ):
            with contextlib.suppress(Exception):
                mclient.create_metric_descriptor(
                    name=f"projects/{project_id}",
                    metric_descriptor=metric_pb2.MetricDescriptor(
                        type=_full_metric_type(metric_prefix, m_suffix),
                        metric_kind=metric_pb2.MetricDescriptor.MetricKind.GAUGE,
                        value_type=metric_pb2.MetricDescriptor.ValueType.DOUBLE,
                        unit="ms",
                        display_name=m_disp,
                        description=f"{m_disp} indexed by session_id and turn_number",
                        labels=[
                            label_pb2.LabelDescriptor(
                                key="session_id",
                                value_type=label_pb2.LabelDescriptor.ValueType.STRING,
                                description="Gemini Live session ID",
                            ),
                            label_pb2.LabelDescriptor(
                                key="turn_number",
                                value_type=label_pb2.LabelDescriptor.ValueType.STRING,
                                description="1-based turn label (e.g. turn_1, turn_2)",
                            ),
                        ],
                    ),
                )

        client = DashboardsServiceClient()
        parent = f"projects/{project_id}"
        target = None
        for d in client.list_dashboards(parent=parent):
            if d.display_name == dashboard_name:
                target = d
                break
        if target is None:
            return None

        # Build the full 12-widget GridLayout dashboard from gemini-live-telemetry
        dash = _build_dashboard(dashboard_name, metric_prefix, dtypes)
        dash.name = target.name
        dash.etag = target.etag

        ttfb_type = _full_metric_type(metric_prefix, "turn.ttfb_ms")
        otel_ttfb_type = _full_metric_type(metric_prefix, "latency.ttfb_ms")
        dur_type = _full_metric_type(metric_prefix, "turn.duration_ms")
        otel_dur_type = _full_metric_type(metric_prefix, "latency.turn_duration_ms")

        legend_tpl = "${metric.labels.session_id} - ${metric.labels.turn_number}"
        otel_legend = "${metric.labels.session_id} (${metric.labels.vad_mode})"

        widget_per_turn_ttfb = dtypes.Widget(
            title="Per-Turn TTFB (ms) by Turn #",
            xy_chart=dtypes.XyChart(
                data_sets=[
                    dtypes.XyChart.DataSet(
                        legend_template=legend_tpl,
                        plot_type=dtypes.XyChart.DataSet.PlotType.LINE,
                        time_series_query=dtypes.TimeSeriesQuery(
                            time_series_filter=dtypes.TimeSeriesFilter(
                                filter=f'metric.type="{ttfb_type}"',
                                aggregation=dtypes.Aggregation(
                                    alignment_period={"seconds": 60},
                                    per_series_aligner=dtypes.Aggregation.Aligner.ALIGN_MAX,
                                    cross_series_reducer=dtypes.Aggregation.Reducer.REDUCE_MAX,
                                    group_by_fields=[
                                        "metric.label.session_id",
                                        "metric.label.turn_number",
                                    ],
                                ),
                            )
                        ),
                    ),
                    dtypes.XyChart.DataSet(
                        legend_template=otel_legend,
                        plot_type=dtypes.XyChart.DataSet.PlotType.LINE,
                        time_series_query=dtypes.TimeSeriesQuery(
                            time_series_filter=dtypes.TimeSeriesFilter(
                                filter=f'metric.type="{otel_ttfb_type}"',
                                aggregation=dtypes.Aggregation(
                                    alignment_period={"seconds": 60},
                                    per_series_aligner=dtypes.Aggregation.Aligner.ALIGN_DELTA,
                                    cross_series_reducer=dtypes.Aggregation.Reducer.REDUCE_MEAN,
                                    group_by_fields=[
                                        "metric.label.session_id",
                                        "metric.label.vad_mode",
                                    ],
                                ),
                            )
                        ),
                    ),
                ],
                y_axis=dtypes.XyChart.Axis(label="ms"),
            ),
        )

        widget_per_turn_dur = dtypes.Widget(
            title="Per-Turn Duration (ms) by Turn #",
            xy_chart=dtypes.XyChart(
                data_sets=[
                    dtypes.XyChart.DataSet(
                        legend_template=legend_tpl,
                        plot_type=dtypes.XyChart.DataSet.PlotType.LINE,
                        time_series_query=dtypes.TimeSeriesQuery(
                            time_series_filter=dtypes.TimeSeriesFilter(
                                filter=f'metric.type="{dur_type}"',
                                aggregation=dtypes.Aggregation(
                                    alignment_period={"seconds": 60},
                                    per_series_aligner=dtypes.Aggregation.Aligner.ALIGN_MAX,
                                    cross_series_reducer=dtypes.Aggregation.Reducer.REDUCE_MAX,
                                    group_by_fields=[
                                        "metric.label.session_id",
                                        "metric.label.turn_number",
                                    ],
                                ),
                            )
                        ),
                    ),
                    dtypes.XyChart.DataSet(
                        legend_template="${metric.labels.session_id}",
                        plot_type=dtypes.XyChart.DataSet.PlotType.LINE,
                        time_series_query=dtypes.TimeSeriesQuery(
                            time_series_filter=dtypes.TimeSeriesFilter(
                                filter=f'metric.type="{otel_dur_type}"',
                                aggregation=dtypes.Aggregation(
                                    alignment_period={"seconds": 60},
                                    per_series_aligner=dtypes.Aggregation.Aligner.ALIGN_DELTA,
                                    cross_series_reducer=dtypes.Aggregation.Reducer.REDUCE_MEAN,
                                    group_by_fields=["metric.label.session_id"],
                                ),
                            )
                        ),
                    ),
                ],
                y_axis=dtypes.XyChart.Axis(label="ms"),
            ),
        )

        widget_ttfb_percentiles = dtypes.Widget(
            title="TTFB Percentiles (P50 / P95 / Max)",
            xy_chart=dtypes.XyChart(
                data_sets=[
                    dtypes.XyChart.DataSet(
                        legend_template="P50 TTFB",
                        plot_type=dtypes.XyChart.DataSet.PlotType.LINE,
                        time_series_query=dtypes.TimeSeriesQuery(
                            time_series_filter=dtypes.TimeSeriesFilter(
                                filter=f'metric.type="{otel_ttfb_type}"',
                                aggregation=dtypes.Aggregation(
                                    alignment_period={"seconds": 60},
                                    per_series_aligner=dtypes.Aggregation.Aligner.ALIGN_DELTA,
                                    cross_series_reducer=dtypes.Aggregation.Reducer.REDUCE_PERCENTILE_50,
                                ),
                            )
                        ),
                    ),
                    dtypes.XyChart.DataSet(
                        legend_template="P95 TTFB",
                        plot_type=dtypes.XyChart.DataSet.PlotType.LINE,
                        time_series_query=dtypes.TimeSeriesQuery(
                            time_series_filter=dtypes.TimeSeriesFilter(
                                filter=f'metric.type="{otel_ttfb_type}"',
                                aggregation=dtypes.Aggregation(
                                    alignment_period={"seconds": 60},
                                    per_series_aligner=dtypes.Aggregation.Aligner.ALIGN_DELTA,
                                    cross_series_reducer=dtypes.Aggregation.Reducer.REDUCE_PERCENTILE_95,
                                ),
                            )
                        ),
                    ),
                ],
                y_axis=dtypes.XyChart.Axis(label="ms"),
            ),
        )

        # Keep all 12 original widgets in `dash.grid_layout.widgets` (6 scorecards + 6 charts)
        # and insert the 3 Per-Turn charts right after the 6 scorecards.
        orig_widgets = list(dash.grid_layout.widgets)
        combined_widgets = (
            orig_widgets[:6]
            + [widget_per_turn_ttfb, widget_per_turn_dur, widget_ttfb_percentiles]
            + orig_widgets[6:]
        )
        del dash.grid_layout.widgets[:]
        dash.grid_layout.widgets.extend(combined_widgets)

        res = client.update_dashboard(request=UpdateDashboardRequest(dashboard=dash))
        return str(res.name)
    except Exception as exc:
        log.debug("Could not patch per-turn widgets onto Cloud Monitoring dashboard: %s", exc)
        return None


def activate_telemetry(cfg: AppConfig) -> dict[str, Any]:
    """Activate `gemini-live-telemetry` once at process startup."""
    global _ACTIVATED, _DASHBOARD_RESOURCE, _GCP_PROJECT, _ENABLE_GCP_EXPORT
    tcfg = cfg.telemetry
    _GCP_PROJECT = cfg.vertex.project
    _ENABLE_GCP_EXPORT = bool(tcfg.enable_gcp_export)
    if not tcfg.enabled:
        return {"enabled": False, "activated": False}

    if _ACTIVATED:
        return {
            "enabled": True,
            "activated": True,
            "dashboard_name": tcfg.dashboard_name,
            "dashboard_url": tcfg.dashboard_url,
            "dashboard_resource": _DASHBOARD_RESOURCE,
        }

    try:
        import io
        import threading

        import gemini_live_telemetry
        import gemini_live_telemetry.otel as _glt_otel
        from gemini_live_telemetry.config import InstrumentationConfig
        from opentelemetry.sdk.metrics.export import ConsoleMetricExporter

        _bridge_gcp_credentials(cfg.vertex.project)
        _glt_otel.ConsoleMetricExporter = lambda *a, **kw: ConsoleMetricExporter(  # type: ignore[misc,assignment]
            *a, out=io.StringIO(), **kw
        )

        inst_cfg = InstrumentationConfig(
            project_id=cfg.vertex.project or "",
            metrics_dir=tcfg.metrics_dir,
            log_dir=tcfg.log_dir,
            export_interval_s=max(10.0, tcfg.export_interval_s),
            json_flush_interval_s=max(1.0, tcfg.json_flush_interval_s),
            enable_gcp_export=tcfg.enable_gcp_export,
            enable_json_export=tcfg.enable_json_export,
            enable_dashboard=tcfg.enable_dashboard,
            dashboard_name=tcfg.dashboard_name,
            metric_prefix=tcfg.metric_prefix,
            service_name=cfg.app.name,
        )
        gemini_live_telemetry.activate(inst_cfg)
        _ACTIVATED = True
        if tcfg.enable_dashboard and cfg.vertex.project:
            threading.Thread(
                target=_ensure_per_turn_dashboard_widgets,
                args=(cfg.vertex.project, tcfg.dashboard_name, tcfg.metric_prefix),
                daemon=True,
            ).start()
        log.info(
            "gemini-live-telemetry activated (project=%s, dashboard=%r, metrics_dir=%s)",
            cfg.vertex.project,
            tcfg.dashboard_name,
            tcfg.metrics_dir,
        )
    except Exception as exc:
        log.warning("Failed to activate gemini-live-telemetry: %s", exc)
        return {"enabled": True, "activated": False, "error": str(exc)}

    return {
        "enabled": True,
        "activated": _ACTIVATED,
        "dashboard_name": tcfg.dashboard_name,
        "dashboard_url": tcfg.dashboard_url,
        "dashboard_resource": _DASHBOARD_RESOURCE,
    }


def mark_session_input_timestamp(session: Any, ts: float | None = None) -> None:
    """Update `gemini-live-telemetry`'s `SessionTimingState` TTFB reference timestamp.

    `gemini-live-telemetry` prioritizes `last_vad_eos_time` over `last_audio_send_time`.
    Setting `last_vad_eos_time` when user speech transcription completes, text input is
    submitted, or a tool response is sent ensures that continuous 20ms silent mic chunks
    from the browser do not overwrite the TTFB reference point.
    """
    if session is None:
        return
    try:
        from gemini_live_telemetry._wrappers import get_session_map

        timing = get_session_map().get_timing(session)
        if timing is not None:
            wall_now = time.time()
            timing.last_vad_eos_time = wall_now
            timing.last_audio_send_time = wall_now
    except Exception:
        pass


def sync_turn_latency_to_telemetry(
    session: Any,
    *,
    turn_index: int,
    ttfb_ms: float | None,
    turn_duration_ms: float | None,
    was_interrupted: bool = False,
) -> None:
    """Ensure `gemini-live-telemetry` MetricsStore, OTel instruments, AppMetricsLogger,
    and Google Cloud Monitoring per-turn GAUGE time series have accurate turn latency
    (`ttfb_ms` and `duration_ms`)."""
    if ttfb_ms is None and turn_duration_ms is None:
        return
    try:
        import threading

        from gemini_live_telemetry import _instruments as inst
        from gemini_live_telemetry import get_app_logger, get_metrics_store
        from gemini_live_telemetry._wrappers import get_session_map
        from gemini_live_telemetry.models import SessionMetrics, TurnMetrics

        store = get_metrics_store()
        if store is None:
            return

        sid = get_session_map().get_session_id(session) if session is not None else None
        if sid is None:
            summaries = store.list_sessions()
            if summaries:
                sid = summaries[-1].session_id
        if sid is None:
            sid = "live-session"
            if not store.has_session(sid):
                store._sessions[sid] = SessionMetrics(session_id=sid)

        sm = store.get_session(sid)
        if not sm.turns:
            sm.turns.append(
                TurnMetrics(
                    turn_number=turn_index + 1,
                    duration_ms=round(turn_duration_ms, 2) if turn_duration_ms else None,
                    ttfb_ms=round(ttfb_ms, 2) if ttfb_ms else None,
                    was_interrupted=was_interrupted,
                )
            )
            if ttfb_ms is not None:
                sm.ttfb_values.append(round(ttfb_ms, 2))
        else:
            last_turn = sm.turns[-1]
            if ttfb_ms is not None:
                prev_ttfb = last_turn.ttfb_ms
                if prev_ttfb is None or prev_ttfb < 25.0:
                    last_turn.ttfb_ms = round(ttfb_ms, 2)
                    if (
                        prev_ttfb is not None
                        and sm.ttfb_values
                        and abs(sm.ttfb_values[-1] - prev_ttfb) < 0.01
                    ):
                        sm.ttfb_values[-1] = round(ttfb_ms, 2)
                    else:
                        sm.ttfb_values.append(round(ttfb_ms, 2))
            if turn_duration_ms is not None and (
                last_turn.duration_ms is None or last_turn.duration_ms <= 0.0
            ):
                last_turn.duration_ms = round(turn_duration_ms, 2)

        # Record in OpenTelemetry histograms with turn label in `vad_mode` & `session_id`
        turn_tag = f"turn_{turn_index + 1}"
        if ttfb_ms is not None and inst.latency_ttfb is not None:
            inst.latency_ttfb.record(
                round(ttfb_ms, 2),
                attributes={inst.ATTR_SESSION_ID: sid, "vad_mode": turn_tag},
            )
        if turn_duration_ms is not None and inst.latency_turn_duration is not None:
            inst.latency_turn_duration.record(
                round(turn_duration_ms, 2),
                attributes={inst.ATTR_SESSION_ID: f"{sid}/{turn_tag}"},
            )

        # Also export direct per-turn GAUGE time series (`gemini_live.turn.ttfb_ms` &
        # `gemini_live.turn.duration_ms`) labeled by (session_id, turn_number) so
        # Cloud Monitoring plots every individual turn without 15-60s histogram bucketing.
        if _ENABLE_GCP_EXPORT and _GCP_PROJECT:
            threading.Thread(
                target=_export_turn_gauge_to_cloud_monitoring,
                kwargs={
                    "project_id": _GCP_PROJECT,
                    "metric_prefix": "gemini_live",
                    "session_id": sid,
                    "turn_number": turn_index + 1,
                    "ttfb_ms": ttfb_ms,
                    "turn_duration_ms": turn_duration_ms,
                },
                daemon=True,
            ).start()

        app_logger = get_app_logger()
        if app_logger is not None and ttfb_ms is not None:
            app_logger.log_ttfb(
                session_id=sid,
                ttfb_ms=round(ttfb_ms, 2),
                vad_mode="native",
            )
    except Exception as exc:
        log.debug("sync_turn_latency_to_telemetry skipped: %s", exc)


def get_telemetry_snapshot(cfg: AppConfig) -> dict[str, Any]:
    """Return a comprehensive snapshot of `gemini-live-telemetry` metrics for the UI/CLI."""
    tcfg = cfg.telemetry
    base: dict[str, Any] = {
        "enabled": tcfg.enabled,
        "activated": _ACTIVATED,
        "package": "gemini-live-telemetry",
        "dashboard_name": tcfg.dashboard_name,
        "dashboard_url": tcfg.dashboard_url,
        "metric_prefix": tcfg.metric_prefix,
        "aggregates": {
            "total_sessions": 0,
            "active_sessions": 0,
            "total_turns": 0,
            "total_interruptions": 0,
            "interruption_rate": 0.0,
            "avg_ttfb_ms": None,
            "p50_ttfb_ms": None,
            "p95_ttfb_ms": None,
            "p99_ttfb_ms": None,
            "min_ttfb_ms": None,
            "max_ttfb_ms": None,
            "avg_turn_duration_ms": None,
            "p50_turn_duration_ms": None,
            "p95_turn_duration_ms": None,
            "avg_setup_latency_ms": None,
            "avg_tool_round_trip_ms": None,
            "total_tool_calls": 0,
            "total_tokens": 0,
            "total_audio_bytes_sent": 0,
            "total_audio_bytes_received": 0,
        },
        "turns": [],
        "tool_calls": [],
        "sessions": [],
    }
    if not tcfg.enabled or not _ACTIVATED:
        return base

    try:
        from gemini_live_telemetry import get_metrics_store

        store = get_metrics_store()
        global_agg = store.get_global_aggregates().to_dict()
        summaries = store.list_sessions()

        all_ttfbs: list[float] = []
        all_durations: list[float] = []
        all_setups: list[float] = []
        all_tool_rtts: list[float] = []
        recent_turns: list[dict[str, Any]] = []
        recent_tools: list[dict[str, Any]] = []
        session_summaries: list[dict[str, Any]] = []
        active_count = 0

        for summary in summaries:
            sid = summary.session_id
            sm = store.get_session(sid)
            s_agg = sm.compute_aggregates().to_dict()
            if sm.end_time is None:
                active_count += 1
            if sm.setup_latency_ms is not None:
                all_setups.append(sm.setup_latency_ms)

            for t in sm.turns:
                if t.ttfb_ms is not None:
                    all_ttfbs.append(t.ttfb_ms)
                if t.duration_ms is not None:
                    all_durations.append(t.duration_ms)
                recent_turns.append(
                    {
                        "session_id": sid,
                        "turn_number": t.turn_number,
                        "ttfb_ms": round(t.ttfb_ms, 2) if t.ttfb_ms is not None else None,
                        "duration_ms": (
                            round(t.duration_ms, 2) if t.duration_ms is not None else None
                        ),
                        "was_interrupted": t.was_interrupted,
                        "prompt_tokens": t.usage.prompt_token_count if t.usage else 0,
                        "response_tokens": t.usage.response_token_count if t.usage else 0,
                        "total_tokens": t.usage.total_token_count if t.usage else 0,
                    }
                )

            for tc in sm.tool_calls:
                if tc.round_trip_ms is not None:
                    all_tool_rtts.append(tc.round_trip_ms)
                recent_tools.append(
                    {
                        "session_id": sid,
                        "tool_id": tc.tool_id,
                        "function_names": [tc.tool_name],
                        "call_count": 1,
                        "round_trip_ms": (
                            round(tc.round_trip_ms, 2) if tc.round_trip_ms is not None else None
                        ),
                        "was_cancelled": tc.was_cancelled,
                    }
                )

            session_summaries.append(
                {
                    "session_id": sid,
                    "is_active": sm.end_time is None,
                    "setup_latency_ms": (
                        round(sm.setup_latency_ms, 2) if sm.setup_latency_ms is not None else None
                    ),
                    "total_turns": s_agg.get("total_turns", 0),
                    "avg_ttfb_ms": s_agg.get("avg_ttfb_ms"),
                    "p95_ttfb_ms": s_agg.get("p95_ttfb_ms"),
                    "avg_turn_duration_ms": s_agg.get("avg_turn_duration_ms"),
                    "total_tool_calls": s_agg.get("total_tool_calls", 0),
                    "avg_tool_round_trip_ms": s_agg.get("avg_tool_round_trip_ms"),
                    "total_tokens": s_agg.get("session_total_tokens", 0),
                }
            )

        all_ttfbs.sort()
        all_durations.sort()

        avg_setup = round(sum(all_setups) / len(all_setups), 2) if all_setups else None
        avg_tool_rtt = round(sum(all_tool_rtts) / len(all_tool_rtts), 2) if all_tool_rtts else None

        base["aggregates"] = {
            "total_sessions": global_agg.get("total_sessions", len(summaries)),
            "active_sessions": active_count,
            "total_turns": global_agg.get("total_turns", len(recent_turns)),
            "total_interruptions": global_agg.get("total_interrupted_turns", 0),
            "interruption_rate": round(global_agg.get("interruption_rate", 0.0), 4),
            "avg_ttfb_ms": (
                global_agg.get("avg_ttfb_ms")
                if global_agg.get("avg_ttfb_ms") is not None
                else _percentile(all_ttfbs, 50)
            ),
            "p50_ttfb_ms": global_agg.get("p50_ttfb_ms") or _percentile(all_ttfbs, 50),
            "p95_ttfb_ms": global_agg.get("p95_ttfb_ms") or _percentile(all_ttfbs, 95),
            "p99_ttfb_ms": global_agg.get("p99_ttfb_ms") or _percentile(all_ttfbs, 99),
            "min_ttfb_ms": round(all_ttfbs[0], 2) if all_ttfbs else None,
            "max_ttfb_ms": round(all_ttfbs[-1], 2) if all_ttfbs else None,
            "avg_turn_duration_ms": (
                global_agg.get("avg_turn_duration_ms")
                if global_agg.get("avg_turn_duration_ms") is not None
                else _percentile(all_durations, 50)
            ),
            "p50_turn_duration_ms": _percentile(all_durations, 50),
            "p95_turn_duration_ms": _percentile(all_durations, 95),
            "avg_setup_latency_ms": avg_setup,
            "avg_tool_round_trip_ms": global_agg.get("avg_tool_round_trip_ms") or avg_tool_rtt,
            "total_tool_calls": global_agg.get("total_tool_calls", len(recent_tools)),
            "total_tokens": global_agg.get("total_tokens", 0),
            "total_audio_bytes_sent": global_agg.get("total_audio_sent_bytes", 0),
            "total_audio_bytes_received": global_agg.get("total_audio_received_bytes", 0),
        }
        base["turns"] = recent_turns[-50:]
        base["tool_calls"] = recent_tools[-30:]
        base["sessions"] = session_summaries[-20:]

        # If in-memory store has no turns yet (e.g. right after a server start),
        # hydrate from the latest non-empty persisted JSON snapshot in metrics_dir.
        if not base["turns"]:
            import json
            from pathlib import Path

            mpath = Path(tcfg.metrics_dir)
            if mpath.exists():
                for f in reversed(sorted(mpath.glob("metrics_*.json"))):
                    try:
                        parsed = json.loads(f.read_text())
                        ga = parsed.get("global_aggregates") or {}
                        if ga.get("total_turns", 0) > 0:
                            for k in (
                                "total_sessions",
                                "total_turns",
                                "avg_ttfb_ms",
                                "p50_ttfb_ms",
                                "p95_ttfb_ms",
                                "p99_ttfb_ms",
                                "avg_turn_duration_ms",
                                "avg_tool_round_trip_ms",
                                "total_tool_calls",
                                "total_tokens",
                            ):
                                if ga.get(k) is not None:
                                    base["aggregates"][k] = ga[k]
                            for sid_k, s_obj in (parsed.get("sessions") or {}).items():
                                if s_obj.get("setup_latency_ms") is not None and (
                                    base["aggregates"]["avg_setup_latency_ms"] is None
                                ):
                                    base["aggregates"]["avg_setup_latency_ms"] = round(
                                        s_obj["setup_latency_ms"], 2
                                    )
                                for pt in s_obj.get("turns") or []:
                                    base["turns"].append(
                                        {
                                            "session_id": sid_k,
                                            "turn_number": pt.get("turn_number", 1),
                                            "ttfb_ms": pt.get("ttfb_ms"),
                                            "duration_ms": pt.get("duration_ms"),
                                            "was_interrupted": pt.get(
                                                "was_interrupted", False
                                            ),
                                        }
                                    )
                                for ptc in s_obj.get("tool_calls") or []:
                                    base["tool_calls"].append(
                                        {
                                            "session_id": sid_k,
                                            "tool_id": ptc.get("tool_id", ""),
                                            "function_names": [
                                                ptc.get("tool_name", "?")
                                            ],
                                            "call_count": 1,
                                            "round_trip_ms": ptc.get("round_trip_ms"),
                                            "was_cancelled": ptc.get(
                                                "was_cancelled", False
                                            ),
                                        }
                                    )
                            break
                    except Exception:
                        continue
    except Exception as exc:
        log.debug("get_telemetry_snapshot failed: %s", exc)
        base["error"] = str(exc)

    return base
