"""Unit tests for metrics/collector.py.

Covers NullMetrics (all methods are no-ops) and MetricsCollector's public
interface (increment / gauge / observe) in both the prometheus_client-available
and unavailable cases.  The tick_loop and _scrape_kv_usage paths are tested
via asyncio without a real Prometheus server.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from prism_serve.metrics.collector import NullMetrics, MetricsCollector


# ---------------------------------------------------------------------------
# NullMetrics — all operations must be silent no-ops
# ---------------------------------------------------------------------------

def test_null_metrics_increment():
    m = NullMetrics()
    m.increment("any_counter")
    m.increment("any_counter", labels={"k": "v"})


def test_null_metrics_gauge():
    m = NullMetrics()
    m.gauge("any_gauge", 42.0)
    m.gauge("any_gauge", 1.0, labels={"dst": "d-0"})


def test_null_metrics_observe():
    m = NullMetrics()
    m.observe("any_histogram", 5.12)
    m.observe("any_histogram", 0.0, labels={"state": "FINISHED"})


@pytest.mark.asyncio
async def test_null_metrics_tick_loop_is_coroutine():
    """tick_loop must be an awaitable coroutine (not a plain function)."""
    m = NullMetrics()
    task = asyncio.create_task(m.tick_loop())
    await asyncio.sleep(0)   # let it start
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


@pytest.mark.asyncio
async def test_null_metrics_flush():
    m = NullMetrics()
    await m.flush()   # must not raise


# ---------------------------------------------------------------------------
# MetricsCollector — prometheus_client not available (graceful degradation)
# ---------------------------------------------------------------------------

def test_collector_no_prometheus_still_callable():
    """When prometheus_client is absent, all public methods must be no-ops."""
    with patch.dict("sys.modules", {"prometheus_client": None}):
        # Re-import to trigger the ImportError branch
        import importlib
        import prism_serve.metrics.collector as mod
        importlib.reload(mod)
        mc = mod.MetricsCollector({})
        assert not mc._available

        mc.increment("kv_transfer_success_total")
        mc.gauge("active_requests", 5)
        mc.observe("request_ttft_ms", 10.0, labels={"state": "FINISHED"})

        # Reload to restore original module state for other tests
        importlib.reload(mod)


# ---------------------------------------------------------------------------
# MetricsCollector — prometheus_client available (happy path)
# ---------------------------------------------------------------------------

def _make_collector_with_mock_prometheus():
    """Return a MetricsCollector whose prometheus objects are all MagicMocks."""
    mc = MetricsCollector.__new__(MetricsCollector)
    mc.config = {}
    mc._kv_usage_scrape_interval_s = 5.0
    mc._infer_client = None
    mc._governor = None
    mc._available = True

    # Stub all prometheus objects
    def _mock_counter():
        c = MagicMock()
        c.labels.return_value = c
        return c

    def _mock_gauge():
        g = MagicMock()
        g.labels.return_value = g
        return g

    def _mock_histogram():
        h = MagicMock()
        h.labels.return_value = h
        return h

    mc._kv_success   = _mock_counter()
    mc._kv_recompute = _mock_counter()
    mc._kv_abort     = _mock_counter()
    mc._kv_congestion = _mock_counter()

    mc._active_reqs    = _mock_gauge()
    mc._waiting_reqs   = _mock_gauge()
    mc._kv_pending_reqs = _mock_gauge()
    mc._deferred_depth = _mock_gauge()
    mc._slot_util      = _mock_gauge()
    mc._stale_slots    = _mock_gauge()

    mc._ttft = _mock_histogram()
    return mc


def test_increment_known_counter_no_labels():
    mc = _make_collector_with_mock_prometheus()
    mc.increment("kv_transfer_success_total")
    mc._kv_success.inc.assert_called_once()


def test_increment_known_counter_with_labels():
    mc = _make_collector_with_mock_prometheus()
    mc.increment("kv_transfer_recompute_total", labels={"reason": "timeout", "attempt": "1"})
    mc._kv_recompute.labels.assert_called_once_with(reason="timeout", attempt="1")
    mc._kv_recompute.labels.return_value.inc.assert_called_once()


def test_increment_unknown_name_is_noop():
    mc = _make_collector_with_mock_prometheus()
    mc.increment("nonexistent_counter")   # must not raise


def test_gauge_known_name_no_labels():
    mc = _make_collector_with_mock_prometheus()
    mc.gauge("active_requests", 7)
    mc._active_reqs.set.assert_called_once_with(7)


def test_gauge_known_name_with_labels():
    mc = _make_collector_with_mock_prometheus()
    mc.gauge("deferred_queue_depth", 3, labels={"dst": "d-0"})
    mc._deferred_depth.labels.assert_called_once_with(dst="d-0")
    mc._deferred_depth.labels.return_value.set.assert_called_once_with(3)


def test_gauge_unknown_name_is_noop():
    mc = _make_collector_with_mock_prometheus()
    mc.gauge("not_a_real_gauge", 99)   # must not raise


def test_observe_ttft_with_labels():
    mc = _make_collector_with_mock_prometheus()
    mc.observe("request_ttft_ms", 5.12, labels={"state": "FINISHED"})
    mc._ttft.labels.assert_called_once_with(state="FINISHED")
    mc._ttft.labels.return_value.observe.assert_called_once_with(5.12)


def test_observe_unknown_name_is_noop():
    mc = _make_collector_with_mock_prometheus()
    mc.observe("no_such_histogram", 1.0)   # must not raise


# ---------------------------------------------------------------------------
# set_governor / set_infer_client
# ---------------------------------------------------------------------------

def test_set_governor_stores_reference():
    mc = _make_collector_with_mock_prometheus()
    gov = MagicMock()
    mc.set_governor(gov)
    assert mc._governor is gov


def test_set_infer_client_stores_reference():
    mc = _make_collector_with_mock_prometheus()
    client = MagicMock()
    mc.set_infer_client(client)
    assert mc._infer_client is client


# ---------------------------------------------------------------------------
# _scrape_kv_usage — propagates to governor + slot_util gauge
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_scrape_kv_usage_propagates_to_governor():
    mc = _make_collector_with_mock_prometheus()
    gov = MagicMock()
    mc.set_governor(gov)

    client = MagicMock()
    client.get_kv_usage_all = AsyncMock(return_value={"d-0": 0.72, "d-1": 0.45})
    mc.set_infer_client(client)

    await mc._scrape_kv_usage()

    gov.update_kv_usage.assert_any_call("d-0", 0.72)
    gov.update_kv_usage.assert_any_call("d-1", 0.45)


@pytest.mark.asyncio
async def test_scrape_kv_usage_no_client_is_noop():
    mc = _make_collector_with_mock_prometheus()
    # _infer_client is None by default
    await mc._scrape_kv_usage()   # must not raise


@pytest.mark.asyncio
async def test_scrape_kv_usage_client_exception_is_swallowed():
    mc = _make_collector_with_mock_prometheus()
    client = MagicMock()
    client.get_kv_usage_all = AsyncMock(side_effect=RuntimeError("network error"))
    mc.set_infer_client(client)
    await mc._scrape_kv_usage()   # must not raise


# ---------------------------------------------------------------------------
# tick_loop — runs and calls _scrape_kv_usage periodically
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_tick_loop_calls_scrape():
    mc = _make_collector_with_mock_prometheus()
    mc._kv_usage_scrape_interval_s = 0.001   # 1 ms for test speed

    call_count = 0

    async def fake_scrape():
        nonlocal call_count
        call_count += 1

    mc._scrape_kv_usage = fake_scrape

    task = asyncio.create_task(mc.tick_loop())
    await asyncio.sleep(0.015)   # ~15 ms → at least a few scrape calls
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    assert call_count >= 2, f"expected ≥2 scrape calls, got {call_count}"


# ---------------------------------------------------------------------------
# flush
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_flush_is_noop():
    mc = _make_collector_with_mock_prometheus()
    await mc.flush()   # must not raise
