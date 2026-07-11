"""Metrics collector tests."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from prism_serve.metrics.collector import NullMetrics, MetricsCollector


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
    m = NullMetrics()
    task = asyncio.create_task(m.tick_loop())
    await asyncio.sleep(0)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


@pytest.mark.asyncio
async def test_null_metrics_flush():
    m = NullMetrics()
    await m.flush()


def test_collector_no_prometheus_still_callable():
    with patch.dict("sys.modules", {"prometheus_client": None}):
        import importlib
        import prism_serve.metrics.collector as mod
        importlib.reload(mod)
        mc = mod.MetricsCollector({})
        assert not mc._available

        mc.increment("kv_transfer_success_total")
        mc.gauge("active_requests", 5)
        mc.observe("request_ttft_ms", 10.0, labels={"state": "FINISHED"})

        importlib.reload(mod)


def _make_collector_with_mock_prometheus():
    mc = MetricsCollector.__new__(MetricsCollector)
    mc.config = {}
    mc._kv_usage_scrape_interval_s = 5.0
    mc._infer_client = None
    mc._governor = None
    mc._scheduler = None
    mc._available = True

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
    mc._prefill_retry = _mock_counter()
    mc._prefill_abort = _mock_counter()
    mc._decode_abort = _mock_counter()
    mc._control_message_error = _mock_counter()

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


def test_increment_stage_timeout_counters():
    mc = _make_collector_with_mock_prometheus()
    mc.increment("prefill_dispatch_retry_total")
    mc.increment("prefill_dispatch_abort_total", labels={"reason": "timeout"})
    mc.increment("decode_abort_total", labels={"reason": "timeout"})

    mc._prefill_retry.inc.assert_called_once()
    mc._prefill_abort.labels.assert_called_once_with(reason="timeout")
    mc._decode_abort.labels.assert_called_once_with(reason="timeout")


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


def test_set_scheduler_stores_reference():
    mc = _make_collector_with_mock_prometheus()
    scheduler = MagicMock()
    mc.set_scheduler(scheduler)
    assert mc._scheduler is scheduler


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
async def test_scrape_kv_usage_propagates_to_scheduler():
    mc = _make_collector_with_mock_prometheus()
    scheduler = MagicMock()
    mc.set_scheduler(scheduler)
    client = MagicMock()
    client.get_kv_usage_all = AsyncMock(return_value={"d-0": 0.91})
    mc.set_infer_client(client)

    await mc._scrape_kv_usage()

    scheduler.update_kv_usage.assert_called_once_with("d-0", 0.91)


@pytest.mark.asyncio
async def test_scrape_kv_usage_no_client_is_noop():
    mc = _make_collector_with_mock_prometheus()
    await mc._scrape_kv_usage()


@pytest.mark.asyncio
async def test_scrape_kv_usage_client_exception_is_swallowed():
    mc = _make_collector_with_mock_prometheus()
    client = MagicMock()
    client.get_kv_usage_all = AsyncMock(side_effect=RuntimeError("network error"))
    mc.set_infer_client(client)
    await mc._scrape_kv_usage()


@pytest.mark.asyncio
async def test_tick_loop_calls_scrape():
    mc = _make_collector_with_mock_prometheus()
    mc._kv_usage_scrape_interval_s = 0.001

    call_count = 0
    scraped_twice = asyncio.Event()

    async def fake_scrape():
        nonlocal call_count
        call_count += 1
        if call_count >= 2:
            scraped_twice.set()

    mc._scrape_kv_usage = fake_scrape

    task = asyncio.create_task(mc.tick_loop())
    await asyncio.wait_for(scraped_twice.wait(), timeout=0.5)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    assert call_count >= 2, f"expected at least 2 scrape calls, got {call_count}"


@pytest.mark.asyncio
async def test_flush_is_noop():
    mc = _make_collector_with_mock_prometheus()
    await mc.flush()
