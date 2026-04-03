"""
Tests for ControlCore.registry.learning — LearningStore

Covers:
  1. record + query single call
  2. multiple records — avg, total, success rate
  3. unknown model returns empty stats
  4. p95 latency with 100 records
  5. stats filtered by intent
  6. stats filtered by time window
  7. refusal rate calculation
  8. avg cost calculation
  9. empty db returns zero stats
"""

from __future__ import annotations

import time

import pytest

from ControlCore.registry.learning import LearningStore, ModelStats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def store(tmp_path) -> LearningStore:
    """Return a fresh LearningStore backed by a temp file."""
    return LearningStore(db_path=str(tmp_path / "learning.db"))


# ---------------------------------------------------------------------------
# Test 1 — record + query single call
# ---------------------------------------------------------------------------

def test_record_and_query_single_call(tmp_path):
    s = store(tmp_path)
    s.record("gpt:4o", latency_ms=320.0, cost=0.005, outcome="success", intent="summarize")

    stats = s.stats("gpt:4o")

    assert stats.call_count == 1
    assert stats.avg_latency_ms == pytest.approx(320.0)
    assert stats.p95_latency_ms == pytest.approx(320.0)
    assert stats.total_cost == pytest.approx(0.005)
    assert stats.avg_cost == pytest.approx(0.005)
    assert stats.success_rate == pytest.approx(1.0)
    assert stats.refusal_rate == pytest.approx(0.0)
    s.close()


# ---------------------------------------------------------------------------
# Test 2 — multiple records: avg, total, success rate
# ---------------------------------------------------------------------------

def test_multiple_records_aggregates(tmp_path):
    s = store(tmp_path)
    calls = [
        (100.0, 0.01, "success"),
        (200.0, 0.02, "success"),
        (300.0, 0.03, "error"),
        (400.0, 0.04, "success"),
    ]
    for latency, cost, outcome in calls:
        s.record("mistral:7b", latency_ms=latency, cost=cost, outcome=outcome, intent="draft")

    stats = s.stats("mistral:7b")

    assert stats.call_count == 4
    assert stats.avg_latency_ms == pytest.approx(250.0)          # (100+200+300+400)/4
    assert stats.total_cost == pytest.approx(0.10)
    assert stats.avg_cost == pytest.approx(0.025)
    assert stats.success_rate == pytest.approx(0.75)              # 3/4
    s.close()


# ---------------------------------------------------------------------------
# Test 3 — unknown model returns empty stats
# ---------------------------------------------------------------------------

def test_unknown_model_returns_empty_stats(tmp_path):
    s = store(tmp_path)
    s.record("known:model", latency_ms=100.0, cost=0.001, outcome="success", intent="lookup")

    stats = s.stats("nonexistent:model")

    assert stats == ModelStats()
    assert stats.call_count == 0
    assert stats.avg_latency_ms == 0.0
    assert stats.success_rate == 0.0
    s.close()


# ---------------------------------------------------------------------------
# Test 4 — p95 latency with 100 records
# ---------------------------------------------------------------------------

def test_p95_latency_with_100_records(tmp_path):
    s = store(tmp_path)
    # Insert 100 records with latency 1..100 ms
    for i in range(1, 101):
        s.record("qwen:32b", latency_ms=float(i), cost=0.0, outcome="success", intent="reason")

    stats = s.stats("qwen:32b")

    assert stats.call_count == 100
    # OFFSET = int(100 * 0.95) = 95  → 0-indexed 96th value when sorted ascending = 96.0
    assert stats.p95_latency_ms == pytest.approx(96.0)
    s.close()


# ---------------------------------------------------------------------------
# Test 5 — stats filtered by intent
# ---------------------------------------------------------------------------

def test_stats_filtered_by_intent(tmp_path):
    s = store(tmp_path)
    s.record("claude:3", latency_ms=100.0, cost=0.01, outcome="success", intent="summarize")
    s.record("claude:3", latency_ms=200.0, cost=0.02, outcome="success", intent="summarize")
    s.record("claude:3", latency_ms=999.0, cost=0.99, outcome="error",   intent="extract")

    stats = s.stats("claude:3", intent="summarize")

    assert stats.call_count == 2
    assert stats.avg_latency_ms == pytest.approx(150.0)
    assert stats.total_cost == pytest.approx(0.03)
    assert stats.success_rate == pytest.approx(1.0)
    s.close()


# ---------------------------------------------------------------------------
# Test 6 — stats filtered by time window
# ---------------------------------------------------------------------------

def test_stats_filtered_by_time_window(tmp_path, monkeypatch):
    s = store(tmp_path)

    # Manually insert an old record by patching time.time
    old_ts = time.time() - 3600   # 1 hour ago
    s._conn.execute(
        "INSERT INTO calls (model_alias, latency_ms, cost, outcome, intent, timestamp) VALUES (?,?,?,?,?,?)",
        ("gpt:4o", 999.0, 0.99, "success", "lookup", old_ts),
    )
    s._conn.commit()

    # Insert a recent record through normal API
    s.record("gpt:4o", latency_ms=50.0, cost=0.005, outcome="success", intent="lookup")

    # Only the recent record should appear in a 60-second window
    stats = s.stats("gpt:4o", window_seconds=60.0)

    assert stats.call_count == 1
    assert stats.avg_latency_ms == pytest.approx(50.0)
    s.close()


# ---------------------------------------------------------------------------
# Test 7 — refusal rate calculation
# ---------------------------------------------------------------------------

def test_refusal_rate_calculation(tmp_path):
    s = store(tmp_path)
    outcomes = ["success", "refused", "refused", "success", "refused"]
    for outcome in outcomes:
        s.record("mixtral:8x7b", latency_ms=100.0, cost=0.0, outcome=outcome, intent="classify")

    stats = s.stats("mixtral:8x7b")

    assert stats.call_count == 5
    assert stats.refusal_rate == pytest.approx(0.6)   # 3/5
    assert stats.success_rate == pytest.approx(0.4)   # 2/5
    s.close()


# ---------------------------------------------------------------------------
# Test 8 — avg cost calculation
# ---------------------------------------------------------------------------

def test_avg_cost_calculation(tmp_path):
    s = store(tmp_path)
    costs = [0.001, 0.003, 0.005, 0.007]
    for cost in costs:
        s.record("llama:70b", latency_ms=50.0, cost=cost, outcome="success", intent="extract")

    stats = s.stats("llama:70b")

    assert stats.avg_cost == pytest.approx(0.004)           # (0.001+0.003+0.005+0.007)/4
    assert stats.total_cost == pytest.approx(0.016)
    s.close()


# ---------------------------------------------------------------------------
# Test 9 — empty db returns zero stats
# ---------------------------------------------------------------------------

def test_empty_db_returns_zero_stats(tmp_path):
    s = store(tmp_path)

    stats = s.stats("any:model")

    assert stats.call_count == 0
    assert stats.avg_latency_ms == 0.0
    assert stats.p95_latency_ms == 0.0
    assert stats.total_cost == 0.0
    assert stats.avg_cost == 0.0
    assert stats.success_rate == 0.0
    assert stats.refusal_rate == 0.0
    s.close()
