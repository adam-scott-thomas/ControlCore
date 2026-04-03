"""
Tests for CircuitBreaker state machine and CircuitBreakerRegistry.

Covers:
- Initial state (CLOSED)
- Allows requests when CLOSED
- Opens after failure threshold
- Rejects requests when OPEN
- Transitions to HALF_OPEN after recovery timeout
- Closes after sufficient successes in HALF_OPEN
- Reopens on any failure in HALF_OPEN
- Registry creates distinct breakers per adapter+model key
- Registry returns the same breaker for the same key
"""

from __future__ import annotations

import time

import pytest

from ghostrouter.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerRegistry,
    CircuitConfig,
    CircuitOpenError,
    CircuitState,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_breaker(
    *,
    failure_threshold: int = 3,
    failure_window_seconds: float = 60.0,
    recovery_timeout_seconds: float = 0.1,  # short for tests
    success_threshold: int = 2,
    name: str = "test:model",
) -> CircuitBreaker:
    """Create a CircuitBreaker with test-friendly defaults."""
    config = CircuitConfig(
        failure_threshold=failure_threshold,
        failure_window_seconds=failure_window_seconds,
        recovery_timeout_seconds=recovery_timeout_seconds,
        success_threshold=success_threshold,
    )
    return CircuitBreaker(name=name, config=config)


def trip_breaker(cb: CircuitBreaker, count: int | None = None) -> None:
    """Record enough failures to open the circuit."""
    n = count if count is not None else cb.config.failure_threshold
    for _ in range(n):
        cb.record_failure()


# ---------------------------------------------------------------------------
# 1. Initial state
# ---------------------------------------------------------------------------

class TestInitialState:
    def test_starts_closed(self):
        cb = make_breaker()
        assert cb.state == CircuitState.CLOSED

    def test_is_open_false_when_closed(self):
        cb = make_breaker()
        assert cb.is_open() is False


# ---------------------------------------------------------------------------
# 2. Allows requests when CLOSED
# ---------------------------------------------------------------------------

class TestClosedAllowsRequests:
    def test_allow_request_returns_true_when_closed(self):
        cb = make_breaker()
        assert cb.allow_request() is True

    def test_repeated_allow_requests_while_closed(self):
        cb = make_breaker(failure_threshold=5)
        for _ in range(10):
            assert cb.allow_request() is True

    def test_failures_below_threshold_keep_circuit_closed(self):
        cb = make_breaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED
        assert cb.allow_request() is True


# ---------------------------------------------------------------------------
# 3. Opens after failure threshold
# ---------------------------------------------------------------------------

class TestOpensAfterThreshold:
    def test_opens_at_exact_threshold(self):
        cb = make_breaker(failure_threshold=3)
        trip_breaker(cb, 3)
        assert cb.state == CircuitState.OPEN

    def test_is_open_true_after_threshold(self):
        cb = make_breaker(failure_threshold=3)
        trip_breaker(cb, 3)
        assert cb.is_open() is True

    def test_stats_reflect_failures(self):
        cb = make_breaker(failure_threshold=3)
        trip_breaker(cb, 3)
        stats = cb.stats
        assert stats.failed_calls == 3


# ---------------------------------------------------------------------------
# 4. Rejects requests when OPEN
# ---------------------------------------------------------------------------

class TestOpenRejectsRequests:
    def test_allow_request_false_when_open(self):
        cb = make_breaker(failure_threshold=3, recovery_timeout_seconds=60.0)
        trip_breaker(cb, 3)
        assert cb.allow_request() is False

    def test_rejected_calls_incremented(self):
        cb = make_breaker(failure_threshold=3, recovery_timeout_seconds=60.0)
        trip_breaker(cb, 3)
        cb.allow_request()
        cb.allow_request()
        assert cb.stats.rejected_calls == 2

    def test_state_remains_open_without_timeout(self):
        cb = make_breaker(failure_threshold=3, recovery_timeout_seconds=60.0)
        trip_breaker(cb, 3)
        assert cb.state == CircuitState.OPEN


# ---------------------------------------------------------------------------
# 5. Transitions to HALF_OPEN after recovery timeout
# ---------------------------------------------------------------------------

class TestHalfOpenTransition:
    def test_transitions_to_half_open_after_timeout(self):
        cb = make_breaker(failure_threshold=3, recovery_timeout_seconds=0.05)
        trip_breaker(cb, 3)
        assert cb.state == CircuitState.OPEN
        time.sleep(0.1)
        # State check triggers _maybe_transition
        assert cb.state == CircuitState.HALF_OPEN

    def test_allow_request_true_in_half_open(self):
        cb = make_breaker(failure_threshold=3, recovery_timeout_seconds=0.05)
        trip_breaker(cb, 3)
        time.sleep(0.1)
        assert cb.allow_request() is True


# ---------------------------------------------------------------------------
# 6. Closes after half-open successes
# ---------------------------------------------------------------------------

class TestClosesAfterHalfOpenSuccess:
    def test_closes_after_success_threshold(self):
        cb = make_breaker(
            failure_threshold=3,
            recovery_timeout_seconds=0.05,
            success_threshold=2,
        )
        trip_breaker(cb, 3)
        time.sleep(0.1)
        # Trigger the OPEN→HALF_OPEN transition via state check
        assert cb.state == CircuitState.HALF_OPEN
        cb.record_success()
        assert cb.state == CircuitState.HALF_OPEN  # not yet
        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_failure_times_cleared_on_close(self):
        cb = make_breaker(
            failure_threshold=3,
            recovery_timeout_seconds=0.05,
            success_threshold=2,
        )
        trip_breaker(cb, 3)
        time.sleep(0.1)
        # Trigger the OPEN→HALF_OPEN transition before recording successes
        assert cb.state == CircuitState.HALF_OPEN
        cb.record_success()
        cb.record_success()
        assert cb.state == CircuitState.CLOSED
        # After closing, requests should be allowed again
        assert cb.allow_request() is True

    def test_single_success_below_threshold_stays_half_open(self):
        cb = make_breaker(
            failure_threshold=3,
            recovery_timeout_seconds=0.05,
            success_threshold=3,
        )
        trip_breaker(cb, 3)
        time.sleep(0.1)
        # Trigger the OPEN→HALF_OPEN transition before recording successes
        assert cb.state == CircuitState.HALF_OPEN
        cb.record_success()
        cb.record_success()
        assert cb.state == CircuitState.HALF_OPEN


# ---------------------------------------------------------------------------
# 7. Reopens on half-open failure
# ---------------------------------------------------------------------------

class TestReopensOnHalfOpenFailure:
    def test_reopens_on_failure_in_half_open(self):
        cb = make_breaker(failure_threshold=3, recovery_timeout_seconds=0.05)
        trip_breaker(cb, 3)
        time.sleep(0.1)
        # Trigger OPEN→HALF_OPEN first
        assert cb.state == CircuitState.HALF_OPEN
        cb.record_failure()
        # Immediately check _state before recovery timeout elapses again.
        # Use the internal _state directly to avoid _maybe_transition re-firing.
        assert cb._state == CircuitState.OPEN

    def test_rejects_requests_after_reopening(self):
        # Use a longer recovery timeout so the re-opened OPEN state persists
        # long enough for the allow_request() call to see it.
        cb = make_breaker(
            failure_threshold=3,
            recovery_timeout_seconds=60.0,
        )
        trip_breaker(cb, 3)
        # At this point circuit is OPEN (no sleep needed; recovery=60s)
        assert cb.state == CircuitState.OPEN
        # Force to HALF_OPEN by adjusting state_changed_at to simulate elapsed time
        with cb._lock:
            cb._state = CircuitState.HALF_OPEN
            cb._half_open_successes = 0
        assert cb._state == CircuitState.HALF_OPEN
        cb.record_failure()  # Failure in half-open → reopens to OPEN (recovery=60s)
        # Now OPEN with 60s timeout: allow_request must reject
        assert cb.allow_request() is False

    def test_half_open_successes_reset_after_reopen(self):
        """Reopen should reset the half-open success counter."""
        cb = make_breaker(
            failure_threshold=3,
            recovery_timeout_seconds=0.05,
            success_threshold=2,
        )
        trip_breaker(cb, 3)
        time.sleep(0.1)
        # Trigger initial OPEN→HALF_OPEN
        assert cb.state == CircuitState.HALF_OPEN
        cb.record_success()  # 1 success, then failure reopens
        cb.record_failure()  # reopens to OPEN, _state set without triggering timeout check
        assert cb._state == CircuitState.OPEN
        # Recover again: sleep past the new recovery timeout
        time.sleep(0.1)
        # Trigger OPEN→HALF_OPEN via state access
        assert cb.state == CircuitState.HALF_OPEN
        cb.record_success()
        # Still needs one more success (counter was reset to 0 on reopen)
        assert cb.state == CircuitState.HALF_OPEN
        cb.record_success()
        assert cb.state == CircuitState.CLOSED


# ---------------------------------------------------------------------------
# 8 & 9. Registry: per-key isolation and identity
# ---------------------------------------------------------------------------

class TestCircuitBreakerRegistry:
    def test_registry_creates_breaker_for_key(self):
        registry = CircuitBreakerRegistry()
        cb = registry.get_circuit("openai", "gpt-4")
        assert isinstance(cb, CircuitBreaker)
        assert cb.name == "openai:gpt-4"

    def test_registry_returns_same_breaker_for_same_key(self):
        registry = CircuitBreakerRegistry()
        cb1 = registry.get_circuit("openai", "gpt-4")
        cb2 = registry.get_circuit("openai", "gpt-4")
        assert cb1 is cb2

    def test_registry_creates_distinct_breakers_per_key(self):
        registry = CircuitBreakerRegistry()
        cb_a = registry.get_circuit("openai", "gpt-4")
        cb_b = registry.get_circuit("anthropic", "claude-3")
        assert cb_a is not cb_b
        assert cb_a.name == "openai:gpt-4"
        assert cb_b.name == "anthropic:claude-3"

    def test_registry_different_models_same_adapter_are_distinct(self):
        registry = CircuitBreakerRegistry()
        cb1 = registry.get_circuit("openai", "gpt-4")
        cb2 = registry.get_circuit("openai", "gpt-3.5")
        assert cb1 is not cb2

    def test_registry_state_is_isolated_between_keys(self):
        config = CircuitConfig(failure_threshold=2, recovery_timeout_seconds=60.0)
        registry = CircuitBreakerRegistry(default_config=config)
        cb_a = registry.get_circuit("openai", "gpt-4")
        cb_b = registry.get_circuit("anthropic", "claude-3")

        # Trip only cb_a
        cb_a.record_failure()
        cb_a.record_failure()

        assert cb_a.state == CircuitState.OPEN
        assert cb_b.state == CircuitState.CLOSED

    def test_registry_get_all_circuits(self):
        registry = CircuitBreakerRegistry()
        registry.get_circuit("openai", "gpt-4")
        registry.get_circuit("anthropic", "claude-3")
        all_circuits = registry.get_all_circuits()
        assert "openai:gpt-4" in all_circuits
        assert "anthropic:claude-3" in all_circuits

    def test_registry_get_open_circuits(self):
        config = CircuitConfig(failure_threshold=1, recovery_timeout_seconds=60.0)
        registry = CircuitBreakerRegistry(default_config=config)
        cb = registry.get_circuit("openai", "gpt-4")
        registry.get_circuit("anthropic", "claude-3")

        cb.record_failure()  # trips openai:gpt-4

        open_circuits = registry.get_open_circuits()
        assert "openai:gpt-4" in open_circuits
        assert "anthropic:claude-3" not in open_circuits

    def test_registry_reset_all(self):
        config = CircuitConfig(failure_threshold=1, recovery_timeout_seconds=60.0)
        registry = CircuitBreakerRegistry(default_config=config)
        cb = registry.get_circuit("openai", "gpt-4")
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        registry.reset_all()
        assert cb.state == CircuitState.CLOSED

    def test_registry_custom_config_per_circuit(self):
        registry = CircuitBreakerRegistry()
        custom = CircuitConfig(failure_threshold=1)
        cb = registry.get_circuit("openai", "gpt-4", config=custom)
        assert cb.config.failure_threshold == 1


# ---------------------------------------------------------------------------
# Extra: CircuitOpenError
# ---------------------------------------------------------------------------

class TestCircuitOpenError:
    def test_circuit_open_error_message(self):
        err = CircuitOpenError("openai:gpt-4")
        assert "openai:gpt-4" in str(err)
        assert err.circuit_name == "openai:gpt-4"


# ---------------------------------------------------------------------------
# Extra: manual reset
# ---------------------------------------------------------------------------

class TestManualReset:
    def test_reset_closes_open_circuit(self):
        cb = make_breaker(failure_threshold=2, recovery_timeout_seconds=60.0)
        trip_breaker(cb, 2)
        assert cb.state == CircuitState.OPEN
        cb.reset()
        assert cb.state == CircuitState.CLOSED

    def test_reset_clears_stats(self):
        cb = make_breaker(failure_threshold=2, recovery_timeout_seconds=60.0)
        trip_breaker(cb, 2)
        cb.reset()
        stats = cb.stats
        assert stats.total_calls == 0
        assert stats.failed_calls == 0

    def test_after_reset_breaker_trips_again(self):
        cb = make_breaker(failure_threshold=2, recovery_timeout_seconds=60.0)
        trip_breaker(cb, 2)
        cb.reset()
        trip_breaker(cb, 2)
        assert cb.state == CircuitState.OPEN


# ---------------------------------------------------------------------------
# Extra: timeout / rate-limit filtering
# ---------------------------------------------------------------------------

class TestFailureFiltering:
    def test_timeout_not_counted_when_disabled(self):
        config = CircuitConfig(
            failure_threshold=2,
            count_timeouts_as_failures=False,
        )
        cb = CircuitBreaker(name="test", config=config)
        cb.record_failure(is_timeout=True)
        cb.record_failure(is_timeout=True)
        assert cb.state == CircuitState.CLOSED

    def test_timeout_counted_when_enabled(self):
        config = CircuitConfig(
            failure_threshold=2,
            count_timeouts_as_failures=True,
        )
        cb = CircuitBreaker(name="test", config=config)
        cb.record_failure(is_timeout=True)
        cb.record_failure(is_timeout=True)
        assert cb.state == CircuitState.OPEN

    def test_rate_limit_not_counted_when_disabled(self):
        config = CircuitConfig(
            failure_threshold=2,
            count_rate_limits_as_failures=False,
        )
        cb = CircuitBreaker(name="test", config=config)
        cb.record_failure(is_rate_limit=True)
        cb.record_failure(is_rate_limit=True)
        assert cb.state == CircuitState.CLOSED

    def test_rate_limit_counted_when_enabled(self):
        config = CircuitConfig(
            failure_threshold=2,
            count_rate_limits_as_failures=True,
        )
        cb = CircuitBreaker(name="test", config=config)
        cb.record_failure(is_rate_limit=True)
        cb.record_failure(is_rate_limit=True)
        assert cb.state == CircuitState.OPEN
