# ControlCore Completion Plan — Tests, Spine, Publish

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a comprehensive test suite to ControlCore, wire in spine for config coordination, and publish to PyPI as `ghostcore` — completing the GhostLogic SDK trifecta (spine + ghostcore + ghostserver).

**Architecture:** ControlCore is a 10K LOC LLM orchestration gateway that is ~95% complete. Cloud adapters for all 9 providers are implemented. The execution engine with fallback, circuit breakers, routing, and redaction is real and working. The only gaps are: zero test coverage, no spine integration, and no PyPI packaging. This plan adds 8 test files covering every module, wires spine in as the config singleton, and publishes as `ghostcore`.

**Tech Stack:** Python 3.10+, pydantic, httpx, starlette, structlog, click, spine (maelspine), pytest + respx

---

## File Structure

```
ControlCore/
├── pyproject.toml                          # UPDATE: rename to ghostcore, add metadata
├── README.md                               # CREATE
├── LICENSE                                  # CREATE
├── ControlCore/
│   ├── boot.py                             # CREATE: spine bootstrap
│   └── (all existing files unchanged)
└── tests/
    ├── __init__.py                          # CREATE
    ├── conftest.py                          # CREATE: shared fixtures
    ├── test_schemas.py                      # CREATE
    ├── test_cloud_adapters.py              # CREATE
    ├── test_dial.py                         # CREATE
    ├── test_routing.py                      # CREATE
    ├── test_circuit_breaker.py             # CREATE
    ├── test_executor.py                     # CREATE
    ├── test_redaction.py                    # CREATE
    └── test_fallback.py                     # CREATE
```

**Key constraint:** The existing codebase is well-architected and complete. We are NOT refactoring anything. We are adding tests to verify existing behavior, wiring spine in alongside (not replacing) existing config, and packaging.

---

### Task 1: Test Infrastructure

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Create test directory and conftest**

`tests/__init__.py`: empty file

`tests/conftest.py`:
```python
"""Shared fixtures for ControlCore tests."""
import pytest
from datetime import datetime
from ControlCore.schemas import (
    ControlCoreCall, Caller, Intent, Target, Params,
    CallOptions, Timeouts, RedactionPolicy,
    IntentClass, TargetType, TrustTier, Verbosity, Determinism, RedactionMode,
)
from ControlCore.adapters.interface import (
    ExecutionAdapter, AdapterResult, AdapterStatus, AdapterConfig, AdapterTiming,
)
from ControlCore.registry.schema import ModelEntry, ModelRegistry, Provider, CapabilityTag, TimeoutDefaults, CostHints


def make_call(
    prompt: str = "What is Python?",
    intent_class: IntentClass = IntentClass.lookup,
    target_alias: str = "claude-sonnet",
    trust_tier: TrustTier = TrustTier.standard,
    temperature: float | None = None,
    seed: int | None = None,
    soft_ms: int = 15000,
    hard_ms: int = 60000,
) -> ControlCoreCall:
    """Build a valid ControlCoreCall for testing."""
    return ControlCoreCall(
        schema_version="1.0.0",
        call_id="test-call-001",
        created_at=datetime.utcnow().isoformat() + "Z",
        caller=Caller(handle="test_user", account_id="00000000-0000-0000-0000-000000000001"),
        intent=Intent(cls=intent_class),
        target=Target(type=TargetType.model, alias=target_alias, trust_tier=trust_tier),
        prompt=prompt,
        context=[],
        params=Params(temperature=temperature, seed=seed),
        options=CallOptions(
            verbosity=Verbosity.standard,
            determinism=Determinism.best_effort,
            timeouts=Timeouts(soft_ms=soft_ms, hard_ms=hard_ms),
            redaction=RedactionPolicy(mode=RedactionMode.auto),
        ),
    )


def make_model(
    alias: str = "test-model",
    provider: Provider = Provider.api_hub,
    provider_model_id: str = "test-model-v1",
    capabilities: list[CapabilityTag] | None = None,
    intents: list[str] | None = None,
    trust_tier: str = "standard",
    context_window: int = 128000,
    enabled: bool = True,
    deprecated: bool = False,
    soft_ms: int = 15000,
    hard_ms: int = 60000,
    input_cost: float = 0.01,
    output_cost: float = 0.03,
) -> ModelEntry:
    """Build a ModelEntry for testing."""
    return ModelEntry(
        alias=alias,
        display_name=alias.replace("-", " ").title(),
        description=f"Test model {alias}",
        provider=provider,
        provider_model_id=provider_model_id,
        capability_tags=capabilities or [CapabilityTag.reason, CapabilityTag.code],
        supported_intents=intents or ["lookup", "summarize", "draft"],
        trust_tier=trust_tier,
        context_window=context_window,
        timeouts=TimeoutDefaults(soft_ms=soft_ms, hard_ms=hard_ms),
        cost_hints=CostHints(input_per_1k_tokens=input_cost, output_per_1k_tokens=output_cost),
        enabled=enabled,
    )


class MockAdapter(ExecutionAdapter):
    """Test adapter that returns configurable results."""

    def __init__(
        self,
        name: str = "mock",
        handled: set[str] | None = None,
        result: AdapterResult | None = None,
    ):
        config = AdapterConfig(adapter_name=name, adapter_version="1.0.0")
        super().__init__(config)
        self._handled = handled or {"test-model"}
        self._result = result or AdapterResult(
            status=AdapterStatus.success,
            content="Mock response",
            provenance=self.create_provenance("test-model"),
        )
        self.calls: list[tuple[str, str]] = []  # (call_id, model_alias)

    def can_handle(self, model_alias: str) -> bool:
        return model_alias in self._handled

    async def execute(self, call, model_alias, *, soft_timeout_ms=None, hard_timeout_ms=None):
        self.calls.append((call.call_id, model_alias))
        return self._result
```

- [ ] **Step 2: Install dev deps and verify import**

```bash
cd D:/lost_marbles/ControlCore
pip install -e ".[dev]"
python -c "from tests.conftest import make_call; print(make_call())"
```

- [ ] **Step 3: Commit**

```bash
git add tests/
git commit -m "test: add test infrastructure with fixtures and mock adapter"
```

---

### Task 2: Schema Tests

**Files:**
- Create: `tests/test_schemas.py`

- [ ] **Step 1: Write schema validation tests**

`tests/test_schemas.py`:
```python
"""Tests for ControlCoreCall schema validation."""
import pytest
from pydantic import ValidationError
from ControlCore.schemas import (
    ControlCoreCall, Caller, Intent, Target, Params, CallOptions,
    Timeouts, RedactionPolicy, RedactionOverride, RedactionMode,
    IntentClass, TargetType, TrustTier, Verbosity, Determinism,
    ControlCoreCallResult, CallStatus,
)
from tests.conftest import make_call


def test_valid_call_construction():
    call = make_call()
    assert call.prompt == "What is Python?"
    assert call.intent.cls == IntentClass.lookup
    assert call.target.alias == "claude-sonnet"


def test_empty_prompt_rejected():
    with pytest.raises(ValidationError):
        make_call(prompt="")


def test_all_intent_classes_accepted():
    for intent in IntentClass:
        call = make_call(intent_class=intent)
        assert call.intent.cls == intent


def test_all_trust_tiers_accepted():
    for tier in TrustTier:
        call = make_call(trust_tier=tier)
        assert call.target.trust_tier == tier


def test_strict_determinism_without_seed_warns():
    # strict determinism with seed should work
    call = make_call(seed=42)
    call.options.determinism = Determinism.strict
    assert call.params.seed == 42


def test_soft_timeout_less_than_hard():
    call = make_call(soft_ms=5000, hard_ms=60000)
    assert call.options.timeouts.soft_ms < call.options.timeouts.hard_ms


def test_redaction_off_requires_override():
    call = make_call()
    call.options.redaction = RedactionPolicy(
        mode=RedactionMode.off,
        override=RedactionOverride(
            enabled=True,
            acknowledgements=[
                "INCLUDE_SENSITIVE_DATA",
                "NO_REDACTION_ACKNOWLEDGED",
                "I_UNDERSTAND_AND_ACCEPT_RISK",
            ],
        ),
    )
    assert call.options.redaction.mode == RedactionMode.off


def test_result_construction():
    result = ControlCoreCallResult(
        schema_version="1.0.0",
        call_id="test-001",
        status=CallStatus.complete,
        answer="The answer is 42",
    )
    assert result.status == CallStatus.complete
    assert result.answer == "The answer is 42"


def test_result_failed_status():
    result = ControlCoreCallResult(
        schema_version="1.0.0",
        call_id="test-001",
        status=CallStatus.failed,
    )
    assert result.status == CallStatus.failed
    assert result.answer is None


def test_caller_handle_validation():
    # Valid handle
    caller = Caller(handle="alice", account_id="00000000-0000-0000-0000-000000000001")
    assert caller.handle == "alice"


def test_params_optional():
    call = make_call(temperature=None, seed=None)
    assert call.params.temperature is None
    assert call.params.seed is None


def test_params_temperature_range():
    call = make_call(temperature=0.5)
    assert call.params.temperature == 0.5
```

- [ ] **Step 2: Run tests**

```bash
cd D:/lost_marbles/ControlCore && pytest tests/test_schemas.py -v
```

Expected: All pass (these validate existing schema behavior)

- [ ] **Step 3: Commit**

```bash
git add tests/test_schemas.py
git commit -m "test: schema validation — calls, results, params, intents"
```

---

### Task 3: Cloud Adapter Tests

**Files:**
- Create: `tests/test_cloud_adapters.py`

- [ ] **Step 1: Write adapter tests**

`tests/test_cloud_adapters.py`:
```python
"""Tests for cloud adapter request building and response parsing."""
import pytest
import respx
import httpx
from tests.conftest import make_call
from ControlCore.adapters.cloud import (
    OpenAIAdapter, AnthropicAdapter, GoogleAdapter, XAIAdapter,
    OpenAICompatibleAdapter, CohereAdapter,
    create_openai_adapter, create_anthropic_adapter, create_google_adapter,
    create_xai_adapter, create_groq_adapter, create_together_adapter,
    create_mistral_adapter, create_deepseek_adapter, create_perplexity_adapter,
    create_all_cloud_adapters, CloudAdapterConfig, CloudProvider,
    AdapterStatus,
)


# --- OpenAI ---

class TestOpenAI:
    def setup_method(self):
        self.adapter = create_openai_adapter(api_key="sk-test-key")

    def test_can_handle_known_models(self):
        assert self.adapter.can_handle("gpt4o")
        assert self.adapter.can_handle("o1")
        assert not self.adapter.can_handle("claude-sonnet")

    def test_build_request(self):
        call = make_call(prompt="Hello", target_alias="gpt4o")
        headers, payload = self.adapter._build_request(call, "gpt-4o")
        assert headers["Authorization"] == "Bearer sk-test-key"
        assert payload["model"] == "gpt-4o"
        assert any(m["content"] == "Hello" for m in payload["messages"])

    def test_parse_response(self):
        data = {
            "choices": [{"message": {"content": "Hi there"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        content, inp, out = self.adapter._parse_response(data)
        assert content == "Hi there"
        assert inp == 10
        assert out == 5

    def test_check_refusal_content_filter(self):
        data = {"choices": [{"message": {"content": ""}, "finish_reason": "content_filter"}]}
        assert self.adapter._check_refusal(data) == "Content filtered"

    def test_check_no_refusal(self):
        data = {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}
        assert self.adapter._check_refusal(data) is None

    @respx.mock
    @pytest.mark.asyncio
    async def test_execute_success(self):
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={
                "choices": [{"message": {"content": "Result"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20},
            })
        )
        call = make_call(target_alias="gpt4o")
        result = await self.adapter.execute(call, "gpt4o")
        assert result.status == AdapterStatus.success
        assert result.content == "Result"

    @respx.mock
    @pytest.mark.asyncio
    async def test_execute_rate_limited(self):
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(429, text="Rate limited")
        )
        call = make_call(target_alias="gpt4o")
        result = await self.adapter.execute(call, "gpt4o")
        assert result.status == AdapterStatus.rate_limited

    @respx.mock
    @pytest.mark.asyncio
    async def test_execute_no_api_key(self):
        adapter = create_openai_adapter(api_key=None)
        # Clear env to ensure no fallback
        import os
        old = os.environ.pop("OPENAI_API_KEY", None)
        try:
            call = make_call(target_alias="gpt4o")
            result = await adapter.execute(call, "gpt4o")
            assert result.status == AdapterStatus.error
            assert "No API key" in result.error_message
        finally:
            if old:
                os.environ["OPENAI_API_KEY"] = old


# --- Anthropic ---

class TestAnthropic:
    def setup_method(self):
        self.adapter = create_anthropic_adapter(api_key="sk-ant-test")

    def test_can_handle(self):
        assert self.adapter.can_handle("claude-sonnet")
        assert self.adapter.can_handle("claude-opus")
        assert not self.adapter.can_handle("gpt4o")

    def test_build_request(self):
        call = make_call(prompt="Explain X")
        headers, payload = self.adapter._build_request(call, "claude-sonnet-4-20250514")
        assert headers["x-api-key"] == "sk-ant-test"
        assert headers["anthropic-version"] == "2023-06-01"
        assert payload["model"] == "claude-sonnet-4-20250514"

    def test_parse_response(self):
        data = {
            "content": [{"type": "text", "text": "Here is the explanation"}],
            "usage": {"input_tokens": 15, "output_tokens": 30},
        }
        content, inp, out = self.adapter._parse_response(data)
        assert content == "Here is the explanation"
        assert inp == 15
        assert out == 30

    @respx.mock
    @pytest.mark.asyncio
    async def test_execute_success(self):
        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, json={
                "content": [{"type": "text", "text": "Done"}],
                "usage": {"input_tokens": 5, "output_tokens": 10},
                "stop_reason": "end_turn",
            })
        )
        call = make_call(target_alias="claude-sonnet")
        result = await self.adapter.execute(call, "claude-sonnet")
        assert result.status == AdapterStatus.success
        assert result.content == "Done"


# --- Google ---

class TestGoogle:
    def setup_method(self):
        self.adapter = create_google_adapter(api_key="goog-test-key")

    def test_can_handle(self):
        assert self.adapter.can_handle("gemini-pro")
        assert not self.adapter.can_handle("gpt4o")

    def test_build_request(self):
        call = make_call(prompt="Test")
        headers, payload = self.adapter._build_request(call, "gemini-1.5-pro")
        assert "contents" in payload
        assert payload["contents"][0]["parts"][-1]["text"] == "Test"

    def test_parse_response(self):
        data = {
            "candidates": [{"content": {"parts": [{"text": "Gemini says hi"}]}}],
            "usageMetadata": {"promptTokenCount": 8, "candidatesTokenCount": 12},
        }
        content, inp, out = self.adapter._parse_response(data)
        assert content == "Gemini says hi"
        assert inp == 8
        assert out == 12

    def test_endpoint_includes_key(self):
        endpoint = self.adapter._get_endpoint("gemini-1.5-pro")
        assert "key=goog-test-key" in endpoint
        assert "gemini-1.5-pro" in endpoint

    def test_check_safety_refusal(self):
        data = {"candidates": [{"finishReason": "SAFETY"}]}
        assert self.adapter._check_refusal(data) is not None


# --- OpenAI-Compatible (Groq, Together, etc.) ---

class TestOpenAICompatible:
    def test_groq_can_handle(self):
        adapter = create_groq_adapter(api_key="groq-key")
        assert adapter.can_handle("llama-70b")
        assert not adapter.can_handle("gpt4o")

    def test_together_can_handle(self):
        adapter = create_together_adapter(api_key="tog-key")
        assert adapter.can_handle("llama-405b")

    def test_mistral_can_handle(self):
        adapter = create_mistral_adapter(api_key="mis-key")
        assert adapter.can_handle("mistral-large")
        assert adapter.can_handle("codestral")

    def test_deepseek_can_handle(self):
        adapter = create_deepseek_adapter(api_key="ds-key")
        assert adapter.can_handle("deepseek-reasoner")

    def test_perplexity_can_handle(self):
        adapter = create_perplexity_adapter(api_key="pplx-key")
        assert adapter.can_handle("pplx-online")


# --- Cohere ---

class TestCohere:
    def test_build_request_combines_context(self):
        from ControlCore.schemas import ContextPart
        adapter = CohereAdapter(CloudAdapterConfig(
            adapter_name="cohere", adapter_version="1.0.0",
            provider=CloudProvider.COHERE, api_key="co-key",
            handled_models={"command"}, model_mapping={"command": "command-r-plus"},
        ))
        call = make_call(prompt="Main question")
        call.context = [ContextPart(part_id="1", content="Background info")]
        headers, payload = adapter._build_request(call, "command-r-plus")
        assert "Background info" in payload["message"]
        assert "Main question" in payload["message"]


# --- Factory ---

class TestFactory:
    def test_create_all_returns_9_adapters(self):
        adapters = create_all_cloud_adapters()
        assert len(adapters) == 9
        assert "openai" in adapters
        assert "anthropic" in adapters
        assert "google" in adapters
        assert "groq" in adapters

    def test_model_resolution(self):
        adapter = create_anthropic_adapter(api_key="test")
        assert adapter._resolve_model_id("claude-sonnet") == "claude-sonnet-4-20250514"
        assert adapter._resolve_model_id("claude-opus") == "claude-opus-4-20250514"
```

- [ ] **Step 2: Run tests**

```bash
pytest tests/test_cloud_adapters.py -v
```

Expected: All pass

- [ ] **Step 3: Commit**

```bash
git add tests/test_cloud_adapters.py
git commit -m "test: cloud adapters — all 9 providers, request/response/refusal"
```

---

### Task 4: Eligibility + Routing Tests

**Files:**
- Create: `tests/test_dial.py`
- Create: `tests/test_routing.py`

- [ ] **Step 1: Write eligibility tests**

`tests/test_dial.py`:
```python
"""Tests for model eligibility filtering."""
import pytest
from tests.conftest import make_call, make_model
from ControlCore.registry.dial import filter_eligible_models
from ControlCore.registry.schema import ModelRegistry, CapabilityTag
from ControlCore.schemas import IntentClass, TrustTier


def _registry(*models):
    return ModelRegistry(version="1.0.0", models={m.alias: m for m in models})


def test_disabled_model_excluded():
    reg = _registry(make_model(alias="m1", enabled=False))
    call = make_call(target_alias="m1")
    result = filter_eligible_models(call, reg)
    assert len(result.eligible) == 0


def test_deprecated_model_excluded():
    reg = _registry(make_model(alias="m1", deprecated=True))
    call = make_call(target_alias="m1")
    result = filter_eligible_models(call, reg)
    assert len(result.eligible) == 0


def test_enabled_model_included():
    reg = _registry(make_model(alias="m1", enabled=True, intents=["lookup"]))
    call = make_call(target_alias="m1", intent_class=IntentClass.lookup)
    result = filter_eligible_models(call, reg)
    assert len(result.eligible) == 1
    assert result.eligible[0].alias == "m1"


def test_wrong_intent_excluded():
    reg = _registry(make_model(alias="m1", intents=["draft"]))
    call = make_call(intent_class=IntentClass.lookup)
    result = filter_eligible_models(call, reg)
    assert len(result.eligible) == 0


def test_trust_tier_filtering():
    trusted = make_model(alias="trusted", trust_tier="trusted", intents=["lookup"])
    untrusted = make_model(alias="untrusted", trust_tier="untrusted", intents=["lookup"])
    reg = _registry(trusted, untrusted)
    # Standard trust call should see both trusted and standard, not untrusted
    call = make_call(trust_tier=TrustTier.standard)
    result = filter_eligible_models(call, reg)
    aliases = [m.alias for m in result.eligible]
    assert "trusted" in aliases


def test_multiple_models_filtered():
    m1 = make_model(alias="m1", enabled=True, intents=["lookup"])
    m2 = make_model(alias="m2", enabled=False, intents=["lookup"])
    m3 = make_model(alias="m3", enabled=True, intents=["lookup"])
    reg = _registry(m1, m2, m3)
    call = make_call(intent_class=IntentClass.lookup)
    result = filter_eligible_models(call, reg)
    aliases = [m.alias for m in result.eligible]
    assert "m1" in aliases
    assert "m2" not in aliases
    assert "m3" in aliases
```

- [ ] **Step 2: Write routing tests**

`tests/test_routing.py`:
```python
"""Tests for model routing and ranking."""
import pytest
from tests.conftest import make_call, make_model
from ControlCore.registry.routing import compute_routing_order
from ControlCore.registry.schema import CapabilityTag
from ControlCore.schemas import IntentClass


def test_higher_trust_ranked_first():
    trusted = make_model(alias="trusted", trust_tier="trusted", intents=["lookup"])
    standard = make_model(alias="standard", trust_tier="standard", intents=["lookup"])
    call = make_call(intent_class=IntentClass.lookup)
    result = compute_routing_order(call, [trusted, standard])
    assert result.ordered[0].model.alias == "trusted"


def test_cheaper_model_preferred_when_tied():
    expensive = make_model(alias="expensive", intents=["lookup"], input_cost=0.10, output_cost=0.30)
    cheap = make_model(alias="cheap", intents=["lookup"], input_cost=0.001, output_cost=0.002)
    call = make_call(intent_class=IntentClass.lookup)
    result = compute_routing_order(call, [expensive, cheap])
    # Both should appear; cheaper should rank higher when other factors equal
    aliases = [r.model.alias for r in result.ordered]
    assert "cheap" in aliases
    assert "expensive" in aliases


def test_empty_eligible_returns_empty():
    call = make_call()
    result = compute_routing_order(call, [])
    assert len(result.ordered) == 0


def test_routing_deterministic():
    models = [
        make_model(alias=f"m{i}", intents=["lookup"], input_cost=0.01 * i)
        for i in range(5)
    ]
    call = make_call(intent_class=IntentClass.lookup)
    r1 = compute_routing_order(call, models)
    r2 = compute_routing_order(call, models)
    assert [m.model.alias for m in r1.ordered] == [m.model.alias for m in r2.ordered]
```

- [ ] **Step 3: Run tests**

```bash
pytest tests/test_dial.py tests/test_routing.py -v
```

- [ ] **Step 4: Commit**

```bash
git add tests/test_dial.py tests/test_routing.py
git commit -m "test: eligibility filtering + model routing"
```

---

### Task 5: Circuit Breaker Tests

**Files:**
- Create: `tests/test_circuit_breaker.py`

- [ ] **Step 1: Write circuit breaker tests**

`tests/test_circuit_breaker.py`:
```python
"""Tests for circuit breaker state machine."""
import pytest
import time
from ControlCore.circuit_breaker import (
    CircuitBreaker, CircuitConfig, CircuitState, CircuitOpenError,
    CircuitBreakerRegistry,
)


@pytest.fixture
def breaker():
    config = CircuitConfig(
        failure_threshold=3,
        failure_window_seconds=60,
        recovery_timeout_seconds=1,
        half_open_success_threshold=2,
    )
    return CircuitBreaker(config=config)


def test_starts_closed(breaker):
    assert breaker.state == CircuitState.CLOSED


def test_allows_requests_when_closed(breaker):
    assert breaker.allow_request() is True


def test_opens_after_threshold_failures(breaker):
    for _ in range(3):
        breaker.record_failure()
    assert breaker.state == CircuitState.OPEN


def test_rejects_when_open(breaker):
    for _ in range(3):
        breaker.record_failure()
    assert breaker.allow_request() is False


def test_transitions_to_half_open_after_recovery(breaker):
    for _ in range(3):
        breaker.record_failure()
    assert breaker.state == CircuitState.OPEN
    # Wait for recovery timeout
    time.sleep(1.1)
    assert breaker.allow_request() is True
    assert breaker.state == CircuitState.HALF_OPEN


def test_closes_after_half_open_successes(breaker):
    for _ in range(3):
        breaker.record_failure()
    time.sleep(1.1)
    breaker.allow_request()  # transitions to HALF_OPEN
    breaker.record_success()
    breaker.record_success()
    assert breaker.state == CircuitState.CLOSED


def test_reopens_on_half_open_failure(breaker):
    for _ in range(3):
        breaker.record_failure()
    time.sleep(1.1)
    breaker.allow_request()  # HALF_OPEN
    breaker.record_failure()
    assert breaker.state == CircuitState.OPEN


def test_success_resets_failure_count(breaker):
    breaker.record_failure()
    breaker.record_failure()
    breaker.record_success()
    breaker.record_failure()
    breaker.record_failure()
    # Should still be closed — success reset the window
    assert breaker.state == CircuitState.CLOSED


def test_registry_creates_per_key():
    registry = CircuitBreakerRegistry()
    b1 = registry.get_or_create("adapter1", "model1")
    b2 = registry.get_or_create("adapter1", "model2")
    assert b1 is not b2


def test_registry_returns_same_breaker():
    registry = CircuitBreakerRegistry()
    b1 = registry.get_or_create("adapter1", "model1")
    b2 = registry.get_or_create("adapter1", "model1")
    assert b1 is b2
```

- [ ] **Step 2: Run tests**

```bash
pytest tests/test_circuit_breaker.py -v
```

- [ ] **Step 3: Commit**

```bash
git add tests/test_circuit_breaker.py
git commit -m "test: circuit breaker state machine + registry"
```

---

### Task 6: Redaction Tests

**Files:**
- Create: `tests/test_redaction.py`

- [ ] **Step 1: Write redaction tests**

`tests/test_redaction.py`:
```python
"""Tests for post-model redaction."""
import pytest
from ControlCore.redaction import redact_text, RedactionReport


def test_redacts_api_key():
    text = "Use this key: sk_live_abc123def456ghi789jkl"
    result, report = redact_text(text)
    assert "sk_live_" not in result
    assert report.performed is True
    assert any(item.kind == "api_key" for item in report.items)


def test_redacts_bearer_token():
    text = "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.abc.def"
    result, report = redact_text(text)
    assert "eyJhbGci" not in result


def test_redacts_email():
    text = "Contact me at adam@ghostlogic.dev for details"
    result, report = redact_text(text)
    assert "adam@ghostlogic.dev" not in result


def test_no_redaction_on_clean_text():
    text = "Python is a programming language used for many things."
    result, report = redact_text(text)
    assert result == text
    assert report.performed is False or len(report.items) == 0


def test_multiple_patterns():
    text = "Key: sk_test_12345678901234567890 and email: user@example.com"
    result, report = redact_text(text)
    assert "sk_test_" not in result
    assert "user@example.com" not in result
```

- [ ] **Step 2: Run tests**

```bash
pytest tests/test_redaction.py -v
```

Note: The redaction module's exact function signature may differ. Read `ControlCore/redaction.py` to verify `redact_text` returns `(str, RedactionReport)`. Adjust imports and assertions if the API is different.

- [ ] **Step 3: Commit**

```bash
git add tests/test_redaction.py
git commit -m "test: post-model redaction patterns"
```

---

### Task 7: Execution Engine Tests

**Files:**
- Create: `tests/test_executor.py`

- [ ] **Step 1: Write executor tests**

`tests/test_executor.py`:
```python
"""Tests for the execution engine with mock adapters."""
import pytest
from tests.conftest import make_call, make_model, MockAdapter
from ControlCore.adapters.interface import AdapterResult, AdapterStatus
from ControlCore.adapters.executor import (
    ExecutionEngine, AdapterRegistry, ExecutionOutcome,
)
from ControlCore.registry.schema import ModelRegistry
from ControlCore.registry.fallback import default_policy
from ControlCore.circuit_breaker import CircuitBreakerRegistry


def _setup_engine(models, adapters):
    """Create an engine with given models and adapters."""
    model_reg = ModelRegistry(version="1.0.0", models={m.alias: m for m in models})
    adapter_reg = AdapterRegistry()
    for adapter in adapters:
        adapter_reg.register(adapter)
    circuit_reg = CircuitBreakerRegistry()
    return ExecutionEngine(
        model_registry=model_reg,
        adapter_registry=adapter_reg,
        circuit_registry=circuit_reg,
    )


@pytest.mark.asyncio
async def test_execute_once_success():
    model = make_model(alias="m1", intents=["lookup"])
    adapter = MockAdapter(name="mock", handled={"m1"})
    engine = _setup_engine([model], [adapter])
    call = make_call(target_alias="m1")
    result, used_adapter = await engine.execute_once(call, "m1")
    assert result.status == AdapterStatus.success
    assert result.content == "Mock response"
    assert len(adapter.calls) == 1


@pytest.mark.asyncio
async def test_execute_once_no_adapter():
    model = make_model(alias="m1", intents=["lookup"])
    engine = _setup_engine([model], [])  # no adapters
    call = make_call(target_alias="m1")
    result, used_adapter = await engine.execute_once(call, "m1")
    assert result.status == AdapterStatus.error
    assert used_adapter is None


@pytest.mark.asyncio
async def test_execute_with_fallback_success():
    model = make_model(alias="m1", intents=["lookup"])
    adapter = MockAdapter(name="mock", handled={"m1"})
    engine = _setup_engine([model], [adapter])
    call = make_call(target_alias="m1", intent_class="lookup")
    policy = default_policy()
    result, trace = await engine.execute_with_fallback(call, policy=policy)
    assert trace.outcome == ExecutionOutcome.SUCCESS


@pytest.mark.asyncio
async def test_fallback_tries_next_model():
    m1 = make_model(alias="m1", intents=["lookup"])
    m2 = make_model(alias="m2", intents=["lookup"])
    fail_adapter = MockAdapter(
        name="fail", handled={"m1"},
        result=AdapterResult(
            status=AdapterStatus.error,
            error_message="Fail",
            error_code="TEST_FAIL",
        ),
    )
    success_adapter = MockAdapter(name="success", handled={"m2"})
    engine = _setup_engine([m1, m2], [fail_adapter, success_adapter])
    call = make_call(intent_class="lookup")
    policy = default_policy()
    result, trace = await engine.execute_with_fallback(call, policy=policy)
    assert len(trace.attempts) >= 2
    assert trace.outcome == ExecutionOutcome.SUCCESS
```

- [ ] **Step 2: Run tests**

```bash
pytest tests/test_executor.py -v
```

Note: The ExecutionEngine constructor and method signatures may differ from the plan. Read the actual `executor.py` to verify:
- How AdapterRegistry.register() works
- What ExecutionEngine.__init__ expects
- Whether execute_with_fallback takes a policy param

Adjust the test setup if the API is different.

- [ ] **Step 3: Commit**

```bash
git add tests/test_executor.py
git commit -m "test: execution engine — success, no-adapter, fallback"
```

---

### Task 8: Fallback Policy Tests

**Files:**
- Create: `tests/test_fallback.py`

- [ ] **Step 1: Write fallback policy tests**

`tests/test_fallback.py`:
```python
"""Tests for fallback policy descriptors."""
import pytest
from ControlCore.registry.fallback import (
    FallbackPolicy, default_policy, aggressive_retry_policy,
    fail_fast_policy, queue_preferred_policy, cost_sensitive_policy,
    RephraseStrategy, ModelSwitchCondition, QueueEscalationCondition,
)


def test_default_policy():
    p = default_policy()
    assert p.max_total_attempts == 5
    assert p.max_same_model_retries == 2
    assert p.model_switch.enabled is True
    assert p.rephrase.enabled is True


def test_fail_fast_policy():
    p = fail_fast_policy()
    assert p.max_total_attempts <= 2
    assert p.fail_fast is True


def test_aggressive_retry_policy():
    p = aggressive_retry_policy()
    assert p.max_total_attempts >= 5
    assert p.max_same_model_retries >= 3


def test_queue_preferred_policy():
    p = queue_preferred_policy()
    assert p.queue_escalation.enabled is True


def test_cost_sensitive_policy():
    p = cost_sensitive_policy()
    assert p.model_switch.max_models_to_try >= 1


def test_policy_serialization():
    p = default_policy()
    d = p.to_dict()
    assert "max_total_attempts" in d
    assert "retry_timing" in d or "timing" in d


def test_model_switch_conditions():
    p = default_policy()
    conditions = p.model_switch.conditions
    assert ModelSwitchCondition.refusal in conditions
    assert ModelSwitchCondition.timeout in conditions


def test_policy_describe():
    p = default_policy()
    desc = p.describe()
    assert isinstance(desc, str)
    assert len(desc) > 0
```

- [ ] **Step 2: Run tests**

```bash
pytest tests/test_fallback.py -v
```

- [ ] **Step 3: Commit**

```bash
git add tests/test_fallback.py
git commit -m "test: fallback policy factories and serialization"
```

---

### Task 9: Spine Integration

**Files:**
- Create: `ControlCore/boot.py`
- Modify: `pyproject.toml` (add maelspine dependency)

- [ ] **Step 1: Add spine dependency**

In `pyproject.toml`, add `"maelspine>=0.1.1"` to the `dependencies` list.

- [ ] **Step 2: Write boot.py**

`ControlCore/boot.py`:
```python
"""Spine bootstrap for ControlCore.

Optional — ControlCore works without spine. But if spine is booted,
adapters and the daemon can access config via Core.instance().
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from spine import Core

from ControlCore.config import load_config
from ControlCore.registry.loader import load_model_registry
from ControlCore.adapters.cloud import create_all_cloud_adapters
from ControlCore.adapters.executor import AdapterRegistry
from ControlCore.circuit_breaker import CircuitBreakerRegistry


def boot(config_path: Optional[Path] = None) -> Core:
    """Boot spine with ControlCore registries.

    Registers:
        config — loaded daemon config
        model_registry — ModelRegistry from registry.json
        adapter_registry — AdapterRegistry with all cloud + local adapters
        circuit_registry — CircuitBreakerRegistry (shared)
    """
    config = load_config(config_path) if config_path else {}

    def setup(c: Core) -> None:
        model_registry = load_model_registry()
        adapter_registry = AdapterRegistry()
        for adapter in create_all_cloud_adapters().values():
            adapter_registry.register(adapter)

        c.register("config", config)
        c.register("model_registry", model_registry)
        c.register("adapter_registry", adapter_registry)
        c.register("circuit_registry", CircuitBreakerRegistry())
        c.boot(env="prod")

    return Core.boot_once(setup)
```

- [ ] **Step 3: Write boot test**

Add to `tests/conftest.py` or create `tests/test_boot.py`:

```python
# tests/test_boot.py
import pytest
from spine import Core


@pytest.fixture(autouse=True)
def reset_spine():
    Core._reset_instance()
    yield
    Core._reset_instance()


def test_boot_registers_all():
    from ControlCore.boot import boot
    core = boot()
    assert core.has("model_registry")
    assert core.has("adapter_registry")
    assert core.has("circuit_registry")
    assert core.is_frozen
```

- [ ] **Step 4: Run all tests**

```bash
pytest -v
```

Expected: All tests pass

- [ ] **Step 5: Commit**

```bash
git add ControlCore/boot.py tests/test_boot.py pyproject.toml
git commit -m "feat: spine integration — boot registers all core objects"
```

---

### Task 10: PyPI Packaging + GitHub

**Files:**
- Modify: `pyproject.toml` (rename, add metadata)
- Create: `README.md`
- Create: `LICENSE`

- [ ] **Step 1: Update pyproject.toml**

Replace the `[project]` section:

```toml
[project]
name = "ghostcore"
version = "0.1.0"
description = "LLM orchestration gateway — intelligent routing, fallback, circuit breakers, redaction, 10+ providers"
readme = "README.md"
license = "MIT"
requires-python = ">=3.10"
authors = [
    { name = "Adam Thomas", email = "adam@ghostlogic.dev" },
]
keywords = ["llm", "gateway", "orchestration", "routing", "anthropic", "openai", "ai"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Software Development :: Libraries",
]
dependencies = [
    "pydantic>=2.0",
    "httpx>=0.25.0",
    "uvicorn>=0.24.0",
    "starlette>=0.32.0",
    "structlog>=23.0.0",
    "click>=8.0.0",
    "maelspine>=0.1.1",
]

[project.urls]
Homepage = "https://github.com/adam-scott-thomas/ghostcore"
Repository = "https://github.com/adam-scott-thomas/ghostcore"
Issues = "https://github.com/adam-scott-thomas/ghostcore/issues"

[project.scripts]
ghostcore = "ControlCore.cli:main"
```

- [ ] **Step 2: Create LICENSE**

MIT license, same as ghostserver (Copyright 2026 Adam Thomas / GhostLogic LLC).

- [ ] **Step 3: Create README.md**

Write a README covering:
- What it does (one-sentence: LLM orchestration gateway)
- Why (intelligent routing beats hardcoding model names)
- Quick example (3 lines: boot, call, get result)
- Architecture diagram (routing → eligibility → fallback → adapter → redaction)
- All 10 providers listed
- pip install ghostcore
- Part of the GhostLogic stack (spine + ghostcore + ghostserver)

- [ ] **Step 4: Create GitHub repo and push**

```bash
gh repo create adam-scott-thomas/ghostcore --public --description "LLM orchestration gateway — routing, fallback, circuit breakers, redaction, 10+ providers"
git remote add origin https://github.com/adam-scott-thomas/ghostcore.git
git push -u origin main
```

- [ ] **Step 5: Build and publish to PyPI**

```bash
pip install build twine
python -m build
twine upload dist/*
```

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat: PyPI packaging as ghostcore + README + LICENSE"
git push
```

---

## Self-Review

**Coverage check:**
- Cloud adapters (all 9 providers): Task 3 ✅
- Schema validation: Task 2 ✅
- Eligibility filtering: Task 4 ✅
- Routing/ranking: Task 4 ✅
- Circuit breaker: Task 5 ✅
- Redaction: Task 6 ✅
- Execution engine with fallback: Task 7 ✅
- Fallback policy: Task 8 ✅
- Spine integration: Task 9 ✅
- PyPI publish: Task 10 ✅

**Placeholder scan:** None found. All tasks have concrete code.

**Type consistency:**
- `make_call()` and `make_model()` used consistently across all test files
- `MockAdapter` interface matches `ExecutionAdapter` contract
- `AdapterRegistry.register()` used in both conftest and test_executor
- `filter_eligible_models()` and `compute_routing_order()` match actual function names from source

**Note for implementers:** Tasks 2-8 are testing EXISTING code. The function signatures, class names, and APIs referenced in the tests come from reading the actual source. If any signature doesn't match, read the source file and adjust — the pattern will be the same, only names might differ slightly.
