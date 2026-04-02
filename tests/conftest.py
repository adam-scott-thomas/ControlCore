"""
Shared test fixtures and helpers for ControlCore tests.

Factory functions use exact field names from:
  - ControlCore/schemas.py         (ControlCoreCall, Caller, Intent, Target, Params, CallOptions)
  - ControlCore/adapters/interface.py (ExecutionAdapter, AdapterResult, AdapterStatus, AdapterConfig)
  - ControlCore/registry/schema.py  (ModelEntry, Provider)
"""

from __future__ import annotations

import uuid
from typing import Optional, Set

import pytest

from ControlCore.schemas import (
    Caller,
    CallOptions,
    ControlCoreCall,
    Intent,
    IntentClass,
    Params,
    Target,
    TargetType,
    TrustTier,
)
from ControlCore.adapters.interface import (
    AdapterConfig,
    AdapterResult,
    AdapterStatus,
    ExecutionAdapter,
)
from ControlCore.registry.schema import (
    ModelEntry,
    Provider,
)


# ---------------------------------------------------------------------------
# Factory: ControlCoreCall
# ---------------------------------------------------------------------------

def make_call(
    *,
    handle: str = "test-user",
    account_id: Optional[str] = None,
    intent_class: IntentClass = IntentClass.lookup,
    target_alias: str = "qwen:32b",
    prompt: str = "What is 2 + 2?",
    **kwargs,
) -> ControlCoreCall:
    """
    Factory for valid ControlCoreCall instances.

    Overridable top-level kwargs are forwarded directly to ControlCoreCall.
    Common sub-fields are exposed as named parameters for convenience.
    """
    if account_id is None:
        account_id = str(uuid.uuid4())

    caller = Caller(
        handle=handle,
        account_id=account_id,
    )

    # Intent uses `cls` as the Python attribute name (Pydantic alias for "class")
    intent = Intent.model_validate({"class": intent_class.value})

    target = Target(
        type=TargetType.model,
        alias=target_alias,
        trust_tier=TrustTier.standard,
    )

    return ControlCoreCall(
        caller=caller,
        intent=intent,
        target=target,
        prompt=prompt,
        params=Params(),
        options=CallOptions(),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Factory: ModelEntry
# ---------------------------------------------------------------------------

def make_model(
    *,
    alias: str = "qwen:32b",
    provider: Provider = Provider.local,
    display_name: Optional[str] = "Test Model",
    enabled: bool = True,
    **kwargs,
) -> ModelEntry:
    """
    Factory for valid ModelEntry instances.

    Overridable kwargs are forwarded to ModelEntry.
    """
    return ModelEntry(
        alias=alias,
        provider=provider,
        display_name=display_name,
        enabled=enabled,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# MockAdapter
# ---------------------------------------------------------------------------

class MockAdapter(ExecutionAdapter):
    """
    Configurable mock adapter for testing.

    - handled_models: set of aliases this adapter claims to handle
    - result: AdapterResult returned by execute() (defaults to success with content)
    - Recorded calls are stored in self.calls as list of dicts with call_id + model_alias
    """

    def __init__(
        self,
        *,
        handled_models: Optional[Set[str]] = None,
        result: Optional[AdapterResult] = None,
        adapter_name: str = "mock",
    ) -> None:
        config = AdapterConfig(adapter_name=adapter_name)
        super().__init__(config)

        self._handled_models: Set[str] = handled_models if handled_models is not None else {"qwen:32b"}
        self._result: AdapterResult = result if result is not None else AdapterResult(
            status=AdapterStatus.success,
            content="mock response",
        )
        self.calls: list[dict] = []

    def can_handle(self, model_alias: str) -> bool:
        return model_alias in self._handled_models

    async def execute(
        self,
        call: ControlCoreCall,
        model_alias: str,
        *,
        soft_timeout_ms=None,
        hard_timeout_ms=None,
    ) -> AdapterResult:
        self.calls.append({"call_id": call.call_id, "model_alias": model_alias})
        return self._result


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_call() -> ControlCoreCall:
    """A ready-to-use ControlCoreCall for tests that don't need customisation."""
    return make_call()


@pytest.fixture
def sample_model() -> ModelEntry:
    """A ready-to-use ModelEntry for tests that don't need customisation."""
    return make_model()


@pytest.fixture
def mock_adapter() -> MockAdapter:
    """A ready-to-use MockAdapter pre-configured to handle qwen:32b."""
    return MockAdapter()
