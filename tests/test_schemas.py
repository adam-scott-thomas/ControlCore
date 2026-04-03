"""
Schema validation tests for ghostrouter.

Covers: ControlCoreCall, ControlCoreCallResult, Caller, Intent, Target,
        Params, CallOptions, RedactionPolicy, all enums.
"""

from __future__ import annotations

import uuid
from datetime import datetime

import pytest
from pydantic import ValidationError

from ghostrouter.schemas import (
    CallError,
    CallOptions,
    CallStatus,
    Caller,
    Confidence,
    ControlCoreCall,
    ControlCoreCallResult,
    Determinism,
    ErrorCode,
    Intent,
    IntentClass,
    NormalizationReport,
    Params,
    Provenance,
    RedactionMode,
    RedactionOverride,
    RedactionPolicy,
    RedactionReport,
    Target,
    TargetType,
    TrustTier,
    Verbosity,
)

from tests.conftest import make_call


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _provenance(**overrides) -> Provenance:
    """Minimal valid Provenance for ControlCoreCallResult construction."""
    base = dict(
        model_alias="qwen:32b",
        trust_tier=TrustTier.standard,
        started_at=datetime.utcnow().isoformat() + "Z",
    )
    base.update(overrides)
    return Provenance(**base)


def _result(**overrides) -> ControlCoreCallResult:
    """Minimal valid ControlCoreCallResult."""
    base = dict(
        call_id=str(uuid.uuid4()),
        provenance=_provenance(),
    )
    base.update(overrides)
    return ControlCoreCallResult(**base)


# ===========================================================================
# 1. Valid call construction via make_call()
# ===========================================================================

class TestMakeCall:
    def test_returns_control_core_call(self):
        call = make_call()
        assert isinstance(call, ControlCoreCall)

    def test_defaults_are_populated(self):
        call = make_call()
        assert call.caller.handle == "test-user"
        assert call.intent.cls == IntentClass.lookup
        assert call.target.alias == "qwen:32b"
        assert call.prompt == "What is 2 + 2?"

    def test_schema_version_defaults_to_semver(self):
        call = make_call()
        parts = call.schema_version.split(".")
        assert len(parts) == 3
        assert all(p.isdigit() for p in parts)

    def test_call_id_is_uuid(self):
        call = make_call()
        uuid.UUID(call.call_id)  # raises if not valid UUID

    def test_created_at_is_set(self):
        call = make_call()
        assert call.created_at.endswith("Z")

    def test_custom_prompt(self):
        call = make_call(prompt="Summarize this document.")
        assert call.prompt == "Summarize this document."

    def test_custom_handle(self):
        call = make_call(handle="alice")
        assert call.caller.handle == "alice"

    def test_custom_target_alias(self):
        call = make_call(target_alias="gpt4:turbo")
        assert call.target.alias == "gpt4:turbo"

    def test_two_calls_have_distinct_call_ids(self):
        a = make_call()
        b = make_call()
        assert a.call_id != b.call_id

    def test_extra_fields_forbidden(self):
        """ControlCoreCall inherits extra='forbid' from SchemaVersioned."""
        with pytest.raises(ValidationError):
            make_call(nonexistent_field="oops")


# ===========================================================================
# 2. IntentClass enum — all values accepted
# ===========================================================================

class TestIntentClassEnum:
    @pytest.mark.parametrize("ic", list(IntentClass))
    def test_all_intent_classes_accepted(self, ic: IntentClass):
        call = make_call(intent_class=ic)
        assert call.intent.cls == ic

    def test_intent_constructed_via_model_validate(self):
        intent = Intent.model_validate({"class": "summarize"})
        assert intent.cls == IntentClass.summarize

    def test_intent_cls_kwarg(self):
        # Intent uses a Pydantic alias ("class" → cls); model_validate is the
        # correct construction path when setting via keyword.
        intent = Intent.model_validate({"class": "reason"})
        assert intent.cls == IntentClass.reason

    def test_invalid_intent_class_rejected(self):
        with pytest.raises(ValidationError):
            Intent.model_validate({"class": "not_a_real_intent"})

    def test_intent_optional_detail(self):
        intent = Intent.model_validate({"class": "draft", "detail": "Write a cover letter"})
        assert intent.detail == "Write a cover letter"

    def test_intent_detail_max_length_exceeded(self):
        with pytest.raises(ValidationError):
            Intent(cls=IntentClass.draft, detail="x" * 513)


# ===========================================================================
# 3. TrustTier — all values accepted
# ===========================================================================

class TestTrustTier:
    @pytest.mark.parametrize("tier", list(TrustTier))
    def test_all_trust_tiers_accepted_on_target(self, tier: TrustTier):
        target = Target(type=TargetType.model, alias="some-model", trust_tier=tier)
        assert target.trust_tier == tier

    @pytest.mark.parametrize("tier", list(TrustTier))
    def test_all_trust_tiers_accepted_on_provenance(self, tier: TrustTier):
        prov = _provenance(trust_tier=tier)
        assert prov.trust_tier == tier

    def test_invalid_trust_tier_rejected(self):
        with pytest.raises(ValidationError):
            Target(type=TargetType.model, alias="x", trust_tier="superuser")


# ===========================================================================
# 4. Params with temperature and seed
# ===========================================================================

class TestParams:
    def test_empty_params_all_none(self):
        p = Params()
        assert p.temperature is None
        assert p.top_p is None
        assert p.seed is None

    def test_temperature_zero(self):
        p = Params(temperature=0.0)
        assert p.temperature == 0.0

    def test_temperature_max(self):
        p = Params(temperature=2.0)
        assert p.temperature == 2.0

    def test_temperature_mid(self):
        p = Params(temperature=0.7)
        assert p.temperature == pytest.approx(0.7)

    def test_temperature_below_zero_rejected(self):
        with pytest.raises(ValidationError):
            Params(temperature=-0.1)

    def test_temperature_above_max_rejected(self):
        with pytest.raises(ValidationError):
            Params(temperature=2.01)

    def test_seed_zero(self):
        p = Params(seed=0)
        assert p.seed == 0

    def test_seed_max(self):
        p = Params(seed=2**31 - 1)
        assert p.seed == 2**31 - 1

    def test_seed_negative_rejected(self):
        with pytest.raises(ValidationError):
            Params(seed=-1)

    def test_seed_above_max_rejected(self):
        with pytest.raises(ValidationError):
            Params(seed=2**31)

    def test_top_p_valid(self):
        p = Params(top_p=0.9)
        assert p.top_p == pytest.approx(0.9)

    def test_top_p_above_one_rejected(self):
        with pytest.raises(ValidationError):
            Params(top_p=1.01)

    def test_all_params_together(self):
        p = Params(temperature=0.5, top_p=0.95, seed=42)
        assert p.temperature == pytest.approx(0.5)
        assert p.top_p == pytest.approx(0.95)
        assert p.seed == 42

    def test_params_embedded_in_call(self):
        # make_call() hard-codes params= so build the call directly here.
        base = make_call()
        call = ControlCoreCall(
            caller=base.caller,
            intent=base.intent,
            target=base.target,
            prompt=base.prompt,
            params=Params(temperature=1.0, seed=99),
            options=base.options,
        )
        assert call.params.temperature == pytest.approx(1.0)
        assert call.params.seed == 99


# ===========================================================================
# 5. CallOptions with various determinism modes
# ===========================================================================

class TestCallOptions:
    def test_defaults(self):
        opts = CallOptions()
        assert opts.verbosity == Verbosity.standard
        assert opts.determinism == Determinism.best_effort
        assert opts.allow_variants is True
        assert opts.max_variants == 5

    @pytest.mark.parametrize("mode", list(Determinism))
    def test_all_determinism_modes(self, mode: Determinism):
        opts = CallOptions(determinism=mode)
        assert opts.determinism == mode

    @pytest.mark.parametrize("v", list(Verbosity))
    def test_all_verbosity_levels(self, v: Verbosity):
        opts = CallOptions(verbosity=v)
        assert opts.verbosity == v

    def test_max_variants_min(self):
        opts = CallOptions(max_variants=1)
        assert opts.max_variants == 1

    def test_max_variants_max(self):
        opts = CallOptions(max_variants=5)
        assert opts.max_variants == 5

    def test_max_variants_zero_rejected(self):
        with pytest.raises(ValidationError):
            CallOptions(max_variants=0)

    def test_max_variants_six_rejected(self):
        with pytest.raises(ValidationError):
            CallOptions(max_variants=6)

    def test_options_embedded_in_call(self):
        # make_call() hard-codes options= so build the call directly here.
        opts = CallOptions(determinism=Determinism.strict, verbosity=Verbosity.full)
        base = make_call()
        call = ControlCoreCall(
            caller=base.caller,
            intent=base.intent,
            target=base.target,
            prompt=base.prompt,
            params=base.params,
            options=opts,
        )
        assert call.options.determinism == Determinism.strict
        assert call.options.verbosity == Verbosity.full


# ===========================================================================
# 6. ControlCoreCallResult construction
# ===========================================================================

class TestControlCoreCallResult:
    def test_minimal_complete_result(self):
        r = _result()
        assert r.status == CallStatus.complete
        assert r.partial is False
        assert r.retryable is False
        assert r.errors == []

    def test_complete_result_with_answer(self):
        r = _result(answer="Four.", status=CallStatus.complete)
        assert r.answer == "Four."
        assert r.status == CallStatus.complete

    def test_failed_result(self):
        err = CallError(code=ErrorCode.adapter_error, message="Adapter timed out")
        r = _result(
            status=CallStatus.failed,
            retryable=True,
            errors=[err],
        )
        assert r.status == CallStatus.failed
        assert r.retryable is True
        assert len(r.errors) == 1
        assert r.errors[0].code == ErrorCode.adapter_error

    def test_queued_result_with_job_id(self):
        job = str(uuid.uuid4())
        r = _result(status=CallStatus.queued, job_id=job)
        assert r.status == CallStatus.queued
        assert r.job_id == job

    def test_multiple_answers_variants(self):
        r = _result(answers=["Answer A", "Answer B", "Answer C"])
        assert len(r.answers) == 3

    def test_partial_flag(self):
        r = _result(partial=True)
        assert r.partial is True

    def test_result_missing_provenance_rejected(self):
        with pytest.raises(ValidationError):
            ControlCoreCallResult(call_id=str(uuid.uuid4()))

    def test_result_schema_version_defaults(self):
        r = _result()
        assert r.schema_version == "1.0.0"

    def test_call_error_details(self):
        err = CallError(
            code=ErrorCode.permission_denied,
            message="Not authorized",
            details={"user": "alice", "resource": "capsule:42"},
        )
        assert err.details["resource"] == "capsule:42"

    @pytest.mark.parametrize("status", list(CallStatus))
    def test_all_call_statuses(self, status: CallStatus):
        r = _result(status=status)
        assert r.status == status

    def test_confidence_fields(self):
        c = Confidence(self_reported=0.9, system_estimate=0.85)
        r = _result(confidence=c)
        assert r.confidence.self_reported == pytest.approx(0.9)
        assert r.confidence.system_estimate == pytest.approx(0.85)

    def test_redaction_report_in_result(self):
        rr = RedactionReport(performed=True, override_used=False)
        r = _result(redaction=rr)
        assert r.redaction.performed is True


# ===========================================================================
# 7. Caller construction
# ===========================================================================

class TestCaller:
    def test_valid_caller_with_uuid(self):
        c = Caller(handle="adam", account_id=str(uuid.uuid4()))
        assert c.handle == "adam"

    def test_handle_min_length(self):
        c = Caller(handle="abc", account_id=str(uuid.uuid4()))
        assert c.handle == "abc"

    def test_handle_too_short_rejected(self):
        with pytest.raises(ValidationError):
            Caller(handle="ab", account_id=str(uuid.uuid4()))

    def test_handle_max_length(self):
        handle = "a" * 64
        c = Caller(handle=handle, account_id=str(uuid.uuid4()))
        assert len(c.handle) == 64

    def test_handle_too_long_rejected(self):
        with pytest.raises(ValidationError):
            Caller(handle="a" * 65, account_id=str(uuid.uuid4()))

    def test_handle_allows_spaces_and_dots(self):
        c = Caller(handle="John Doe", account_id=str(uuid.uuid4()))
        assert c.handle == "John Doe"

    def test_handle_allows_hyphens_and_underscores(self):
        c = Caller(handle="test-user_1", account_id=str(uuid.uuid4()))
        assert c.handle == "test-user_1"

    def test_handle_invalid_character_rejected(self):
        with pytest.raises(ValidationError):
            Caller(handle="bad@handle", account_id=str(uuid.uuid4()))

    def test_account_id_valid_uuid(self):
        aid = str(uuid.uuid4())
        c = Caller(handle="user-one", account_id=aid)
        assert c.account_id == aid

    def test_account_id_non_uuid_long_enough(self):
        """Non-UUID account_id is accepted if it is >= 8 chars."""
        c = Caller(handle="user-one", account_id="opaque-id-longerthan8")
        assert c.account_id == "opaque-id-longerthan8"

    def test_account_id_too_short_non_uuid_rejected(self):
        """Non-UUID account_id shorter than 8 chars must be rejected."""
        with pytest.raises(ValidationError):
            Caller(handle="user-one", account_id="short")

    def test_optional_key_id(self):
        c = Caller(handle="user-one", account_id=str(uuid.uuid4()), key_id="fp:abc123")
        assert c.key_id == "fp:abc123"

    def test_optional_fingerprint_ref(self):
        c = Caller(
            handle="user-one",
            account_id=str(uuid.uuid4()),
            fingerprint_ref="ref:xyz",
        )
        assert c.fingerprint_ref == "ref:xyz"

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            Caller(handle="user-one", account_id=str(uuid.uuid4()), secret="leak")


# ===========================================================================
# 8. RedactionPolicy with override
# ===========================================================================

class TestRedactionPolicy:
    def test_default_mode_is_auto(self):
        rp = RedactionPolicy()
        assert rp.mode == RedactionMode.auto

    def test_auto_mode_no_override_needed(self):
        call = make_call()  # default options → auto mode
        assert call.options.redaction.mode == RedactionMode.auto

    def _make_call_with_options(self, opts: CallOptions) -> ControlCoreCall:
        """Build a ControlCoreCall with explicit options (bypassing make_call hardcoding)."""
        base = make_call()
        return ControlCoreCall(
            caller=base.caller,
            intent=base.intent,
            target=base.target,
            prompt=base.prompt,
            params=base.params,
            options=opts,
        )

    def test_off_mode_requires_override_enabled(self):
        """mode=off without override.enabled should raise at call level."""
        opts = CallOptions(
            redaction=RedactionPolicy(mode=RedactionMode.off)
        )
        with pytest.raises(ValidationError, match="redaction.mode off requires"):
            self._make_call_with_options(opts)

    def test_off_mode_with_override_enabled_accepted(self):
        override = RedactionOverride(
            enabled=True,
            acknowledgements=["I accept full responsibility for redaction off"],
            reason="Internal audit tooling",
        )
        opts = CallOptions(
            redaction=RedactionPolicy(mode=RedactionMode.off, override=override)
        )
        call = self._make_call_with_options(opts)
        assert call.options.redaction.mode == RedactionMode.off
        assert call.options.redaction.override.enabled is True

    def test_off_mode_with_override_not_enabled_rejected(self):
        """override present but enabled=False still fails the model_validator."""
        override = RedactionOverride(enabled=False, reason="test")
        opts = CallOptions(
            redaction=RedactionPolicy(mode=RedactionMode.off, override=override)
        )
        with pytest.raises(ValidationError, match="redaction.mode off requires"):
            self._make_call_with_options(opts)

    def test_override_reason_max_length(self):
        override = RedactionOverride(enabled=True, reason="r" * 256)
        assert len(override.reason) == 256

    def test_override_reason_too_long_rejected(self):
        with pytest.raises(ValidationError):
            RedactionOverride(enabled=True, reason="r" * 257)

    def test_redaction_override_acknowledgements_list(self):
        override = RedactionOverride(
            enabled=True,
            acknowledgements=["ack1", "ack2"],
        )
        assert override.acknowledgements == ["ack1", "ack2"]


# ===========================================================================
# 9. schema_version validator
# ===========================================================================

class TestSchemaVersion:
    def test_valid_semver_accepted(self):
        call = make_call(schema_version="2.3.0")
        assert call.schema_version == "2.3.0"

    def test_non_semver_rejected(self):
        with pytest.raises(ValidationError, match="semver"):
            make_call(schema_version="1.0")

    def test_non_semver_with_prerelease_rejected(self):
        with pytest.raises(ValidationError):
            make_call(schema_version="1.0.0-alpha")


# ===========================================================================
# 10. Target construction
# ===========================================================================

class TestTarget:
    def test_model_target(self):
        t = Target(type=TargetType.model, alias="qwen:32b")
        assert t.type == TargetType.model

    def test_tool_target(self):
        t = Target(type=TargetType.tool, alias="api_hub:search")
        assert t.type == TargetType.tool

    def test_capability_tags_accepted(self):
        t = Target(
            type=TargetType.model,
            alias="qwen:32b",
            capability_tags=["code", "reasoning"],
        )
        assert "code" in t.capability_tags

    def test_too_many_capability_tags_rejected(self):
        with pytest.raises(ValidationError, match="Too many"):
            Target(
                type=TargetType.model,
                alias="qwen:32b",
                capability_tags=["tag"] * 65,
            )

    def test_capability_tag_too_long_rejected(self):
        with pytest.raises(ValidationError, match="tag too long"):
            Target(
                type=TargetType.model,
                alias="qwen:32b",
                capability_tags=["a" * 65],
            )

    def test_alias_min_length(self):
        t = Target(type=TargetType.model, alias="x")
        assert t.alias == "x"

    def test_alias_empty_rejected(self):
        with pytest.raises(ValidationError):
            Target(type=TargetType.model, alias="")
