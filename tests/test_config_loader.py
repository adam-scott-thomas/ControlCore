"""Tests for ControlCore/registry/config_loader.py"""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from ControlCore.registry.config_loader import load_router_config, RouterConfig
from ControlCore.registry.budget import BudgetConfig
from ControlCore.registry.preferences import Preferences, AffinityRule


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_toml(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "ghostrouter.toml"
    p.write_text(content, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Test 1: full TOML with all sections
# ---------------------------------------------------------------------------

FULL_TOML = """\
[budget]
daily_limit = 10.0
hourly_limit = 2.0

[learning]
db_path = "~/.ghostrouter/learning.db"

[[preferences.affinity]]
intent = "code"
model = "gpt4o"
boost = 20.0

[[preferences.affinity]]
intent = "draft"
model = "claude-sonnet"
boost = 15.0

[preferences.caller.alice]
blocklist = ["gpt4o"]
preferred = "claude-sonnet"

[preferences.caller.bob]
blocklist = ["claude-sonnet"]
"""


def test_load_full_toml(tmp_path):
    path = _write_toml(tmp_path, FULL_TOML)
    cfg = load_router_config(path)

    # Budget
    assert cfg.budget.daily_limit == 10.0
    assert cfg.budget.hourly_limit == 2.0

    # Learning db path
    assert cfg.learning_db_path == "~/.ghostrouter/learning.db"

    # Affinity rules
    prefs = cfg.preferences
    # gpt4o gets +20 for intent "code"
    assert prefs.get_boost("gpt4o", intent="code") == 20.0
    assert prefs.get_boost("gpt4o", intent="draft") == 0.0
    # claude-sonnet gets +15 for intent "draft"
    assert prefs.get_boost("claude-sonnet", intent="draft") == 15.0
    assert prefs.get_boost("claude-sonnet", intent="code") == 0.0

    # Caller blocklist
    assert prefs.is_blocked("gpt4o", caller="alice") is True
    assert prefs.is_blocked("claude-sonnet", caller="alice") is False
    assert prefs.is_blocked("claude-sonnet", caller="bob") is True

    # Caller preferred
    assert prefs.get_preferred("alice") == "claude-sonnet"
    assert prefs.get_preferred("bob") is None


# ---------------------------------------------------------------------------
# Test 2: missing file — returns defaults
# ---------------------------------------------------------------------------

def test_missing_file_returns_defaults(tmp_path):
    path = tmp_path / "nonexistent.toml"
    cfg = load_router_config(path)

    assert isinstance(cfg, RouterConfig)
    assert cfg.budget.daily_limit == 0.0
    assert cfg.budget.hourly_limit == 0.0
    assert cfg.learning_db_path == "~/.ghostrouter/learning.db"
    assert isinstance(cfg.preferences, Preferences)
    # No rules — boost is always 0
    assert cfg.preferences.get_boost("any-model", intent="code") == 0.0


def test_no_path_no_candidates_returns_defaults(monkeypatch, tmp_path):
    """Passing path=None with no ghostrouter.toml anywhere reachable returns defaults."""
    # Ensure cwd has no ghostrouter.toml and home dir has none
    monkeypatch.chdir(tmp_path)
    # Patch Path.home() to a temp dir with no config
    home_stub = tmp_path / "fakehome"
    home_stub.mkdir()
    monkeypatch.setattr(Path, "home", staticmethod(lambda: home_stub))

    cfg = load_router_config(None)
    assert cfg.budget.daily_limit == 0.0
    assert cfg.budget.hourly_limit == 0.0
    assert cfg.learning_db_path == "~/.ghostrouter/learning.db"


# ---------------------------------------------------------------------------
# Test 3: empty file — returns defaults
# ---------------------------------------------------------------------------

def test_empty_file_returns_defaults(tmp_path):
    path = _write_toml(tmp_path, "")
    cfg = load_router_config(path)

    assert cfg.budget.daily_limit == 0.0
    assert cfg.budget.hourly_limit == 0.0
    assert cfg.learning_db_path == "~/.ghostrouter/learning.db"
    assert cfg.preferences.get_boost("any-model", intent="code") == 0.0


# ---------------------------------------------------------------------------
# Test 4: only budget section — budget populated, preferences default
# ---------------------------------------------------------------------------

BUDGET_ONLY_TOML = """\
[budget]
daily_limit = 5.0
hourly_limit = 1.5
"""


def test_budget_only_toml(tmp_path):
    path = _write_toml(tmp_path, BUDGET_ONLY_TOML)
    cfg = load_router_config(path)

    assert cfg.budget.daily_limit == 5.0
    assert cfg.budget.hourly_limit == 1.5

    # Learning and preferences should fall back to defaults
    assert cfg.learning_db_path == "~/.ghostrouter/learning.db"
    assert cfg.preferences.get_boost("any-model", intent="anything") == 0.0
    assert cfg.preferences.get_preferred("anyone") is None


# ---------------------------------------------------------------------------
# Test 5: affinity with wildcard intent
# ---------------------------------------------------------------------------

WILDCARD_TOML = """\
[[preferences.affinity]]
intent = "*"
model = "local-llm"
boost = 5.0
"""


def test_wildcard_affinity(tmp_path):
    path = _write_toml(tmp_path, WILDCARD_TOML)
    cfg = load_router_config(path)

    prefs = cfg.preferences
    # Wildcard should match any intent
    assert prefs.get_boost("local-llm", intent="code") == 5.0
    assert prefs.get_boost("local-llm", intent="draft") == 5.0
    assert prefs.get_boost("local-llm", intent="") == 5.0
    # Other models still get 0
    assert prefs.get_boost("gpt4o", intent="code") == 0.0


# ---------------------------------------------------------------------------
# Test 6: custom learning db_path
# ---------------------------------------------------------------------------

CUSTOM_DB_TOML = """\
[learning]
db_path = "/tmp/custom_learning.db"
"""


def test_custom_learning_db_path(tmp_path):
    path = _write_toml(tmp_path, CUSTOM_DB_TOML)
    cfg = load_router_config(path)

    assert cfg.learning_db_path == "/tmp/custom_learning.db"
    # Budget should be default (0 = unlimited)
    assert cfg.budget.daily_limit == 0.0
