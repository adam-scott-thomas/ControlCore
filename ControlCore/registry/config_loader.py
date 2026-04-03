"""Load ghostrouter.toml into typed config objects."""
import tomllib
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from ControlCore.registry.budget import BudgetConfig
from ControlCore.registry.preferences import Preferences, AffinityRule


@dataclass
class RouterConfig:
    budget: BudgetConfig
    learning_db_path: str
    preferences: Preferences


def load_router_config(path: Optional[Path] = None) -> RouterConfig:
    """Load from ghostrouter.toml. Returns defaults if file not found."""
    if path is None:
        # Look in cwd, then ~/.ghostrouter/
        candidates = [
            Path("ghostrouter.toml"),
            Path.home() / ".ghostrouter" / "ghostrouter.toml",
        ]
        for p in candidates:
            if p.exists():
                path = p
                break

    if path is None or not path.exists():
        return RouterConfig(
            budget=BudgetConfig(),
            learning_db_path="~/.ghostrouter/learning.db",
            preferences=Preferences(),
        )

    with open(path, "rb") as f:
        raw = tomllib.load(f)

    # Budget
    budget_raw = raw.get("budget", {})
    budget = BudgetConfig(
        daily_limit=budget_raw.get("daily_limit", 0.0),
        hourly_limit=budget_raw.get("hourly_limit", 0.0),
    )

    # Learning
    learning_raw = raw.get("learning", {})
    db_path = learning_raw.get("db_path", "~/.ghostrouter/learning.db")

    # Preferences
    prefs_raw = raw.get("preferences", {})
    affinities = []
    for rule in prefs_raw.get("affinity", []):
        affinities.append(AffinityRule(
            intent=rule.get("intent", "*"),
            model_alias=rule.get("model", ""),
            boost=rule.get("boost", 10.0),
        ))

    caller_blocklists = {}
    caller_preferred = {}
    for caller_name, caller_conf in prefs_raw.get("caller", {}).items():
        if "blocklist" in caller_conf:
            caller_blocklists[caller_name] = caller_conf["blocklist"]
        if "preferred" in caller_conf:
            caller_preferred[caller_name] = caller_conf["preferred"]

    preferences = Preferences(
        affinities=affinities,
        caller_blocklists=caller_blocklists,
        caller_preferred=caller_preferred,
    )

    return RouterConfig(budget=budget, learning_db_path=db_path, preferences=preferences)
