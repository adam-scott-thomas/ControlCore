"""Spine bootstrap for ControlCore.

Optional — ControlCore works without spine. But if spine is booted,
any module can access registries via Core.instance().
"""
from __future__ import annotations
from typing import Optional
from pathlib import Path
from spine import Core


def boot(config_path: Optional[Path] = None) -> Core:
    """Boot spine with ControlCore registries."""
    from ControlCore.config import ControlCoreConfig, load_model_registry
    from ControlCore.adapters.cloud import create_all_cloud_adapters
    from ControlCore.adapters.executor import AdapterRegistry
    from ControlCore.circuit_breaker import CircuitBreakerRegistry

    def setup(c: Core) -> None:
        from ControlCore.registry.learning import LearningStore
        from ControlCore.registry.budget import BudgetTracker
        from ControlCore.registry.config_loader import load_router_config

        config = ControlCoreConfig()
        if config_path is not None:
            config.registry_path = str(config_path)

        model_registry = load_model_registry(config)
        adapter_registry = AdapterRegistry()
        for adapter in create_all_cloud_adapters().values():
            adapter_registry.register(adapter)

        router_config = load_router_config()
        c.register("model_registry", model_registry)
        c.register("adapter_registry", adapter_registry)
        c.register("circuit_registry", CircuitBreakerRegistry())
        c.register("learning", LearningStore(db_path=router_config.learning_db_path))
        c.register("budget", BudgetTracker(router_config.budget))
        c.register("preferences", router_config.preferences)
        c.boot(env="prod")

    return Core.boot_once(setup)
