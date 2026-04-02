import pytest
from spine import Core


@pytest.fixture(autouse=True)
def reset():
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


def test_boot_idempotent():
    from ControlCore.boot import boot
    c1 = boot()
    c2 = boot()
    assert c1 is c2
