import logging

import pytest

from speaches.config import IDLE_OFFLOAD_DEFAULT_SECONDS, resolve_idle_offload_seconds
from speaches.executors.shared.base_model_manager import SelfDisposingModel


def test_idle_offload_default_is_300() -> None:
    assert resolve_idle_offload_seconds(environ={}) == IDLE_OFFLOAD_DEFAULT_SECONDS


def test_idle_offload_env_var_sets_to_1800() -> None:
    assert resolve_idle_offload_seconds(environ={"WHISPER_IDLE_OFFLOAD_SECONDS": "1800"}) == 1800


def test_invalid_idle_offload_env_var_falls_back_with_warning(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING)

    resolved = resolve_idle_offload_seconds(environ={"WHISPER_IDLE_OFFLOAD_SECONDS": "abc"})

    assert resolved == IDLE_OFFLOAD_DEFAULT_SECONDS
    assert "Invalid idle offload seconds" in caplog.text


def test_zero_idle_offload_disables_scheduling() -> None:
    idle_offload_seconds = resolve_idle_offload_seconds(environ={"WHISPER_IDLE_OFFLOAD_SECONDS": "0"})
    model = SelfDisposingModel[str](model_id="m", load_fn=lambda: "loaded", ttl=idle_offload_seconds)

    with model:
        pass

    assert model.expire_timer is None
    assert model.model == "loaded"
