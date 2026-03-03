import logging

import pytest

from speaches.config import IDLE_OFFLOAD_DEFAULT_SECONDS, resolve_idle_offload_seconds
from speaches.executors.shared.base_model_manager import SelfDisposingModel


def test_idle_offload_default_is_300() -> None:
    assert resolve_idle_offload_seconds(environ={}) == IDLE_OFFLOAD_DEFAULT_SECONDS


@pytest.mark.parametrize("raw_value", ["abc", "-2"])
def test_invalid_idle_offload_values_fall_back_to_default(raw_value: str, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING)

    resolved = resolve_idle_offload_seconds(environ={"WHISPER_IDLE_OFFLOAD_SECONDS": raw_value})

    assert resolved == IDLE_OFFLOAD_DEFAULT_SECONDS
    assert "Invalid idle offload seconds" in caplog.text


def test_idle_offload_env_var_minus_one_disables_offload() -> None:
    assert resolve_idle_offload_seconds(environ={"WHISPER_IDLE_OFFLOAD_SECONDS": "-1"}) == -1


def test_idle_offload_env_var_zero_unloads_immediately() -> None:
    assert resolve_idle_offload_seconds(environ={"WHISPER_IDLE_OFFLOAD_SECONDS": "0"}) == 0


def test_idle_offload_env_var_sets_to_300() -> None:
    assert resolve_idle_offload_seconds(environ={"WHISPER_IDLE_OFFLOAD_SECONDS": "300"}) == 300


def test_minus_one_idle_offload_disables_scheduling() -> None:
    idle_offload_seconds = resolve_idle_offload_seconds(environ={"WHISPER_IDLE_OFFLOAD_SECONDS": "-1"})
    model = SelfDisposingModel[str](model_id="m", load_fn=lambda: "loaded", ttl=idle_offload_seconds)

    with model:
        pass

    assert model.expire_timer is None
    assert model.model == "loaded"
