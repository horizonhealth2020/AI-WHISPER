from __future__ import annotations

from collections.abc import Generator
import logging

import pytest

from speaches.executors.shared.base_model_manager import BaseModelManager, SelfDisposingModel


class _FakeTimer:
    def __init__(self, interval: int, callback) -> None:  # noqa: ANN001
        self.interval = interval
        self.callback = callback
        self.started = False
        self.canceled = False

    def start(self) -> None:
        self.started = True

    def cancel(self) -> None:
        self.canceled = True


class _TestModelManager(BaseModelManager[str]):
    def __init__(self, ttl: int) -> None:
        super().__init__(ttl)

    def _load_fn(self, model_id: str) -> str:
        return f"loaded:{model_id}"

    def stream(self, model_id: str) -> Generator[int]:
        with self.load_model(model_id):
            for i in range(2):
                yield i


def test_refcount_underflow_does_not_schedule(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr("speaches.executors.shared.base_model_manager.threading.Timer", _FakeTimer)
    caplog.set_level(logging.WARNING)
    model = SelfDisposingModel[str](model_id="m", load_fn=lambda: "loaded", ttl=300)

    model._decrement_ref()

    assert model.ref_count == 0
    assert model.expire_timer is None
    assert "below zero" in caplog.text


def test_schedule_only_on_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("speaches.executors.shared.base_model_manager.threading.Timer", _FakeTimer)
    model = SelfDisposingModel[str](model_id="m", load_fn=lambda: "loaded", ttl=300)

    model._increment_ref()
    model._increment_ref()
    model._decrement_ref()
    assert model.ref_count == 1
    assert model.expire_timer is None

    model._decrement_ref()

    assert model.ref_count == 0
    assert isinstance(model.expire_timer, _FakeTimer)
    assert model.expire_timer.started


def test_timer_canceled_on_increment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("speaches.executors.shared.base_model_manager.threading.Timer", _FakeTimer)
    model = SelfDisposingModel[str](model_id="m", load_fn=lambda: "loaded", ttl=300)

    model._increment_ref()
    model._decrement_ref()
    timer = model.expire_timer
    assert isinstance(timer, _FakeTimer)

    model._increment_ref()

    assert timer.canceled
    assert model.expire_timer is None


def test_no_unload_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("speaches.executors.shared.base_model_manager.threading.Timer", _FakeTimer)
    model = SelfDisposingModel[str](model_id="m", load_fn=lambda: "loaded", ttl=0)

    with model:
        pass

    assert model.expire_timer is None
    assert model.model == "loaded"


def test_streaming_keeps_ref_until_consumed() -> None:
    manager = _TestModelManager(ttl=300)
    model_handle = manager.load_model("stream")

    stream = manager.stream("stream")
    assert model_handle.ref_count == 0

    first = next(stream)
    assert first == 0
    assert model_handle.ref_count == 1

    rest = list(stream)
    assert rest == [1]
    assert model_handle.ref_count == 0
