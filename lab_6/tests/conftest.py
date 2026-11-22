"""Общие фикстуры: каждый тест получает отдельное состояние сервера."""

import asyncio
from datetime import datetime
from types import SimpleNamespace

import pytest

import main as server


class WebSocketStub:
    """Соединение без сети, которое сохраняет сообщения и имитирует ошибки."""

    def __init__(self, send_error=None, close_error=None, on_send=None, on_close=None):
        self.messages = []
        self.send_error = send_error
        self.close_error = close_error
        self.on_send = on_send
        self.on_close = on_close
        self.wait_closed_calls = 0

    async def send(self, message):
        if self.on_send is not None:
            self.on_send()
        if self.send_error is not None:
            raise self.send_error
        self.messages.append(message)

    async def wait_closed(self):
        self.wait_closed_calls += 1
        if self.on_close is not None:
            self.on_close()
        if self.close_error is not None:
            raise self.close_error


@pytest.fixture(autouse=True)
def isolated_clients(monkeypatch):
    monkeypatch.setattr(server, "clients", set())


@pytest.fixture
def websocket_factory():
    return WebSocketStub


@pytest.fixture
def frozen_time(monkeypatch):
    value = datetime(2026, 11, 22, 12, 34, 56)
    clock = SimpleNamespace(now=lambda: value)
    monkeypatch.setattr(server, "datetime", clock)
    return value


@pytest.fixture
def run_broadcast(monkeypatch, frozen_time):
    """Выполняет нужное число проходов send_date без настоящего ожидания."""

    def run(cycles=1):
        delays = []

        async def controlled_sleep(delay):
            delays.append(delay)
            if len(delays) >= cycles:
                raise asyncio.CancelledError

        # Меняем ссылку внутри main, а не asyncio.sleep для всего интерпретатора.
        monkeypatch.setattr(server, "asyncio", SimpleNamespace(sleep=controlled_sleep))
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(server.send_date())
        return delays

    return run
