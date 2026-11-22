"""Формат сообщений, рассылка и ошибки отдельных WebSocket-клиентов."""

import asyncio
from datetime import datetime
import json

import pytest
from websockets.exceptions import ConnectionClosedError

import main as server


@pytest.mark.parametrize(
    "moment, expected",
    [
        (datetime(2026, 11, 22, 12, 34, 56), "2026-11-22 12:34:56"),
        (datetime(2026, 1, 2, 3, 4, 5), "2026-01-02 03:04:05"),
        (datetime(2024, 2, 29), "2024-02-29 00:00:00"),
        (datetime(2026, 12, 31, 23, 59, 59), "2026-12-31 23:59:59"),
    ],
    ids=["regular", "leading-zeroes", "leap-day", "year-end"],
)
def test_broadcast_sends_date_json(moment, expected, monkeypatch,
                                   websocket_factory, run_broadcast):
    client = websocket_factory()
    server.clients.add(client)
    monkeypatch.setattr(server.datetime, "now", lambda: moment)

    run_broadcast()

    assert [json.loads(message) for message in client.messages] == [
        {"type": "date", "data": expected}
    ]


@pytest.mark.parametrize("client_count", [1, 2, 5])
def test_broadcast_delivers_to_every_client(client_count, websocket_factory,
                                           run_broadcast):
    connected = [websocket_factory() for _ in range(client_count)]
    server.clients.update(connected)

    run_broadcast()

    expected = [{"type": "date", "data": "2026-11-22 12:34:56"}]
    assert [
        [json.loads(message) for message in client.messages]
        for client in connected
    ] == [expected] * client_count


def test_broadcast_keeps_successful_clients(websocket_factory, run_broadcast):
    client = websocket_factory()
    server.clients.add(client)

    run_broadcast()

    assert server.clients == {client}


def test_broadcast_waits_when_no_clients_are_connected(run_broadcast):
    delays = run_broadcast(cycles=2)

    assert delays == [1, 1]


def test_broadcast_updates_date_on_next_iteration(monkeypatch, websocket_factory,
                                                run_broadcast):
    moments = iter([datetime(2026, 12, 31, 23, 59, 59), datetime(2027, 1, 1)])
    monkeypatch.setattr(server.datetime, "now", lambda: next(moments))
    client = websocket_factory()
    server.clients.add(client)

    run_broadcast(cycles=2)

    assert [json.loads(message)["data"] for message in client.messages] == [
        "2026-12-31 23:59:59", "2027-01-01 00:00:00"
    ]


@pytest.mark.parametrize(
    "error_factory",
    [ConnectionError, RuntimeError, lambda: ConnectionClosedError(None, None)],
    ids=["connection-error", "send-error", "websocket-closed"],
)
def test_broadcast_removes_failed_client(error_factory, websocket_factory,
                                        run_broadcast):
    client = websocket_factory(send_error=error_factory())
    server.clients.add(client)

    run_broadcast()

    assert server.clients == set()


def test_broadcast_continues_after_failed_client(websocket_factory, run_broadcast):
    broken = websocket_factory(send_error=ConnectionError())
    healthy = websocket_factory()
    server.clients.update([broken, healthy])

    run_broadcast(cycles=2)

    assert (server.clients, len(healthy.messages), broken.messages) == (
        {healthy}, 2, []
    )


def test_broadcast_handles_all_clients_failing(websocket_factory, run_broadcast):
    server.clients.update(websocket_factory(send_error=ConnectionError())
                          for _ in range(3))

    delays = run_broadcast(cycles=2)

    assert (server.clients, delays) == (set(), [1, 1])


def test_broadcast_tolerates_client_removed_during_send(websocket_factory,
                                                      run_broadcast):
    client = websocket_factory(send_error=ConnectionError())
    client.on_send = lambda: server.clients.discard(client)
    server.clients.add(client)

    run_broadcast()

    assert server.clients == set()


def test_broadcast_does_not_swallow_task_cancellation(websocket_factory,
                                                    run_broadcast):
    client = websocket_factory(send_error=asyncio.CancelledError())
    server.clients.add(client)

    delays = run_broadcast()

    assert (server.clients, delays) == ({client}, [])
