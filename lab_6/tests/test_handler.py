"""Жизненный цикл соединения и очистка общего списка клиентов."""

import asyncio

import pytest

import main as server


def test_handler_registers_client_before_waiting(websocket_factory):
    observed = []
    client = websocket_factory()
    client.on_close = lambda: observed.append(client in server.clients)

    asyncio.run(server.handler(client))

    assert observed == [True]


def test_handler_waits_for_connection_close_once(websocket_factory):
    client = websocket_factory()

    asyncio.run(server.handler(client))

    assert client.wait_closed_calls == 1


def test_handler_removes_disconnected_client(websocket_factory):
    client = websocket_factory()

    asyncio.run(server.handler(client))

    assert server.clients == set()


def test_handler_keeps_other_clients(websocket_factory):
    other = websocket_factory()
    server.clients.add(other)

    asyncio.run(server.handler(websocket_factory()))

    assert server.clients == {other}


@pytest.mark.parametrize("error_type", [ConnectionError, RuntimeError])
def test_handler_cleans_up_when_waiting_fails(error_type, websocket_factory):
    client = websocket_factory(close_error=error_type("connection failed"))

    with pytest.raises(error_type, match="connection failed"):
        asyncio.run(server.handler(client))

    assert server.clients == set()


def test_handler_cleans_up_when_cancelled(websocket_factory):
    client = websocket_factory(close_error=asyncio.CancelledError())

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(server.handler(client))

    assert server.clients == set()


def test_handler_tolerates_client_already_removed_by_broadcast(websocket_factory):
    client = websocket_factory()
    client.on_close = lambda: server.clients.discard(client)

    asyncio.run(server.handler(client))

    assert server.clients == set()
