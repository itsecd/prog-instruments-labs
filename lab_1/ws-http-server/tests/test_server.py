"""Integration tests for the original HTTP and WebSocket behavior."""

import asyncio
import json
from contextlib import suppress
from datetime import datetime

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve

from main import DateBroadcaster, index


@pytest.mark.asyncio
async def test_http_returns_index() -> None:
    app = web.Application()
    app.router.add_get("/", index)

    async with TestClient(TestServer(app)) as client:
        response = await client.get("/")
        assert response.status == 200
        body = await response.text()
        assert 'new WebSocket("ws://localhost:8020")' in body


@pytest.mark.asyncio
async def test_websocket_broadcasts_date() -> None:
    broadcaster = DateBroadcaster()

    async with serve(broadcaster.handler, "127.0.0.1", 0) as server:
        sockets = server.sockets
        assert sockets is not None
        port = sockets[0].getsockname()[1]

        task = asyncio.create_task(broadcaster.broadcast())
        try:
            async with connect(f"ws://127.0.0.1:{port}") as websocket:
                raw_message = await asyncio.wait_for(websocket.recv(), timeout=3)
                assert isinstance(raw_message, str)
                message = json.loads(raw_message)
                assert message["type"] == "date"
                datetime.strptime(message["data"], "%Y-%m-%d %H:%M:%S")
        finally:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
