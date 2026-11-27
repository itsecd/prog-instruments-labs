#!/usr/bin/env python3
"""HTTP page and WebSocket server that broadcasts the current time."""

import asyncio
import json
from contextlib import suppress
from datetime import datetime
from pathlib import Path

from aiohttp import web
from websockets.asyncio.server import ServerConnection, serve
from websockets.exceptions import ConnectionClosed

HOST = "localhost"
HTTP_PORT = 7878
WEBSOCKET_PORT = 8020
INDEX_PATH = Path(__file__).with_name("index.html")


class DateBroadcaster:
    """Track WebSocket clients and broadcast the current local time."""

    def __init__(self) -> None:
        self.clients: set[ServerConnection] = set()

    async def handler(self, websocket: ServerConnection) -> None:
        self.clients.add(websocket)
        try:
            await websocket.wait_closed()
        finally:
            self.clients.discard(websocket)

    async def broadcast(self) -> None:
        while True:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            message = json.dumps({"type": "date", "data": now})

            for client in tuple(self.clients):
                try:
                    await client.send(message)
                except ConnectionClosed:
                    self.clients.discard(client)

            await asyncio.sleep(1)


async def index(_request: web.Request) -> web.FileResponse:
    return web.FileResponse(INDEX_PATH)


async def main() -> None:
    broadcaster = DateBroadcaster()

    app = web.Application()
    app.router.add_get("/", index)
    runner = web.AppRunner(app)
    await runner.setup()

    try:
        site = web.TCPSite(runner, HOST, HTTP_PORT)
        await site.start()

        async with serve(broadcaster.handler, HOST, WEBSOCKET_PORT):
            broadcast_task = asyncio.create_task(broadcaster.broadcast())
            try:
                print(f"HTTP server is running on http://{HOST}:{HTTP_PORT}")
                print(f"WebSocket server is running on ws://{HOST}:{WEBSOCKET_PORT}")
                await asyncio.Event().wait()
            finally:
                broadcast_task.cancel()
                with suppress(asyncio.CancelledError):
                    await broadcast_task
    finally:
        await runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
