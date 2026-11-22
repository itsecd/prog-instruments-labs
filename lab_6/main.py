#!/usr/bin/env python3 

from datetime import datetime
from aiohttp import web

import websockets
import asyncio
import json

url = 'localhost'
port = 7878
socket_port = 8020

clients = set()

async def send_date():
    while True:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        msg = json.dumps({"type": "date", "data": now})

        for client in list(clients):
            try:
                await client.send(msg)
            except Exception:
                clients.discard(client)
        
        await asyncio.sleep(1)

async def handler(websocket):
    clients.add(websocket)
    try:
        await websocket.wait_closed()
    finally:
        clients.discard(websocket)

async def index(request):
    return web.FileResponse('index.html')

async def main():
    asyncio.create_task(send_date())
    await websockets.serve(handler, url, socket_port)
    
    app = web.Application()
    app.router.add_get('/', index)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, url, port)
    await site.start()
    print(f'HTTP server is running on http://{url}:{port}')
    print(f'Websocket server is running on ws://{url}:{socket_port}')

    while True:
        await asyncio.sleep(3600)

if __name__ == '__main__':
    asyncio.run(main())
