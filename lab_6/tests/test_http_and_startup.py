"""HTTP-обработчик и настройка сервера без открытия сетевых портов."""

import asyncio
import importlib.util
import runpy
from types import SimpleNamespace

import pytest

import main as server


def test_index_returns_file_response(monkeypatch):
    requested_paths = []
    expected_response = object()

    def file_response(path):
        requested_paths.append(path)
        return expected_response

    monkeypatch.setattr(server, "web", SimpleNamespace(FileResponse=file_response))

    response = asyncio.run(server.index(object()))

    assert (response, requested_paths) == (expected_response, ["index.html"])


def test_import_does_not_start_server(monkeypatch):
    started = []

    def record_run(coroutine):
        started.append(coroutine.cr_code.co_name)
        coroutine.close()

    monkeypatch.setattr(asyncio, "run", record_run)
    spec = importlib.util.spec_from_file_location("server_import_check", server.__file__)
    module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(module)

    assert started == []


def test_script_entrypoint_runs_main(monkeypatch):
    started = []

    def record_run(coroutine):
        started.append(coroutine.cr_code.co_name)
        coroutine.close()

    monkeypatch.setattr(asyncio, "run", record_run)

    runpy.run_path(server.__file__, run_name="__main__")

    assert started == ["main"]


def test_main_configures_and_starts_servers(monkeypatch, capsys):
    events = []
    # Реальный aiohttp-маршрутизатор; подменены только запуск сети и задачи.
    app = server.web.Application()

    def create_task(coroutine):
        events.append(("task", coroutine.cr_code.co_name))
        coroutine.close()

    async def serve(handler, host, port):
        events.append(("websocket", handler, host, port))

    async def setup():
        events.append(("setup",))

    def app_runner(application):
        events.append(("runner", application))
        return runner

    async def start():
        events.append(("start",))

    def tcp_site(received_runner, host, port):
        events.append(("http", received_runner, host, port))
        return SimpleNamespace(start=start)

    async def stop_sleep(delay):
        events.append(("sleep", delay))
        raise asyncio.CancelledError

    runner = SimpleNamespace(setup=setup)
    monkeypatch.setattr(server, "url", "127.0.0.1")
    monkeypatch.setattr(server, "port", 9000)
    monkeypatch.setattr(server, "socket_port", 9001)
    monkeypatch.setattr(server, "asyncio", SimpleNamespace(
        create_task=create_task, sleep=stop_sleep))
    monkeypatch.setattr(server, "websockets", SimpleNamespace(serve=serve))
    monkeypatch.setattr(server, "web", SimpleNamespace(
        Application=lambda: app, AppRunner=app_runner, TCPSite=tcp_site))

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(server.main())

    assert events == [
        ("task", "send_date"),
        ("websocket", server.handler, "127.0.0.1", 9001),
        ("runner", app),
        ("setup",),
        ("http", runner, "127.0.0.1", 9000),
        ("start",),
        ("sleep", 3600),
    ]
    assert [(route.method, route.resource.canonical, route.handler)
            for route in app.router.routes()] == [
        ("HEAD", "/", server.index),
        ("GET", "/", server.index),
    ]
    assert capsys.readouterr().out.splitlines() == [
        "HTTP server is running on http://127.0.0.1:9000",
        "Websocket server is running on ws://127.0.0.1:9001",
    ]
