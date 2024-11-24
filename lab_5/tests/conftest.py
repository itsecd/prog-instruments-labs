import pytest
import asyncio


@pytest.fixture(autouse=True, scope='session')
def prepare_env_file():
    new_env: str = None
    with open(".env.test", mode="r", encoding="utf-8") as f:
        new_env = f.read()

    with open(".env", mode="r+", encoding="utf-8") as f:
        old_env  = f.read()

        f.seek(0)
        f.truncate()

        f.write(new_env)

        f.seek(0)

    yield

    with open(".env", mode="w", encoding="utf-8") as f:
        f.write(old_env)

@pytest.fixture(scope="session")
def event_loop():
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
    yield loop
    loop.close()