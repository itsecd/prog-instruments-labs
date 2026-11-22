from utils.config_loader import ConfigLoader


def test_load_and_cache(config_file):
    """Проверяет паттерн Singleton: создается один экземпляр класса и данные кэшируются."""
    loader_a = ConfigLoader()
    loader_b = ConfigLoader()
    assert loader_a is loader_b

    first = loader_a.load(config_file)
    assert first["crypto"]["rsa"][0] == 2048

    second = loader_a.load("nonexistent.json")
    assert first is second