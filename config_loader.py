import os
import sys
from pathlib import Path
from typing import Optional

from omegaconf import OmegaConf, DictConfig


CONFIGS_DIR = Path(__file__).parent / "configs"
VALID_ENVIRONMENTS = ("dev", "prod")


def load_config(env: Optional[str] = None) -> DictConfig:
    """Загружает базовый конфиг и применяет переопределения окружения.

    Порядок мерджа (каждый следующий слой переопределяет предыдущий):
        1. configs/base.yaml
        2. configs/{env}.yaml (если задан env)

    Значение env определяется в порядке приоритета:
        1. Аргумент функции
        2. Переменная окружения APP_ENV
        3. Значение по умолчанию ("dev")
    """
    if env is None:
        env = os.environ.get("APP_ENV", "dev")

    env = env.lower()
    if env not in VALID_ENVIRONMENTS:
        print(
            f"Неверное окружение '{env}'. "
            f"Допустимые значения: {', '.join(VALID_ENVIRONMENTS)}. "
            "Используется 'dev'.",
            file=sys.stderr,
        )
        env = "dev"

    base_path = CONFIGS_DIR / "base.yaml"
    env_path = CONFIGS_DIR / f"{env}.yaml"

    if not base_path.exists():
        raise FileNotFoundError(f"Базовый конфиг не найден: {base_path}")

    # Загружаем базовый слой
    base_config = OmegaConf.load(base_path)

    # Применяем слой окружения, если файл существует
    if env_path.exists():
        env_config = OmegaConf.load(env_path)
        # OmegaConf.merge рекурсивно объединяет конфиги,
        # значения из env_config перекрывают значения из base_config
        config = OmegaConf.merge(base_config, env_config)
    else:
        config = base_config

    # Сохраняем активное окружение в самом конфиге для удобства
    config.environment = env

    return config


if __name__ == "__main__":
    # Демонстрация: запуск с разными окружениями
    print("Конфиг для DEV окружения")
    dev_cfg = load_config("dev")
    print(OmegaConf.to_yaml(dev_cfg, resolve=True))

    print("\nКонфиг для PROD окружения")
    prod_cfg = load_config("prod")
    print(OmegaConf.to_yaml(prod_cfg, resolve=True))