from pathlib import Path


def main() -> None:
    """Создаёт файл с анкетами для профилирования."""
    path = Path(__file__).with_name("input.txt")

    with path.open("w", encoding="utf-8") as file:
        for i in range(1, 20001):
            file.write(
                f"{i})\n"
                f"Фамилия: Иванов{i % 1000}\n"
                f"Имя: Иван{i % 100}\n\n"
            )


if __name__ == "__main__":
    main()
