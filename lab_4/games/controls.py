from typing import Dict


class Controls:
    def __init__(self) -> None:
        """Инициализация управления клавишами."""
        self.key_map: Dict[str, str] = {
            'w': 'up',
            'a': 'left',
            's': 'down',
            'd': 'right'
        }

    def get_direction(self, input_key: str) -> str:
        """Получение направления на основе нажатой клавиши.

        Args:
            input_key (str): Нажатая клавиша.

        Returns:
            str: Направление движения или None, если клавиша не распознана.
        """
        return self.key_map.get(input_key)

    def display_controls(self) -> None:
        """Вывод доступных клавиш управления."""
        print("Доступные клавиши управления:")
        for key, direction in self.key_map.items():
            print(f"{key.upper()} - {direction.capitalize()}")
