import importlib
from typing import Dict


class GameLauncher:
    """Класс для управления запуском игр."""

    def __init__(self) -> None:
        """Инициализация игрового лаунчера с доступными играми."""
        self.games: Dict[str, str] = {
            "Угадай число": "games.guess_the_number",
            "Крестики-нолики": "games.tic_tac_toe",
            "Виселица": "games.hangman",
            "2048": "games.game_2048",
            "Лабиринт": "games.maze"
        }
        self.game_functions: Dict[str, str] = {
            "Угадай число": "guess_the_number",
            "Крестики-нолики": "tic_tac_toe",
            "Виселица": "hangman",
            "2048": "game_2048",
            "Лабиринт": "maze"
        }

    def display_games(self) -> None:
        """Отображение доступных игр."""
        print("Доступные игры:")
        for game in self.games.keys():
            print(f"- {game}")

    def launch_game(self, selected_game: str) -> None:
        """
        Запуск выбранной игры.

        Args:
            selected_game (str): Название выбранной игры.
        """
        game_module_name = self.games.get(selected_game)

        if game_module_name:
            game_module = importlib.import_module(game_module_name)
            function_name = self.game_functions.get(selected_game)
            if function_name:
                game_module_function = getattr(game_module, function_name)
                game_module_function()
            else:
                print("Извините, функция игры не найдена.")
        else:
            print("Извините, такой игры нет. Пожалуйста, выберите из списка.")

    def run(self) -> None:
        """Основной игровой процесс."""
        while True:
            self.display_games()
            selected_game = input(
                "Введите название игры, в которую хотите поиграть (или 'выход' для завершения): "
            )

            if selected_game.lower() == "выход":
                print("Спасибо за игру! До свидания!")
                break

            self.launch_game(selected_game)


if __name__ == "__main__":
    game_launcher = GameLauncher()
    game_launcher.run()
