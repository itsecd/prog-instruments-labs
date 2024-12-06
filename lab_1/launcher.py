import importlib

class GameLauncher:

    def display_games(self):
        """Отображение доступных игр."""
        print("Доступные игры:")
        print("- Угадай число")
        print("- Крестики-нолики")
        print("- Виселица")
        print("- 2048")
        print("- Лабиринт")

    def launch_game(self, selected_game):
        """Запуск выбранной игры."""

        game_module = None
        if selected_game == "Угадай число":
            game_module = importlib.import_module("games.guess_the_number")
            game_module.guess_the_number()

        elif selected_game == "Крестики-нолики":
            game_module = importlib.import_module("games.tic_tac_toe")
            game_module.tic_tac_toe()

        elif selected_game == "Виселица":
            game_module = importlib.import_module("games.hangman")
            game_module.hangman()

        elif selected_game == "2048":
            game_module = importlib.import_module("games.game_2048")
            game_module.game_2048()

        elif selected_game == "Лабиринт":
            game_module = importlib.import_module("games.maze")
            game_module.maze()

        else:
            print("Извините, такой игры нет. Пожалуйста, выберите из списка.")

    def run(self):
        """Основной игровой процесс."""
        while True:
            self.display_games()
            selected_game = input("Введите название игры, в которую хотите поиграть (или 'выход' для завершения): ")

            if selected_game.lower() == 'выход':
                print("Спасибо за игру! До свидания!")
                break

            self.launch_game(selected_game)

if __name__ == "__main__":
    game_launcher = GameLauncher()
    game_launcher.run()