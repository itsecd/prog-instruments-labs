import unittest
from unittest.mock import patch, MagicMock
from io import StringIO
from launcher import GameLauncher


class TestGameLauncher(unittest.TestCase):

    def setUp(self):
        """Создаем экземпляр GameLauncher перед каждым тестом."""
        self.launcher = GameLauncher()

    @patch('sys.stdout', new_callable=StringIO)
    def test_display_games(self, mock_stdout):
        """Тестирование отображения доступных игр."""
        self.launcher.display_games()
        output = mock_stdout.getvalue().strip()
        expected_output = "Доступные игры:\n- Угадай число\n- Крестики-нолики\n- Виселица\n- 2048\n- Лабиринт"
        self.assertEqual(output, expected_output)

    @patch('builtins.input', side_effect=['Крестики-нолики', 'выход'])
    @patch('importlib.import_module')
    def test_launch_game_function_not_found(self, mock_import_module, mock_input):
        """Тестирование случая, когда функция игры не найдена."""
        mock_game_module = MagicMock()
        mock_import_module.return_value = mock_game_module

        self.launcher.launch_game('Крестики-нолики')

        # Проверяем, что функция не была вызвана
        mock_game_module.assert_not_called()

    @patch('builtins.input', side_effect=['выход'])
    @patch('sys.stdout', new_callable=StringIO)
    def test_run_exit(self, mock_stdout, mock_input):
        """Тестирование выхода из игрового лаунчера."""
        self.launcher.run()
        output = mock_stdout.getvalue().strip()
        self.assertIn("Спасибо за игру! До свидания!", output)


if __name__ == "__main__":
    unittest.main()
