from typing import List
from hash_card import CardSearcher


class PerformanceBenchmark:
    """Класс для тестирования производительности поиска номера карты при разном числе процессов.

    Позволяет:
    - Запускать тесты производительности для разного количества процессов
    - Визуализировать результаты тестирования
    - Определять оптимальное количество процессов для поиска

    Attributes:
        bins (List[str]): Список BIN-кодов для тестирования
        results (List[Tuple[int, float]]): Результаты тестов (процессы, время)
    """

    def __init__(self, bins: List[str]):
        """Инициализирует экземпляр PerformanceBenchmark.

        Args:
            bins: Список BIN-кодов карт для использования в тестах
        """
        self.bins = bins
        self.results = []

    def run_tests(self, max_processes: int):
        """Выполняет серию тестов производительности.

        Запускает поиск номера карты для каждого количества процессов от 1 до max_processes,
        измеряя время выполнения каждого теста.

        Args:
            max_processes: Максимальное количество процессов для тестирования

        Note:
            Для каждого теста создается новый экземпляр CardSearcher для чистоты измерений.
            Результаты сохраняются в self.results в формате (процессы, время).
        """
        for nproc in range(1, max_processes + 1):
            searcher = CardSearcher()
            _, duration = searcher.search_card_number(nproc)
            self.results.append((nproc, duration))
            print(f"Test completed with {nproc} processes, took {duration:.2f} sec.")

    def plot_performance(self):
        """Визуализирует результаты тестирования производительности.

        Строит график зависимости времени выполнения от количества процессов.
        Использует библиотеку matplotlib для отображения графика.

        Raises:
            ValueError: Если тесты не были выполнены (results пуст)
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Matplotlib не установлен. График не может быть построен.")
            print("Результаты:", self.results)
            return

        if not self.results:
            raise ValueError("Сначала выполните тесты с помощью run_tests()")

        processes, durations = zip(*self.results)

        plt.figure(figsize=(10, 6))
        plt.plot(processes, durations, marker='o', linestyle='-', color='blue')
        plt.title("Зависимость времени поиска от числа процессов")
        plt.xlabel("Количество процессов")
        plt.ylabel("Время выполнения (секунды)")
        plt.grid(True)
        plt.show()