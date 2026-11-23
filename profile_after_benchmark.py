"""
Скрипт для замера производительности Currency Converter (После оптимизации).
Эмулирует 1000 нажатий кнопки Convert с оптимизированной логикой.
"""
import cProfile
import pstats
import tracemalloc
import io
from pstats import SortKey
from bs4 import BeautifulSoup
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import Frame, Tk

MOCK_HTML = """
<html><body>
<table class="tablesorter ratesTable">
    <tr><td>US Dollar</td><td>0.01205</td></tr>
    <tr><td>Euro</td><td>0.01102</td></tr>
    <tr><td>British Pound</td><td>0.00950</td></tr>
    <tr><td>Japanese Yen</td><td>1.85000</td></tr>
    <tr><td>Chinese Yuan Renmimbi</td><td>0.08700</td></tr>
</table>
</body></html>
"""

soup = BeautifulSoup(MOCK_HTML, "html.parser")
rates_dict = {}
for row in soup.find("table", class_="tablesorter ratesTable").find_all("tr"):
    cells = row.find_all("td")
    if len(cells) >= 2:
        try:
            rates_dict[cells[0].get_text(strip=True)] = float(cells[1].get_text(strip=True))
        except ValueError:
            pass

fig3 = plt.Figure(figsize=(5, 3), dpi=100)
ax3 = fig3.add_subplot(111)
dummy_root = Tk()
dummy_frame = Frame(dummy_root)
canvas = FigureCanvasTkAgg(fig3, dummy_frame)
canvas.get_tk_widget().grid(row=0, column=0)

scatter_plot = None


def simulate_optimized():
    """Симулирует 1000 нажатий кнопки Convert с оптимизацией."""
    global scatter_plot
    currency_name = "US Dollar"

    for _ in range(1000):
        # Быстрый поиск O(1) вместо Soup.find с lambda
        rate = rates_dict[currency_name]

        # Обновление данных существующего scatter вместо clear+scatter
        if scatter_plot is None:
            scatter_plot = ax3.scatter([10], [10 * rate])
        else:
            scatter_plot.set_offsets([[10, 10 * rate]])
        canvas.draw()


if __name__ == "__main__":
    tracemalloc.start()

    profiler = cProfile.Profile()
    profiler.enable()

    simulate_optimized()

    profiler.disable()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"\nRAM: Пиковое использование: {peak / 1024 / 1024:.2f} MB")

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats(SortKey.CUMULATIVE)
    stats.print_stats(10)

    with open("profile_after.txt", "w", encoding="utf-8") as f:
        f.write(f"Peak RAM: {peak / 1024 / 1024:.2f} MB\n")
        f.write(stream.getvalue())
    print("Отчет 'После' сохранен в profile_after.txt")