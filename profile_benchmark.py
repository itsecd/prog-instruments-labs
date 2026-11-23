"""
Скрипт для замера производительности Currency Converter (До оптимизации).
Использует мок-HTML для изоляции затрат CPU на парсинг от сетевых задержек.
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

# Создаем реалистичный мок-HTML с таблицей курсов, чтобы не зависеть от сети
MOCK_HTML = """
<html><body>
<table class="tablesorter ratesTable">
    <tr><td>US Dollar</td><td>0.01205</td><td>82.9902</td></tr>
    <tr><td>Euro</td><td>0.01102</td><td>90.7451</td></tr>
    <tr><td>British Pound</td><td>0.00950</td><td>105.2631</td></tr>
    <tr><td>Japanese Yen</td><td>1.85000</td><td>0.5405</td></tr>
    <tr><td>Chinese Yuan Renmimbi</td><td>0.08700</td><td>11.4942</td></tr>
</table>
</body></html>
"""

Soup = BeautifulSoup(MOCK_HTML, 'html.parser')

fig3 = plt.Figure(figsize=(5, 3), dpi=100)
dummy_root = Tk()
dummy_frame = Frame(dummy_root)

def simulate_bottlenecks():
    """Симулирует 1000 нажатий кнопки Convert в исходном коде."""
    currency_name = "US Dollar"
    
    for _ in range(1000):
        # Узкое место 1: Медленный поиск по DOM с lambda-функцией 
        rate_str = Soup.find('table', class_='tablesorter ratesTable').find('td', text=lambda x: x == currency_name).find_next('td').get_text()
        rate = float(rate_str)
        
        # Узкое место 2: Создание нового Canvas при каждом обновлении (утечка RAM)
        scatter3 = FigureCanvasTkAgg(fig3, dummy_frame)
        fig3.gca().clear()
        fig3.gca().scatter(10, 10 * rate)

if __name__ == "__main__":
    tracemalloc.start()
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    simulate_bottlenecks()
    
    profiler.disable()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    print(f"\nRAM: Пиковое использование: {peak / 1024 / 1024:.2f} MB")
    
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats(SortKey.CUMULATIVE)
    stats.print_stats(10) # Топ-10 самых медленных вызовов
    
    with open("profile_before.txt", "w", encoding="utf-8") as f:
        f.write(f"Peak RAM: {peak / 1024 / 1024:.2f} MB\n")
        f.write(stream.getvalue())
    print("Отчет 'До' сохранен в profile_before.txt")