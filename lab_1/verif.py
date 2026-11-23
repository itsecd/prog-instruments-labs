import csv
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

EXECUTABLE = Path("./lab1")
SIZES = [200, 400, 800, 1200, 1600, 2000]

CSV_FILE = Path("results.csv")
PLOT_FILE = Path("plot.png")


@dataclass
class BenchmarkEntry:
    size: int
    time: Optional[float]


def create_matrix(size: int, path: Path) -> None:
    matrix = np.random.uniform(-5, 5, (size, size))

    with path.open("w") as f:
        f.write(f"{size}\n")
        for row in matrix:
            f.write(" ".join(f"{v:.8f}" for v in row) + "\n")


def parse_time(output: str) -> Optional[float]:
    pattern = r"([0-9]+\.[0-9]+)"

    for line in output.splitlines():
        if "time" in line.lower() or "Execution" in line:
            match = re.search(pattern, line)
            if match:
                return float(match.group(1))

    return None


def run_program(a: Path, b: Path, result: Path) -> tuple[int, str]:
    process = subprocess.run(
        [str(EXECUTABLE.resolve()), str(a), str(b), str(result)],
        capture_output=True,
        text=True,
    )

    return process.returncode, process.stdout


def execute_case(size: int) -> BenchmarkEntry:
    file_a = Path(f"A-{size}.txt")
    file_b = Path(f"B-{size}.txt")
    file_res = Path(f"C-{size}.txt")

    print(f"\nРазмер {size}")

    create_matrix(size, file_a)
    create_matrix(size, file_b)

    code, output = run_program(file_a, file_b, file_res)

    if code != 0:
        print("Ошибка выполнения")
        return BenchmarkEntry(size, None)

    exec_time = parse_time(output)

    if exec_time is None:
        print("Не удалось распознать время")
        return BenchmarkEntry(size, None)

    print(f"Время: {exec_time:.6f} сек")

    return BenchmarkEntry(size, exec_time)


def save_csv(entries: list[BenchmarkEntry]) -> None:
    with CSV_FILE.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Size", "Time"])

        for e in entries:
            writer.writerow([e.size, f"{e.time:.6f}" if e.time else "N/A"])


def plot_results(entries: list[BenchmarkEntry]) -> None:
    valid: list[tuple[int, float]] = [
        (e.size, e.time) for e in entries if e.time is not None
    ]

    if not valid:
        print("Нет данных для графика")
        return

    sizes = [size for size, _ in valid]
    times = [t for _, t in valid]

    plt.figure(figsize=(9, 5))
    plt.plot(sizes, times, marker="o")
    plt.xlabel("Размер матрицы")
    plt.ylabel("Время выполнения (сек)")
    plt.title("Зависимость времени умножения от размера")
    plt.grid(True)

    plt.savefig(PLOT_FILE)


def main() -> None:
    if not EXECUTABLE.exists():
        raise RuntimeError("Исполняемый файл не найден")

    results = []

    print("Запуск серии экспериментов")

    for size in SIZES:
        entry = execute_case(size)
        results.append(entry)

    save_csv(results)
    plot_results(results)

    print("\nРезультаты:")

    for r in results:
        t = f"{r.time:.6f}" if r.time else "N/A"
        print(f"{r.size:6d}  {t}")


if __name__ == "__main__":
    main()
