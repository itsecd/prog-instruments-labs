"""Анализ ориентации изображений из CSV-файла с аннотациями."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

ORIENTATION_ORDER = {
    "горизонтальное": 0,
    "квадратное": 1,
    "вертикальное": 2,
}
ORIENTATION_COLORS = {
    "горизонтальное": "skyblue",
    "квадратное": "lightgreen",
    "вертикальное": "lightcoral",
}


def read_annotation(file_path: str | Path) -> pd.DataFrame:
    """Прочитать CSV без заголовка: абсолютный и относительный пути."""
    return pd.read_csv(
        file_path,
        sep=";",
        header=None,
        names=["absolute_path", "relative_path"],
        dtype="string",
    )


def determine_orientation(image_path: str | Path) -> str | None:
    """Вернуть ориентацию изображения либо None для недоступного файла."""
    try:
        with Image.open(image_path) as image:
            width, height = image.size
    except (OSError, ValueError, TypeError):
        return None

    if width == height:
        return "квадратное"
    if width > height:
        return "горизонтальное"
    return "вертикальное"


def add_orientation_column(df: pd.DataFrame) -> pd.DataFrame:
    """Добавить ориентации и исключить записи с недоступными картинками."""
    result = df.copy()
    result["orientation"] = result["absolute_path"].apply(determine_orientation)
    return result.dropna(subset=["orientation"]).copy()


def sort_by_orientation(df: pd.DataFrame, ascending: bool = True) -> pd.DataFrame:
    """Отсортировать по порядку: горизонтальная, квадратная, вертикальная."""
    result = df.copy()
    result["orientation_order"] = result["orientation"].map(ORIENTATION_ORDER)
    result = result.sort_values(
        by="orientation_order", ascending=ascending, kind="stable"
    )
    return result.drop(columns=["orientation_order"])


def filter_by_orientation(df: pd.DataFrame, orientation: str) -> pd.DataFrame:
    """Вернуть копию строк с указанной ориентацией."""
    return df.loc[df["orientation"] == orientation].copy()


def create_histogram(
    df: pd.DataFrame,
    save_path: str | Path = "orientation_histogram.png",
) -> None:
    """Сохранить гистограмму распределения ориентаций в PNG."""
    counts = df["orientation"].value_counts()
    fig, ax = plt.subplots(figsize=(8, 5))
    try:
        bars = ax.bar(
            counts.index,
            counts.values,
            color=[ORIENTATION_COLORS.get(str(item), "gray") for item in counts.index],
            edgecolor="black",
        )
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{int(height)}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        ax.set_title("Распределение ориентаций изображений")
        ax.set_xlabel("Ориентация")
        ax.set_ylabel("Количество изображений")
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(save_path, dpi=300)
    finally:
        plt.close(fig)


def save_results(df: pd.DataFrame, file_path: str | Path) -> None:
    """Сохранить результаты в UTF-8 CSV с разделителем ';'."""
    df.to_csv(file_path, sep=";", index=False, encoding="utf-8")


def display_statistics(df: pd.DataFrame) -> None:
    """Напечатать число изображений и процент каждой ориентации."""
    print(f"Всего изображений: {len(df)}")
    counts = df["orientation"].value_counts()
    print("\nРаспределение ориентаций:")
    for orientation, count in counts.items():
        percentage = float(count) / len(df) * 100
        print(f"  {orientation}: {count} ({percentage:.1f}%)")


def run_analysis(annotation_path: str | Path, output_dir: str | Path) -> pd.DataFrame:
    """Выполнить анализ и создать CSV и гистограмму."""
    print("Лабораторная работа №4, вариант 10")
    print("Анализ ориентации изображений")
    print("=" * 50)

    df = read_annotation(annotation_path)
    print(f"Загружено {len(df)} записей из файла аннотации")

    df = add_orientation_column(df)
    print(f"Обработано {len(df)} изображений")
    display_statistics(df)

    sorted_df = sort_by_orientation(df)
    print("\nДанные отсортированы по ориентации")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    chart_path = destination / "orientation_histogram.png"
    create_histogram(sorted_df, chart_path)
    print(f"Гистограмма сохранена как '{chart_path}'")

    csv_path = destination / "lab4_results.csv"
    save_results(sorted_df, csv_path)
    print(f"Результаты сохранены в '{csv_path}'")

    print("\nПервые 5 строк обработанных данных:")
    print(sorted_df.head())
    print("\nЛабораторная работа успешно завершена!")
    return sorted_df


def main(argv: Sequence[str] | None = None) -> int:
    """Точка входа в программу; вернуть 0 при успехе и 1 при ошибке."""
    parser = argparse.ArgumentParser(description="Анализ ориентации изображений")
    parser.add_argument(
        "--annotation", type=Path, default=Path("annotation.csv"), help="CSV аннотаций"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("."), help="Каталог результатов"
    )
    args = parser.parse_args(argv)

    try:
        run_analysis(args.annotation, args.output_dir)
    except (OSError, ValueError, pd.errors.ParserError) as error:
        print(f"Произошла ошибка: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
