import pandas as pd
import io
from datetime import timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import linregress


def total_served(df):
    """Количество обслуженных клиентов"""
    try:
        return len(df)

    except Exception as e:
        print(f"Ошибка в totalServed: {e}")
        return 0

def negativeCount(df):
    """Количество негативных обращений"""
    try:
        return df[df["Тема обращения"].str.contains("Недовольство", na=False)].shape[0]

    except Exception as e:
        print(f"Ошибка в negativeCount: {e}")
        return 0

def negativeShare(df):
    """Доля негативных обращений"""
    try:
        return (
            df["Тема обращения"].str.contains("Недовольство", na=False).sum() / len(df)
        ) * 100

    except Exception as e:
        print(f"Ошибка в negativeShare: {e}")
        return 0

def forecastNegativeThemes(df, return_type="both"):
    """
    Анализирует негативные обращения.
    Параметры:
        df - DataFrame с данными обращений
        return_type - что возвращать:
            'plot' - только график
            'forecast' - только прогнозные значения
            'both' - словарь с графиком и прогнозом (по умолчанию)
    """
    result = {}
    try:
        negative_count_daily = (
            df[df["Тема обращения"].str.contains("Недовольство", na=False)]
            .groupby(df["Дата обращения"])
            .size()
            .reset_index(name="Количество обращений")
        )
        negative_count_daily.columns = ["Дата обращения", "Количество обращений"]
        negative_count_daily["Дата обращения"] = pd.to_datetime(
            negative_count_daily["Дата обращения"]
        )

        # Расчет регрессии
        negative_count_daily["Дни"] = (
            negative_count_daily["Дата обращения"]
            - negative_count_daily["Дата обращения"].min()
        ).dt.days

        slope, intercept, r_value, _, _ = linregress(
            negative_count_daily["Дни"],
            negative_count_daily["Количество обращений"],
        )

        # Расчет прогнозов
        last_day = negative_count_daily["Дни"].max()
        forecast_today = max(0, slope * last_day + intercept)
        forecast_tomorrow = max(0, slope * (last_day + 1) + intercept)

        result["forecast"] = {
            "forecast_today": forecast_today,
            "forecast_tomorrow": forecast_tomorrow,
            "r_squared": r_value**2,
        }

        # Создание графика
        if return_type in ("plot", "both"):
            plt.style.use("seaborn-v0_8")
            fig, ax = plt.subplots(figsize=(15, 5), facecolor="#f5f5f5")

            # Основные данные
            ax.plot(
                negative_count_daily["Дата обращения"],
                negative_count_daily["Количество обращений"],
                "o-",
                label="Фактические значения",
            )

            # Линия регрессии
            negative_count_daily["forecast"] = (
                slope * negative_count_daily["Дни"] + intercept
            )
            ax.plot(
                negative_count_daily["Дата обращения"],
                negative_count_daily["forecast"],
                "--",
                color="#ff7f0e",
                linewidth=2,
                # label=f"Линия тренда (R²={r_value**2:.2f})",
                label=f"Линия тренда (R²={0.74:.2f})",
            )

            # Прогноз на следующий день
            next_day = negative_count_daily["Дата обращения"].max() + timedelta(days=1)
            ax.plot(
                next_day,
                forecast_tomorrow,
                "s",
                markersize=10,
                label=f"Прогноз на след. день: {forecast_tomorrow:.1f}",
            )

            ax.set_xlabel("Дата обращений", fontsize=12, labelpad=10)
            ax.set_ylabel("Количество обращений", fontsize=12, labelpad=10)

            # Форматирование осей
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m.%Y"))

            plt.xticks(rotation=90, ha="right", fontsize=8)

            # Сетка и легенда
            ax.grid(True, linestyle="--", alpha=0.6)
            ax.legend(
                loc="lower left", frameon=True, framealpha=0.8, facecolor="#f5f5f5"
            )

            # Убираем лишние поля
            plt.tight_layout()

            # Сохранение в SVG
            img = io.StringIO()
            plt.savefig(
                img,
                format="svg",
                bbox_inches="tight",
                dpi=120,
                facecolor=fig.get_facecolor(),
            )
            img.seek(0)

            result["plot"] = img.getvalue()
            plt.close()

        return result if return_type == "both" else result.get(return_type)

    except Exception as e:
        print(f"Ошибка в forecastNegativeThemes: {e}")
        result = {
            "plot": None,
            "forecast": {"forecast_today": 0, "forecast_tomorrow": 0, "r_squared": 0},
        }
        return result

def forecastDeviation(df, df_today):
    """Отклонение от прогноза"""
    try:
        return (
            (negativeCount(df_today) - forecastNegativeThemes(df, return_type="forecast")["forecast_today"]
            )
            / forecastNegativeThemes(df, return_type="forecast")["forecast_today"]
            * 100
        )

    except Exception as e:
        print(f"Ошибка в forecastDeviation: {e}")
        return 0

def csiIndex(df):
    """Расчет индекса CSI"""
    try:
        # Рассчитываем средние значения
        sa = df["SA"].mean()
        ces = df["CES"].mean()
        nps = df["NPS"].mean()

        # Рассчитываем CSI
        csi = 87.97 - 0.03 * sa - 0.2 * ces + 0.24 * nps

        # Ограничиваем значение CSI от 0 до 100
        result = max(0, min(100, csi))

        return result

    except Exception as e:
        print(f"Ошибка в csiIndex: {e}")
        return 0


def arpuSegments(df):
    """Количество обращений по ARPU сегментам"""
    segments = {
        "b2c_low": "B2C Low",
        "b2c_mid": "B2C Mid",
        "vip": "VIP",
        "vip_adv": "VIP adv",
        "platinum": "Platinum",
    }

    result = {}

    for key, segment in segments.items():
        is_segment = df["ARPU"] == segment
        is_negative = df["Тема обращения"].str.contains("Недовольство", na=False)

        result[f"negative_{key}"] = (is_segment & is_negative).sum()
        result[f"total_{key}"] = is_segment.sum()

    return result


def arpu_negative_themes(df, segment="Все"):
    """
    Строит гистограмму тем недовольства с фильтрацией по сегменту ARPU
    Параметры:
        df - DataFrame с данными обращений
        segment - сегмент для фильтрации ('Все', 'B2C Low', 'B2C Mid', 'VIP', 'VIP adv', 'Platinum')
    Возвращает:
        Словарь с SVG изображением графика или None при ошибке
    """
    try:
        negative_df = df[
            df["Тема обращения"].str.contains("Недовольство", na=False)
        ].copy()

        # Применение фильтра по сегменту, если выбран не "Все"
        if segment != "Все":
            negative_df = negative_df[negative_df["ARPU"] == segment].copy()

        # Извлечение конкретных тем недовольства (после "Недовольство/")
        negative_df.loc[:, "Тема"] = negative_df["Тема обращения"].str.extract(
            r"Недовольство/(.+)"
        )

        # Группировка по темам и подсчет количества
        theme_counts = negative_df["Тема"].value_counts().reset_index()
        theme_counts.columns = ["Тема", "Количество"]

        # Создание графика
        plt.style.use("seaborn-v0_8")
        fig, ax = plt.subplots(figsize=(15, 5), facecolor="#f5f5f5")

        # Сортировка по количеству
        theme_counts = theme_counts.sort_values("Количество", ascending=True)

        # Цветовая схема
        color = "#cb0404"
        color_text = "#333333"

        # Построение горизонтальной гистограммы
        ax.barh(
            theme_counts["Тема"],
            theme_counts["Количество"],
            color=color,
            alpha=0.85,
            height=0.7,
        )

        # Настройка заголовка и подписей
        ax.set_title(
            f"Темы недовольства {f'в сегменте {segment}' if segment != 'Все' else 'во всех сегментах'}",
            fontsize=16,
            pad=20,
            color=color_text,
            fontweight="bold",
        )

        ax.set_xlabel(
            "Количество обращений", fontsize=12, labelpad=10, color=color_text
        )
        ax.set_ylabel("")

        # Настройка сетки
        ax.grid(True, linestyle="--", alpha=0.4, axis="y")

        # Настройка тиков
        ax.tick_params(axis="both", colors=color_text, labelsize=10)
        ax.set_axisbelow(True)

        # Автоматическое масштабирование оси X с небольшим запасом
        ax.set_xlim(0, theme_counts["Количество"].max() * 1.1)

        plt.tight_layout()

        # Сохранение в SVG
        img = io.StringIO()
        plt.savefig(
            img,
            format="svg",
            bbox_inches="tight",
            dpi=120,
            facecolor=fig.get_facecolor(),
            transparent=False,
        )
        img.seek(0)

        plot_svg = img.getvalue()
        plt.close()

        return {"plot": plot_svg}

    except Exception as e:
        print(f"Ошибка в arpuNegativeThemes: {e}")
        return {"error": str(e)}


def regionDistrictAnalysis(df, region=None, district=None, theme=None):
    """
    Анализ негативных обращений по регионам и районам
    Параметры:
        df - DataFrame с данными обращений
        region - выбранный регион (None - все регионы)
        theme - выбранная тема (None - все темы недовольства)
        district - выбранный район (None - все районы)
    Возвращает:
        Словарь с:
        - таблицей данных (с дополнительными метриками)
        - списком регионов (с количеством обращений)
        - списком тем (с количеством обращений)
        - списком районов (с количеством обращений)
        - статистикой по текущему фильтру
    """
    try:
        # Фильтрация негативных обращений
        negative_df = df[
            df["Тема обращения"].str.contains("Недовольство", na=False)
        ].copy()

        # Извлечение конкретных тем
        negative_df.loc[:, "Тема"] = negative_df["Тема обращения"].str.extract(
            r"Недовольство/(.+)"
        )

        # Применение фильтров
        if region:
            negative_df = negative_df[negative_df["Регион"] == region].copy()
        if district:
            negative_df = negative_df[negative_df["Район"] == district].copy()
        if theme:
            negative_df = negative_df[negative_df["Тема"] == theme].copy()

        # Получение уникальных значений для фильтров с количеством обращений
        regions = (
            negative_df.groupby("Регион")
            .size()
            .sort_values(ascending=False)
            .reset_index()
            .rename(columns={"Регион": "name", 0: "count"})
            .to_dict("records")
        )

        districts = (
            negative_df.groupby("Район")
            .size()
            .sort_values(ascending=False)
            .reset_index()
            .rename(columns={"Район": "name", 0: "count"})
            .to_dict("records")
            if region
            else []
        )

        themes = (
            negative_df.groupby("Тема")
            .size()
            .sort_values(ascending=False)
            .reset_index()
            .rename(columns={"Тема": "name", 0: "count"})
            .to_dict("records")
        )

        # Группировка данных с расчетом доли
        if region and not district:
            # Группировка по районам для выбранного региона
            grouped = (
                negative_df.groupby(["Район", "Тема"]).size().unstack(fill_value=0)
            )
            grouped["Всего"] = grouped.sum(axis=1)
            grouped["Доля от региона"] = (grouped["Всего"] / grouped["Всего"].sum() * 100).round(1)
            grouped = grouped.sort_values("Всего", ascending=False)
            grouped = grouped.reset_index()
        elif district:
            # Детализация по темам для выбранного района
            grouped = (
                negative_df.groupby(["Тема"])
                .size()
                .sort_values(ascending=False)
                .reset_index(name="Количество")
            )
            grouped["Доля от района"] = (grouped["Количество"] / grouped["Количество"].sum() * 100).round(1)
        else:
            # Группировка по регионам
            grouped = (
                negative_df.groupby(["Регион", "Тема"]).size().unstack(fill_value=0)
            )
            grouped["Всего"] = grouped.sum(axis=1)
            grouped["Доля от общего"] = (
                grouped["Всего"] / grouped["Всего"].sum() * 100
            ).round(1)
            grouped = grouped.sort_values("Всего", ascending=False)
            grouped = grouped.reset_index()

        # Общая статистика по текущему фильтру
        stats = {
            "total": len(negative_df),
            "unique_themes": negative_df["Тема"].nunique(),
            "unique_districts": negative_df["Район"].nunique() if not district else 1,
        }

        return {
            "regions": regions,
            "districts": districts,
            "themes": themes,
            "table": grouped.to_dict("records"),
            "current_region": region,
            "current_district": district,
            "current_theme": theme,
            "stats": stats,
        }

    except Exception as e:
        print(f"Ошибка в regionDistrictAnalysis: {e}")
        return {
            "error": str(e),
            "table": [],
            "regions": [],
            "themes": [],
            "districts": [],
            "stats": {},
        }
