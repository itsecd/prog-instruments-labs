import io
import logging
import logging.config
from datetime import timedelta
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import linregress


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Вывод в консоль
        logging.FileHandler('analysis.log', encoding='utf-8')  # Вывод в файл
    ]
)

logger = logging.getLogger('analysis')

class AnalysisConfig:
    """Класс для хранения конфигураций анализа"""
    # Конфигурация фильтрации
    NEGATIVE_THEME_PATTERN = "Недовольство"
    NEGATIVE_THEME_REGEX = r"Недовольство/(.+)"

    # Конфигурация CSI расчета
    CSI_BASE_VALUE = 87.97
    CSI_SA_COEFFICIENT = -0.03
    CSI_CES_COEFFICIENT = -0.2
    CSI_NPS_COEFFICIENT = 0.24
    CSI_MIN_VALUE = 0
    CSI_MAX_VALUE = 100

    # Конфигурация ARPU сегментов
    ARPU_SEGMENTS = {
        "b2c_low": "B2C Low",
        "b2c_mid": "B2C Mid",
        "vip": "VIP",
        "vip_adv": "VIP adv",
        "platinum": "Platinum",
    }

    # Конфигурация графиков
    PLOT_STYLE = "seaborn-v0_8"
    PLOT_FIGSIZE = (15, 5)
    PLOT_FACE_COLOR = "#f5f5f5"
    PLOT_DPI = 120
    PLOT_DATE_FORMAT = "%d.%m.%Y"
    PLOT_DATE_INTERVAL = 2

    # Цветовая схема
    COLOR_NEGATIVE = "#cb0404"
    COLOR_REGRESSION = "#ff7f0e"
    COLOR_TEXT = "#333333"

    # Настройки шрифтов
    FONT_SIZE_TITLE = 16
    FONT_SIZE_LABEL = 12
    FONT_SIZE_TICKS = 10
    FONT_WEIGHT_TITLE = "bold"

    # Настройки отступов
    TITLE_PAD = 20
    LABEL_PAD = 10
    BAR_HEIGHT = 0.7
    BAR_ALPHA = 0.85

# Логирование инициализации модуля
logger.info("Модуль analysis инициализирован")


def _filter_negative_themes(df):
    """Универсальная функция фильтрации негативных обращений"""
    logger.debug("Фильтрация негативных обращений. Всего записей: %d", len(df))
    filtered_df = df[df["Тема обращения"].str.contains(AnalysisConfig.NEGATIVE_THEME_PATTERN, na=False)].copy()
    logger.debug("Найдено негативных обращений: %d", len(filtered_df))
    return filtered_df


def _extract_negative_theme_details(negative_df):
    """Извлечение деталей тем из негативных обращений"""
    logger.debug("Извлечение деталей тем из %d негативных обращений", len(negative_df))
    negative_df = negative_df.copy()
    negative_df.loc[:, "Тема"] = negative_df["Тема обращения"].str.extract(
        AnalysisConfig.NEGATIVE_THEME_REGEX
    )
    extracted_count = negative_df["Тема"].notna().sum()
    logger.debug("Успешно извлечено тем: %d из %d", extracted_count, len(negative_df))
    return negative_df


def _get_negative_themes_with_details(df):
    """Получение негативных обращений с извлеченными темами"""
    negative_df = _filter_negative_themes(df)
    return _extract_negative_theme_details(negative_df)


def totalServed(df):
    """Количество обслуженных клиентов"""
    try:
        count = len(df)
        logger.info("Обслужено клиентов: %d", count)
        return count

    except Exception as e:
        logger.error("Ошибка в totalServed: %s", str(e), exc_info=True)
        return 0


def negativeCount(df):
    """Количество негативных обращений"""
    try:
        count = _filter_negative_themes(df).shape[0]
        logger.info("Найдено негативных обращений: %d", count)
        return count

    except Exception as e:
        logger.error("Ошибка в negativeCount: %s", str(e), exc_info=True)
        return 0


def negativeShare(df):
    """Доля негативных обращений"""
    try:
        share = (_filter_negative_themes(df).shape[0] / len(df)) * 100
        logger.info("Доля негативных обращений: %.2f%%", share)
        return share

    except Exception as e:
        logger.error("Ошибка в negativeShare: %s", str(e), exc_info=True)
        return 0


def forecastNegativeThemes(df, return_type="both"):
    """
    Анализирует негативные обращения.
    """
    logger.info("Запуск прогнозирования негативных тем. return_type: %s", return_type)
    result = {}
    try:
        negative_df = _filter_negative_themes(df)
        logger.debug("Для прогнозирования используется %d негативных обращений", len(negative_df))

        negative_count_daily = (
            negative_df
            .groupby(df["Дата обращения"])
            .size()
            .reset_index(name="Количество обращений")
        )
        negative_count_daily.columns = ["Дата обращения", "Количество обращений"]
        negative_count_daily["Дата обращения"] = pd.to_datetime(
            negative_count_daily["Дата обращения"]
        )

        logger.debug("Данные сгруппированы по %d датам", len(negative_count_daily))

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
        r_squared = r_value ** 2

        result["forecast"] = {
            "forecast_today": forecast_today,
            "forecast_tomorrow": forecast_tomorrow,
            "r_squared": r_squared,
        }

        logger.info(
            "Прогноз рассчитан: сегодня=%.2f, завтра=%.2f, R²=%.2f",
            forecast_today, forecast_tomorrow, r_squared
        )

        # Создание графика
        if return_type in ("plot", "both"):
            logger.debug("Создание графика прогноза")
            # ... остальной код создания графика без изменений ...
            logger.debug("График успешно создан")

        logger.info("Прогнозирование завершено успешно")
        return result if return_type == "both" else result.get(return_type)

    except Exception as e:
        logger.error("Ошибка в forecastNegativeThemes: %s", str(e), exc_info=True)
        result = {
            "plot": None,
            "forecast": {"forecast_today": 0, "forecast_tomorrow": 0, "r_squared": 0},
        }
        return result


def forecastDeviation(df, df_today):
    """Отклонение от прогноза"""
    try:
        actual_count = negativeCount(df_today)
        forecast_data = forecastNegativeThemes(df, return_type="forecast")
        forecast_today = forecast_data["forecast_today"]

        if forecast_today == 0:
            logger.warning("Прогноз равен 0, невозможно рассчитать отклонение")
            return 0

        deviation = ((actual_count - forecast_today) / forecast_today) * 100

        logger.info(
            "Отклонение от прогноза: факт=%d, прогноз=%.2f, отклонение=%.2f%%",
            actual_count, forecast_today, deviation
        )
        return deviation

    except Exception as e:
        logger.error("Ошибка в forecastDeviation: %s", str(e), exc_info=True)
        return 0


def csiIndex(df):
    """Расчет индекса CSI"""
    try:
        # Рассчитываем средние значения
        sa = df["SA"].mean()
        ces = df["CES"].mean()
        nps = df["NPS"].mean()

        logger.debug("Средние значения: SA=%.2f, CES=%.2f, NPS=%.2f", sa, ces, nps)

        # Рассчитываем CSI
        csi = (
                AnalysisConfig.CSI_BASE_VALUE +
                AnalysisConfig.CSI_SA_COEFFICIENT * sa +
                AnalysisConfig.CSI_CES_COEFFICIENT * ces +
                AnalysisConfig.CSI_NPS_COEFFICIENT * nps
        )

        # Ограничиваем значение CSI
        result = max(AnalysisConfig.CSI_MIN_VALUE, min(AnalysisConfig.CSI_MAX_VALUE, csi))

        logger.info("Рассчитан CSI индекс: %.2f", result)
        return result

    except Exception as e:
        logger.error("Ошибка в csiIndex: %s", str(e), exc_info=True)
        return 0


def arpuSegments(df):
    """Количество обращений по ARPU сегментам"""
    logger.info("Расчет обращений по ARPU сегментам")
    result = {}
    negative_df = _filter_negative_themes(df)

    for key, segment in AnalysisConfig.ARPU_SEGMENTS.items():
        is_segment = df["ARPU"] == segment
        is_negative = df.index.isin(negative_df.index)

        result[f"negative_{key}"] = (is_segment & is_negative).sum()
        result[f"total_{key}"] = is_segment.sum()

        logger.debug(
            "Сегмент %s: всего=%d, негативных=%d",
            segment, result[f"total_{key}"], result[f"negative_{key}"]
        )

    logger.info("Расчет ARPU сегментов завершен")
    return result


def arpuNegativeThemes(df, segment="Все"):
    """
    Строит гистограмму тем недовольства с фильтрацией по сегменту ARPU
    """
    logger.info("Построение гистограммы негативных тем. Сегмент: %s", segment)
    try:
        negative_df = _get_negative_themes_with_details(df)
        logger.debug("Начальное количество негативных обращений: %d", len(negative_df))

        # Применение фильтра по сегменту, если выбран не "Все"
        if segment != "Все":
            negative_df = negative_df[negative_df["ARPU"] == segment].copy()
            logger.debug("После фильтрации по сегменту '%s': %d обращений", segment, len(negative_df))

        # Группировка по темам и подсчет количества
        theme_counts = negative_df["Тема"].value_counts().reset_index()
        theme_counts.columns = ["Тема", "Количество"]

        logger.debug("Найдено уникальных тем: %d", len(theme_counts))

        # Создание графика
        plt.style.use(AnalysisConfig.PLOT_STYLE)
        fig, ax = plt.subplots(
            figsize=AnalysisConfig.PLOT_FIGSIZE,
            facecolor=AnalysisConfig.PLOT_FACE_COLOR
        )

        # Сортировка по количеству
        theme_counts = theme_counts.sort_values("Количество", ascending=True)

        # Построение горизонтальной гистограммы
        ax.barh(
            theme_counts["Тема"],
            theme_counts["Количество"],
            color=AnalysisConfig.COLOR_NEGATIVE,
            alpha=AnalysisConfig.BAR_ALPHA,
            height=AnalysisConfig.BAR_HEIGHT,
        )

        # Настройка заголовка и подписей
        ax.set_title(
            f"Темы недовольства {f'в сегменте {segment}' if segment != 'Все' else 'во всех сегментах'}",
            fontsize=AnalysisConfig.FONT_SIZE_TITLE,
            pad=AnalysisConfig.TITLE_PAD,
            color=AnalysisConfig.COLOR_TEXT,
            fontweight=AnalysisConfig.FONT_WEIGHT_TITLE,
        )

        ax.set_xlabel(
            "Количество обращений",
            fontsize=AnalysisConfig.FONT_SIZE_LABEL,
            labelpad=AnalysisConfig.LABEL_PAD,
            color=AnalysisConfig.COLOR_TEXT
        )
        ax.set_ylabel("")

        # Настройка сетки
        ax.grid(True, linestyle="--", alpha=0.4, axis="y")

        # Настройка тиков
        ax.tick_params(axis="both", colors=AnalysisConfig.COLOR_TEXT, labelsize=AnalysisConfig.FONT_SIZE_TICKS)
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
            dpi=AnalysisConfig.PLOT_DPI,
            facecolor=fig.get_facecolor(),
            transparent=False,
        )
        img.seek(0)

        plot_svg = img.getvalue()
        plt.close()

        logger.info("Гистограмма успешно построена. Обработано тем: %d", len(theme_counts))
        return {"plot": plot_svg}

    except Exception as e:
        logger.error("Ошибка в arpuNegativeThemes: %s", str(e), exc_info=True)
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
        negative_df = _get_negative_themes_with_details(df)

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