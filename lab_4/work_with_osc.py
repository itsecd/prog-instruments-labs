import os
import random
import sys
from collections import Counter

import numpy as np
from numpy.fft import ifft, fft

from Fourier import Fourier

# Теперь импортируем модуль для работы с osc
module_path = os.path.abspath("cpp/build/Debug")
sys.path.append(module_path)
try:
    import Aegis_osc
except ImportError as e:
    raise "Не удалось импортировать модуль для работы с осциллограммами!"


class DataOsc:
    def __init__(self, file_name: str):
        if len(file_name) == 0:
            return

    @staticmethod
    def create_datasets_with_osc(list_osc: list
                                 , csv_categories: list
                                 , augment: bool = False) -> (list[list], list[str]):
        """"""
        data_oscs = []
        categories = []
        # цикл для заполнения списков значениями осциллограмм и категорий
        for ind, file_name in enumerate(list_osc):
            osc_file = Aegis_osc.File_osc(file_name)
            num_osc = osc_file.m_sdoHdr.NumOSC
            categories.extend([csv_categories[ind] for _ in range(num_osc)])
            list_osc_data = osc_file.getDotsOSC(0, num_osc)
            data_oscs.extend([list_osc_data[i] for i in range(num_osc)])

        # цикл для подсчёта кол-ва данных, относящихся к одной категории
        counts = Counter(categories)
        delete_el: list[list] = []
        add_el: list[list] = []
        for name in counts:
            part_of_all = counts[name] / len(categories)
            # если данных одной категории слишком много,
            # то запоминаем какой класс нужно уменьшить
            if part_of_all >= 0.4:
                delete_el.append([name, len(categories) * 0.2])
                continue

            if augment and part_of_all < 0.2:
                add_el.append([name, len(categories) * 0.2])

        # Увеличиваем выборку путём аугментации, если необходимо
        if len(add_el) != 0:
            length_cat = len(categories)
            for a in add_el:
                count_add = a[1] - counts[a[0]]
                for i in range(length_cat):
                    if count_add > 1 and categories[i] == a[0]:
                        new_osc = DataOsc.augmentation_on_time_cycle(data_oscs[i])
                        data_oscs.append(new_osc)
                        categories.append(a[0])
                        count_add -= 1
                    elif count_add <= 0:
                        break

        # уменьшаем выборку, если это необходимо
        if len(delete_el) != 0:
            for d in delete_el:
                for i in range(len(categories) - 1, -1, -1):
                    if d[1] <= counts[d[0]] and categories[i] == d[0]:
                        categories.pop(i)
                        data_oscs.pop(i)
                        counts[d[0]] -= 1
                    elif d[1] > counts[d[0]]:
                        break

        counts = Counter(categories)

        return data_oscs, categories

    @staticmethod
    def augmentation_on_time_cycle(list_osc: list) -> list:
        augm_list = []
        for osc in list_osc:
            augm_list.append(np.roll(osc, random.randint(5, 50)))
        return  augm_list

    @staticmethod
    def augmentation_on_time(list_osc: list) -> list:
        augm_list = []
        for osc in list_osc:
            shift = random.randint(5, 50)
            augm_list.append(np.roll(osc, shift))
            augm_list[-1][:shift] = 0
        return augm_list

    @staticmethod
    def fill_dataset_for_nulls(signal: list, target_len: int) -> list:
        """метод, добавляющий нули в конец датасета до определённой длины"""
        if len(signal) >= target_len:
            return signal[:target_len]

        fill_len = target_len - len(signal)
        signal.extend([0 for _ in range(fill_len)])
        return signal

    @staticmethod
    def fill_dataset_for_normal_rule_fft(signal: list, target_len: int) -> list | np.ndarray:
        """
        метод, добавляющий в конец сигнала значения, построенные на мат.статистике
        последних 30% значений оциллограммы, чтобы максимально
        не отличаться по внешнему виду от изначального сигнала
        """
        if len(signal) >= target_len:
            return signal[:target_len]

        current_length = len(signal)

        # Инициализация дополненного массива
        padded_signal = np.zeros(target_len)
        padded_signal[:current_length] = signal.copy()

        # Генерируем шум на основе статистики сигнала
        mean = np.mean(signal[round(len(signal) * 0.7):])
        std = np.std(signal[round(len(signal) * 0.7):])
        noise = np.random.normal(loc=mean, scale=std, size=target_len - current_length)

        # Добавляем шум к оставшейся части массива
        padded_signal[current_length:] = noise

        # Переходим в частотную область, чтобы сгладить переход
        spectrum_padded = fft(padded_signal)

        # Применяем обратное преобразование Фурье для получения дополненного сигнала
        from_spectr = np.real(ifft(spectrum_padded))
        from_spectr[:current_length] = signal
        return from_spectr

    @staticmethod
    def fill_dataset_for_normal_rule_fourier(signal: list, target_len: int) -> list:
        """
        метод, добавляющий в конец сигнала значения, построенные на мат.статистике
        последних 30% значений оциллограммы, чтобы максимально
        не отличаться по внешнему виду от изначального сигнала
        """
        if len(signal) >= target_len:
            return signal[:target_len]

        current_length = len(signal)

        # Инициализация дополненного массива
        padded_signal = np.zeros(target_len)
        padded_signal[:current_length] = signal.copy()

        # Генерируем шум на основе статистики сигнала
        mean = np.mean(signal[round(len(signal) * 0.7):])
        std = np.std(signal[round(len(signal) * 0.7):])
        noise = np.random.normal(loc=mean, scale=std, size=target_len - current_length)

        # Добавляем шум к оставшейся части массива
        padded_signal[current_length:] = noise

        # Переходим в частотную область, чтобы сгладить переход
        spectrum_padded = Fourier.four2(padded_signal)

        # Применяем обратное преобразование Фурье для получения дополненного сигнала
        from_spectr = Fourier.four2(spectrum_padded, d=1)
        from_spectr[:current_length] = signal
        return from_spectr

    @staticmethod
    def get_math_features(signal: list) -> dict:
        """метод считает характеристики мат. статистики для сигнала"""
        # Получаем значения из мат. статистики, которые могут помочь
        # в обучении нейронной сети
        features = {"mean": np.mean(signal), "std_dev": np.std(signal), "variance": np.var(signal),
                    "min_val": np.min(signal), "max_val": np.max(signal)}
        features["kurtosis"] = np.mean((signal - features["mean"]) ** 4) / (features["std_dev"] ** 4)
        features["energy"] = np.sum([i ** 2 for i in signal])
        return features

    @staticmethod
    def get_spectr(signal: list) -> list:
        """ получаю спектр из осциллограммы """
        return Fourier.four2(signal, d=-1)

    @staticmethod
    def __get_osc_from_spectr(signal: list, current_length: int) -> list:
        """ получаю осциллограмму из спектра """
        return Fourier.four2(signal, d=1)
