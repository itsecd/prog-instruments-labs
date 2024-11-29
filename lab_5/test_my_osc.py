import pytest
import numpy as np
import random
from unittest.mock import MagicMock
from work_with_osc import DataOsc
import Aegis_osc


@pytest.fixture
def mock_file_osc():
    """
    Мокирует класс File_osc и его методы для тестирования
    """
    file_osc_mock = MagicMock()
    file_osc_mock.m_sdoHdr.NumOSC = 2  # Количество осциллограмм в файле
    file_osc_mock.getDotsOSC.return_value = [
        [1, 2, 3, 4, 5],  # Первая осциллограмма
        [6, 7, 8, 9, 10]  # Вторая осциллограмма
    ]
    return file_osc_mock


@pytest.fixture
def moc_rand_int():
    rand_mock = MagicMock()
    rand_mock.randint.return_value = 20
    return rand_mock


def test_create_datasets_with_osc(mock_file_osc, monkeypatch):
    """
    Тестирование создания наборов данных с осциллограммами и категориями.
    """
    # Подменяем класс File_osc в модуле Aegis_osc на mock_file_osc
    monkeypatch.setattr(Aegis_osc, 'File_osc', lambda x: mock_file_osc)

    # Список "файлов" (фиктивные имена, потому что мы используем моки)
    list_osc_files = ["file1.osc", "file2.osc"]
    csv_categories = ["category1", "category2"]

    # Вызов тестируемого метода
    data_oscs, categories = DataOsc.create_datasets_with_osc(list_osc_files, csv_categories)

    # Проверяем, что было возвращено 4 осциллограммы (по 2 из каждого "файла")
    assert data_oscs == [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
    assert categories == ["category1", "category1", "category2", "category2"]


def test_augmentation_on_time_cycle(monkeypatch):
    """
    Тестирование аугментации циклическим сдвигом
    """

    original_signal = [range(0, 2000)]
    augmented_signal = DataOsc.augmentation_on_time_cycle(original_signal)

    assert len(augmented_signal[0]) == 2000
    assert augmented_signal[0][0] != original_signal[0][0]  # Сигнал должен быть изменён


def test_fill_dataset_for_nulls():
    """
    Тестирование заполнения массива нулями до нужной длины
    """
    signal = [1, 2, 3]
    target_len = 5
    filled_signal = DataOsc.fill_dataset_for_nulls(signal, target_len)

    assert len(filled_signal) == target_len
    assert filled_signal == [1, 2, 3, 0, 0]


def test_fill_dataset_for_normal_rule_fft():
    """
    Тестирование дополнения данных до нужной длины с помощью FFT
    """
    signal = [1, 2, 3]
    target_len = 6
    filled_signal = DataOsc.fill_dataset_for_normal_rule_fft(signal, target_len)

    assert len(filled_signal) == target_len
    assert isinstance(filled_signal, np.ndarray)


def test_get_math_features():
    """
    Тестирование получения математических характеристик сигнала
    """
    signal = [1, 2, 3, 4, 5]
    features = DataOsc.get_math_features(signal)

    assert features["mean"] == 3
    assert features["std_dev"] == pytest.approx(1.414, 0.01)
    assert features["min_val"] == 1
    assert features["max_val"] == 5


@pytest.mark.parametrize(
    "signal",
    [
        list(range(random.randint(-1000, 1000), random.randint(-1000, 1000))),
        list(range(random.randint(-1000, 1000), random.randint(-1000, 1000))),
        list(range(random.randint(-1000, 1000), random.randint(-1000, 1000))),
        list(range(random.randint(-1000, 1000), random.randint(-1000, 1000))),
    ]
)
def test_get_spectr(signal):
    """
    Тестирование получения спектра сигнала
    """
    signal = [1, 2, 3, 4, 5]
    spectr = DataOsc.get_spectr(signal.copy())
    val = 1
    while val < len(signal):
        val <<= 1

    assert len(spectr) == val
