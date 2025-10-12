import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch


def friendly_dtype(dtype):
    if np.issubdtype(dtype, np.number):
        return "Integer" if np.issubdtype(dtype, np.integer) else "Float"
    elif np.issubdtype(dtype, np.datetime64):
        return "Datetime"
    else:
        return "Categorical"


def numeric_summary(data, fields):
    summary_list = []
    for col in fields:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers_count = \
        data[(data[col] < Q1 - 1.5 * IQR) | (data[col] > Q3 + 1.5 * IQR)].shape[
            0]
        desc = data[col].describe()
        summary_list.append({
            "Field": col,
            "Type": friendly_dtype(data[col].dtype),
            "Mean": round(desc["mean"], 2),
            "Std": round(desc["std"], 2),
            "Min": round(desc["min"], 2),
            "25%": round(Q1, 2),
            "50%": round(desc["50%"], 2),
            "75%": round(Q3, 2),
            "Max": round(desc["max"], 2),
            "Skewness": round(data[col].skew(), 2),
            "Kurtosis": round(data[col].kurtosis(), 2),
            "Outliers": outliers_count,
            "Missing": data[col].isnull().sum()
        })
    return pd.DataFrame(summary_list)


def categorical_summary(data, fields):
    summary_list = []
    for col in fields:
        top_val = data[col].mode()[0] if not data[col].mode().empty else "N/A"
        top_freq = data[col].value_counts().iloc[0] if len(
            data[col].value_counts()) > 0 else 0
        summary_list.append({
            "Field": col,
            "Type": friendly_dtype(data[col].dtype),
            "Unique": data[col].nunique(),
            "Most Frequent": str(top_val),
            "Frequency": top_freq,
            "Missing": data[col].isnull().sum()
        })
    return pd.DataFrame(summary_list)


@pytest.fixture
def sample_data():
    """Фикстура с тестовыми данными"""
    return pd.DataFrame({
        'numeric_col': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'categorical_col': ['A', 'B', 'A', 'C', 'B', 'A', 'A', 'C', 'B', 'A'],
        'float_col': [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.0],
        'mixed_col': ['X', 'Y', 'X', 'Z', 'Y', 'X', 'X', 'Z', 'Y', 'X']
    })


@pytest.fixture
def sample_data_with_missing():
    """Фикстура с тестовыми данными с пропущенными значениями"""
    return pd.DataFrame({
        'numeric_col': [1, 2, None, 4, 5, None, 7, 8, 9, 10],
        'categorical_col': ['A', 'B', None, 'C', 'B', 'A', None, 'C', 'B', 'A'],
    })


def test_friendly_dtype_basic():
    """Тест базовой функциональности friendly_dtype"""
    assert friendly_dtype(np.dtype('int64')) == "Integer"
    assert friendly_dtype(np.dtype('float64')) == "Float"
    assert friendly_dtype(np.dtype('object')) == "Categorical"


def test_numeric_summary_structure(sample_data):
    """Тест структуры результата numeric_summary"""
    result = numeric_summary(sample_data, ['numeric_col'])

    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    expected_columns = ["Field", "Type", "Mean", "Std", "Min", "25%", "50%",
                        "75%", "Max",
                        "Skewness", "Kurtosis", "Outliers", "Missing"]
    assert all(col in result.columns for col in expected_columns)


def test_categorical_summary_structure(sample_data):
    """Тест структуры результата categorical_summary"""
    result = categorical_summary(sample_data, ['categorical_col'])

    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    expected_columns = ["Field", "Type", "Unique", "Most Frequent", "Frequency",
                        "Missing"]
    assert all(col in result.columns for col in expected_columns)


def test_numeric_summary_with_missing_values(sample_data_with_missing):
    """Тест numeric_summary с пропущенными значениями"""
    result = numeric_summary(sample_data_with_missing, ['numeric_col'])

    assert result.iloc[0]['Missing'] == 2  # 2 пропущенных значения
    assert result.iloc[0]['Field'] == 'numeric_col'


def test_categorical_summary_with_missing_values(sample_data_with_missing):
    """Тест categorical_summary с пропущенными значениями"""
    result = categorical_summary(sample_data_with_missing, ['categorical_col'])

    assert result.iloc[0]['Missing'] == 2  # 2 пропущенных значения
    assert result.iloc[0]['Field'] == 'categorical_col'


@pytest.mark.parametrize("dtype,expected", [
    (np.dtype('int64'), "Integer"),
    (np.dtype('float64'), "Float"),
    (np.dtype('object'), "Categorical"),
    (np.dtype('datetime64[ns]'), "Datetime"),
    (np.dtype('bool'), "Categorical")
])
def test_friendly_dtype_parametrized(dtype, expected):
    """Параметризованный тест для friendly_dtype"""
    assert friendly_dtype(dtype) == expected


def test_numeric_summary_with_mock():
    """Тест numeric_summary с использованием моков"""
    test_df = pd.DataFrame({'test_col': [1.0, 2.0, 3.0]})

    with patch.object(pd.Series, 'quantile') as mock_quantile, \
            patch.object(pd.Series, 'describe') as mock_describe, \
            patch.object(pd.Series, 'skew') as mock_skew, \
            patch.object(pd.Series, 'kurtosis') as mock_kurtosis:
        mock_quantile.side_effect = [2.0, 8.0]  # Q1, Q3
        mock_describe.return_value = pd.Series({
            'mean': 5.0, 'std': 2.0, 'min': 1.0, '50%': 5.0, 'max': 9.0
        })
        mock_skew.return_value = 0.0
        mock_kurtosis.return_value = -1.2

        result = numeric_summary(test_df, ['test_col'])

    assert not result.empty
    assert result.iloc[0]['Field'] == 'test_col'
    assert result.iloc[0]['Type'] == 'Float'
    assert result.iloc[0]['Mean'] == 5.0
    assert result.iloc[0]['Skewness'] == 0.0
    assert result.iloc[0]['Kurtosis'] == -1.2


def test_mixed_data_types_summary():
    """Тест функций summary с различными типами данных"""
    mixed_data = pd.DataFrame({
        'integers': [1, 2, 3, 4, 5],
        'floats': [1.1, 2.2, 3.3, 4.4, 5.5],
        'strings': ['a', 'b', 'c', 'd', 'e'],
        'booleans': [True, False, True, False, True]
    })


    numeric_result = numeric_summary(mixed_data, ['integers', 'floats'])
    assert len(numeric_result) == 2
    assert set(numeric_result['Field'].values) == {'integers', 'floats'}

    categorical_result = categorical_summary(mixed_data,
                                             ['strings', 'booleans'])
    assert len(categorical_result) == 2
    assert set(categorical_result['Field'].values) == {'strings', 'booleans'}


def test_empty_dataframe():
    """Тест обработки пустого DataFrame"""
    empty_df = pd.DataFrame()

    numeric_result = numeric_summary(empty_df, [])
    assert numeric_result.empty

    categorical_result = categorical_summary(empty_df, [])
    assert categorical_result.empty


def test_summary_functions_with_single_value():
    """Тест функций summary с данными, содержащими только одно значение"""
    single_value_data = pd.DataFrame({
        'single_num': [5] * 10,
        'single_cat': ['A'] * 10
    })

    numeric_result = numeric_summary(single_value_data, ['single_num'])
    categorical_result = categorical_summary(single_value_data, ['single_cat'])

    assert numeric_result.iloc[0][
               'Std'] == 0
    assert categorical_result.iloc[0][
               'Unique'] == 1