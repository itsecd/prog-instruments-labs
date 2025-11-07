import numpy as np
import pandas as pd
import pytest

from process_df import *


def test_add_columns(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Test adding image dimension columns
    :param monkeypatch: pytest fixture for modifying objects
    """
    df: pd.DataFrame = pd.DataFrame( { 'abs.path': ['img1.jpg'],
                                      'rel_path': ['rel1'] } )
    
    def mock_imread(path: str) -> np.ndarray:
        return np.ones((100, 150, 3))
    
    monkeypatch.setattr('process_df.cv2.imread', mock_imread)
    result: pd.DataFrame = add_columns(df)
    assert 'Height' in result.columns
    assert result['Height'].iloc[0] == 100
    assert result['Width'].iloc[0] == 150
    assert result['Depth'].iloc[0] == 3


def test_sorted_df_filtering() -> None:
    """
    Test filtering by height and width
    """
    df: pd.DataFrame = pd.DataFrame( { 'Height': [500, 1200, 800], 
                                      'Width': [600, 900, 700] } )
    
    result: pd.DataFrame = sorted_df(df, 1000, 1000)
    assert len(result) == 2
    assert all(result['Height'] < 1000)
    assert all(result['Width'] < 1000)


def test_add_area_column() -> None:
    """
    Test area calculation
    """
    df: pd.DataFrame = pd.DataFrame( { 'Height': [100], 
                                      'Width': [150] } )
    
    result: pd.DataFrame = add_area_column(df)
    assert 'Area' in result.columns
    assert result['Area'].iloc[0] == 15000


def test_sort_areas_ascending() -> None:
    """
    Test sorting by area in ascending order
    """
    df: pd.DataFrame = pd.DataFrame( { 'Area': [300, 100, 200] } )

    result: pd.DataFrame = sort_areas(df)
    assert list(result['Area']) == [100, 200, 300]


def test_add_columns_with_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Test handling invalid images
    :param monkeypatch: pytest fixture for modifying objects
    """
    df: pd.DataFrame = pd.DataFrame( { 'abs_path': ['bad.jpg'], 
                                      'rel_path': ['rel1'] } )
    
    def mock_imread(path: str) -> None:
        return None
    
    monkeypatch.setattr('process_df.cv2.imread', mock_imread)
    result: pd.DataFrame = add_columns(df)
    assert result['Height'].iloc[0] == 0
    assert result['Width'].iloc[0] == 0
    assert result['Depth'].iloc[0] == 0


def test_area_edge_cases() -> None:
    """
    Test area calculation with edge cases
    """
    df: pd.DataFrame = pd.DataFrame( { 'Height': [0, -5], 
                                      'Width': [10, 20] } )
    
    result: pd.DataFrame = add_area_column(df)
    assert result['Area'].iloc[0] == 0
    assert result['Area'].iloc[1] == -100


def test_empty_dataframe() -> None:
    """
    Test functions with empty dataframe
    """
    empty_df: pd.DataFrame = pd.DataFrame(columns = ['Height', 'Width'])

    result: pd.DataFrame = add_area_column(empty_df)
    assert 'Area' in result.columns
    assert len(result) == 0