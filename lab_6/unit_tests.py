import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from process_df import *
from process_hist import make_hist


def test_complete_workflow(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Test complete image processing workflow
    :param monkeypatch: pytest fixture for modifying objects
    """
    df: pd.DataFrame = pd.DataFrame( {
        'abs_path': ['img1.jpg', 'img2.jpg'],
        'rel_path': ['rel1', 'rel2']
    } )
    
    def mock_imread(path: str) -> np.ndarray:
        if 'img1' in path:
            return np.ones((800, 600, 3))
        else:
            return np.ones((1200, 900, 3))
    
    monkeypatch.setattr('process_df.cv2.imread', mock_imread)
    
    df_with_dims: pd.DataFrame = add_columns(df)
    filtered_df: pd.DataFrame = sorted_df(df_with_dims, 1000, 1000)
    df_with_area: pd.DataFrame = add_area_column(filtered_df)
    sorted_df_result: pd.DataFrame = sort_areas(df_with_area)
    fig: plt.Figure = make_hist(sorted_df_result)
    
    assert 'Area' in df_with_area.columns
    assert len(filtered_df) == 1
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_workflow_with_invalid_images(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Test workflow with invalid images
    :param monkeypatch: pytest fixture for modifying objects
    """
    df: pd.DataFrame = pd.DataFrame( {
        'abs_path': ['bad.jpg', 'good.jpg'],
        'rel_path': ['rel1', 'rel2']
    } )
    
    def mock_imread(path: str) -> np.ndarray | None:
        if 'bad' in path:
            return None
        else:
            return np.ones( (500, 500, 3) )
    
    monkeypatch.setattr('process_df.cv2.imread', mock_imread)
    
    df_with_dims: pd.DataFrame = add_columns(df)
    df_with_area: pd.DataFrame = add_area_column(df_with_dims)
    sorted_df_result: pd.DataFrame = sort_areas(df_with_area)
    fig: plt.Figure = make_hist(sorted_df_result)
    
    assert len(df_with_dims) == 2
    assert isinstance(fig, plt.Figure)
    plt.close(fig)