import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from process_hist import make_hist


def test_make_hist() -> None:
    """
    Test histogram creation
    """
    df: pd.DataFrame = pd.DataFrame( { 'Area': [1000, 2000, 3000] } )
    fig: plt.Figure = make_hist(df)

    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_hist_with_nan_values() -> None:
    """
    Test histogram with NaN values
    """
    df: pd.DataFrame = pd.DataFrame( { 'Area': [1000, np.nan, 2000] } )
    fig: plt.Figure = make_hist(df)

    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def make_hist_empty_data() -> None:
    """
    Test histogram with empty dataframe
    """
    df: pd.DataFrame = pd.DataFrame(columns = ['Area'])
    fig: plt.Figure = make_hist(df)

    assert isinstance(fig, plt.Figure)
    plt.close(fig)