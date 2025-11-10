import matplotlib.pyplot as plt
import pandas as pd

from process_df import *


def make_hist(data_frame: pd.DataFrame) -> plt.Figure:
    """
    Create histogram of area of images
    :param data_frame: DataFrame
    """
    if data_frame.empty or 'Area' not in data_frame.columns:
        fig, ax = plt.subplots()
        return fig
    
    fig, ax = plt.subplots()
    data_frame['Area'].dropna().hist(ax = ax)
    
    ax.set_title('Histogram')
    ax.set_xlabel('Area')
    ax.set_ylabel('Area count')
    
    return fig