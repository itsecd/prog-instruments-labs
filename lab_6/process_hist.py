import matplotlib.pyplot as plt
import pandas as pd

from process_df import *

def make_hist(data_frame: pd.DataFrame) -> None:
    """
    Create histogram of area of images
    :param data_frame: DataFrame
    """
    plt.figure()
    data_frame['Area'].dropna().hist()

    plt.title('Histogram')
    plt.xlabel('Area')
    plt.ylabel('Area count')
    plt.show()