import numpy as np
from matplotlib import pyplot as plt


def plot_results(x: np.ndarray, y: np.ndarray, title: str, xlabel: str,
                 ylabel: str, color: str) -> None:
    """
    Function for visualizing results.
    """
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(x, y, color=color)
    plt.grid()
    plt.show()
