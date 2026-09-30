import numpy as np
import pytest
from numpy.typing import NDArray


@pytest.fixture
def samplerate() -> int:
    return 8000


@pytest.fixture
def audio(samplerate: int) -> NDArray[np.float64]:
    return np.linspace(-1.0, 1.0, num=2 * samplerate, dtype=np.float64)
