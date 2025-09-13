from abc import ABC, abstractmethod
from .surface import Surface


class Construction:#(ABC):
    # @abstractmethod
    def construct(self, surface: Surface): pass
