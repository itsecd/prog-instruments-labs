from abc import ABC, abstractmethod

# class Construction:#(ABC):
#     # @abstractmethod
#     def construct(self, canvas: Canvas): pass

class Construction:
    pass


class Context(Construction):
    pass


class Action(Construction):#(ABC):
    # @abstractmethod
    def construct(self, context: Context): pass
