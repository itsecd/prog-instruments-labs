from abc import ABC, abstractmethod


class Construction:
    "Base class for construction"
    pass

class Context(Construction):
    """A construction for storing data (for example for canvas)"""
    pass

class Action(Construction, ABC):
    """A construction for performing actions (such as drawing)"""
    @abstractmethod
    def construct(self, context: Context):
        """Method for drawing a shape inside a matrix

        Args:
            context (Context): the context where the drawing will take place
        """
        pass