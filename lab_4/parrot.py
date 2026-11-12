from abc import ABC, abstractmethod


class Parrot(ABC):
    """Базовый класс для всех попугаев."""

    @abstractmethod
    def speed(self):
        pass

    @abstractmethod
    def cry(self):
        pass


class EuropeanParrot(Parrot):
    def speed(self):
        return 12.0

    def cry(self):
        return "Sqoork!"


class AfricanParrot(Parrot):
    def __init__(self, number_of_coconuts):
        self._number_of_coconuts = number_of_coconuts

    def speed(self):
        load_factor = 9.0
        base_speed = 12.0
        return max(0, base_speed - load_factor * self._number_of_coconuts)

    def cry(self):
        return "Sqaark!"


class NorwegianBlueParrot(Parrot):
    def __init__(self, voltage, nailed):
        self._voltage = voltage
        self._nailed = nailed

    def speed(self):
        if self._nailed:
            return 0.0
        return min(24.0, self._voltage * 12.0)

    def cry(self):
        return "Bzzzzzz" if self._voltage > 0 else "..."