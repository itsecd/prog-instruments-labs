# Original code: Emily Bache, MIT License
# https://github.com/emilybache/Parrot-Refactoring-Kata
"""Parrot speed and cry, refactored with polymorphic subclasses."""
from abc import ABC, abstractmethod
from enum import Enum

BASE_SPEED = 12.0
LOAD_FACTOR = 9.0
MAX_SPEED = 24.0


class ParrotType(Enum):
    EUROPEAN = 1
    AFRICAN = 2
    NORWEGIAN_BLUE = 3


class Parrot(ABC):
    """Common interface for all kinds of parrots."""

    @abstractmethod
    def speed(self):
        """Return the flying speed of the parrot."""

    @abstractmethod
    def cry(self):
        """Return the sound the parrot makes."""


class EuropeanParrot(Parrot):

    def speed(self):
        return BASE_SPEED

    def cry(self):
        return "Sqoork!"


class AfricanParrot(Parrot):

    def __init__(self, number_of_coconuts):
        self._number_of_coconuts = number_of_coconuts

    def speed(self):
        load = LOAD_FACTOR * self._number_of_coconuts
        return max(0, BASE_SPEED - load)

    def cry(self):
        return "Sqaark!"


class NorwegianBlueParrot(Parrot):

    def __init__(self, voltage, nailed):
        self._voltage = voltage
        self._nailed = nailed

    def speed(self):
        if self._nailed:
            return 0
        return min(MAX_SPEED, self._voltage * BASE_SPEED)

    def cry(self):
        return "Bzzzzzz" if self._voltage > 0 else "..."


def create_parrot(type_of_parrot, number_of_coconuts, voltage, nailed):
    """Create a parrot of the given type (same arguments as the original)."""
    if type_of_parrot == ParrotType.EUROPEAN:
        return EuropeanParrot()
    if type_of_parrot == ParrotType.AFRICAN:
        return AfricanParrot(number_of_coconuts)
    if type_of_parrot == ParrotType.NORWEGIAN_BLUE:
        return NorwegianBlueParrot(voltage, nailed)
    raise ValueError(f"Unknown parrot type: {type_of_parrot}")
