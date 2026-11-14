# Original code: Emily Bache, MIT License
# https://github.com/emilybache/Parrot-Refactoring-Kata

from enum import Enum


class ParrotType(Enum):
    EUROPEAN = 1
    AFRICAN = 2
    NORWEGIAN_BLUE = 3


class Parrot:

    def __init__(self, type_of_parrot, number_of_coconuts, voltage, nailed):
        self._type = type_of_parrot
        self._number_of_coconuts = number_of_coconuts
        self._voltage = voltage
        self._nailed = nailed

    def speed(self):
        match self._type:
            case ParrotType.EUROPEAN:
                return self._european_speed()
            case ParrotType.AFRICAN:
                return self._african_speed()
            case ParrotType.NORWEGIAN_BLUE:
                return self._norwegian_blue_speed()
        raise ValueError("should be unreachable")

    def cry(self):
        match self._type:
            case ParrotType.EUROPEAN:
                return "Sqoork!"
            case ParrotType.AFRICAN:
                return "Sqaark!"
            case ParrotType.NORWEGIAN_BLUE:
                return self._norwegian_blue_cry()
        raise ValueError("should be unreachable")

    def _european_speed(self):
        return self._base_speed()

    def _african_speed(self):
        load = self._load_factor() * self._number_of_coconuts
        return max(0, self._base_speed() - load)

    def _norwegian_blue_speed(self):
        if self._nailed:
            return 0
        return self._compute_base_speed_for_voltage(self._voltage)

    def _norwegian_blue_cry(self):
        return "Bzzzzzz" if self._voltage > 0 else "..."

    def _compute_base_speed_for_voltage(self, voltage):
        return min([24.0, voltage * self._base_speed()])

    def _load_factor(self):
        return 9.0

    def _base_speed(self):
        return 12.0