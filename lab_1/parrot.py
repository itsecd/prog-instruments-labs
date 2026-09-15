"""Модуль для работы с попугаями.

Содержит класс Parrot и перечисление ParrotType для моделирования
различных видов попугаев с их криками и скоростями.
"""

from enum import Enum


class ParrotType(Enum):
    """Типы попугаев.

    Каждый тип определяет поведение попугая: его скорость и крик.
    """

    EUROPEAN = 1
    AFRICAN = 2
    NORWEGIAN_BLUE = 3


class Parrot:
    """Класс, представляющий попугая.

    Попугай имеет тип, который определяет его скорость и крик.
    В зависимости от типа могут использоваться дополнительные параметры:
    количество кокосов, напряжение и состояние "прибитости".

    Attributes:
        _type: Тип попугая (ParrotType).
        _number_of_coconuts: Количество кокосов (для африканского попугая).
        _voltage: Напряжение (для норвежского синего попугая).
        _nailed: Прибит ли попугай к насесту (для норвежского синего попугая).

    """

    def __init__(
        self,
        type_of_parrot: ParrotType,
        number_of_coconuts: int,
        voltage: float,
        nailed: bool,
    ) -> None:
        """Инициализирует попугая с заданными параметрами.

        Args:
            type_of_parrot: Тип попугая (значение из ParrotType).
            number_of_coconuts: Количество кокосов, которые несёт попугай.
            voltage: Напряжение питания попугая.
            nailed: True, если попугай прибит к насесту.

        """
        self._type = type_of_parrot
        self._number_of_coconuts = number_of_coconuts
        self._voltage = voltage
        self._nailed = nailed

    def speed(self) -> float:
        """Возвращает скорость попугая.

        Скорость зависит от типа попугая:
        - Европейский: базовая скорость.
        - Африканский: базовая скорость минус нагрузка от кокосов.
        - Норвежский синий: 0, если прибит, иначе зависит от напряжения.

        Returns:
            float: Скорость попугая в условных единицах.

        """
        match self._type:
            case ParrotType.EUROPEAN:
                return self._base_speed()
            case ParrotType.AFRICAN:
                return max(
                    0,
                    self._base_speed() - self._load_factor() * self._number_of_coconuts,
                )
            case ParrotType.NORWEGIAN_BLUE:
                if self._nailed:
                    return 0
                else:
                    return self._compute_base_speed_for_voltage(self._voltage)
            case _:
                raise TypeError("This parrot not exists")

    def cry(self) -> str:
        """Возвращает крик попугая.

        Крик зависит от типа попугая:
        - Европейский: "Sqoork!".
        - Африканский: "Sqaark!".
        - Норвежский синий: "Bzzzzzz", если напряжение > 0, иначе "...".

        Returns:
            str: Строка, представляющая крик попугая.

        """
        match self._type:
            case ParrotType.EUROPEAN:
                return "Sqoork!"
            case ParrotType.AFRICAN:
                return "Sqaark!"
            case ParrotType.NORWEGIAN_BLUE:
                return "Bzzzzzz" if self._voltage > 0 else "..."
            case _:
                raise TypeError("This parrot not exists")

    def _compute_base_speed_for_voltage(self, voltage: float) -> float:
        """Вычисляет скорость норвежского синего попугая по напряжению.

        Скорость ограничена максимальным значением 24.0 и вычисляется
        как произведение напряжения на базовую скорость.

        Args:
            voltage: Напряжение питания попугая.

        Returns:
            float: Скорость попугая, не превышающая 24.0.

        """
        return min([24.0, voltage * self._base_speed()])

    def _load_factor(self) -> float:
        """Возвращает коэффициент нагрузки от кокосов.

        Returns:
            float: Коэффициент нагрузки (9.0).

        """
        return 9.0

    def _base_speed(self) -> float:
        """Возвращает базовую скорость попугая.

        Returns:
            float: Базовая скорость (12.0).

        """
        return 12.0
