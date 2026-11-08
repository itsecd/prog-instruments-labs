from abc import ABC, abstractmethod


class Parrot(ABC):
    """Abstract base class for all parrot species.

    Defines the common interface (cry and speed) that every concrete
    parrot must implement. The base speed constant is shared among all
    species unless overridden.

    Attributes:
        BASE_SPEED (float): The default speed for a parrot in flight
            conditions, measured in abstract units.
    """

    BASE_SPEED: float = 12.0

    @abstractmethod
    def cry(self) -> str:
        """Produce the characteristic sound of the parrot.

        Returns:
            str: A string representing the parrot's cry.
        """

    @abstractmethod
    def speed(self) -> float:
        """Calculate the current speed of the parrot.

        Returns:
            float: The speed in abstract units. The value is always
                non-negative; if the parrot cannot move, 0.0 is returned.
        """


class EuropeanParrot(Parrot):
    """A European parrot.

    The simplest species: it always flies at the base speed and has a
    cheerful cry. No additional state is required.
    """

    def cry(self) -> str:
        """Produce the European parrot's cry.

        Returns:
            str: The string "Sqoork!".
        """
        return "Sqoork!"

    def speed(self) -> float:
        """Return the European parrot's speed.

        Returns:
            float: The base speed, unaffected by any external factors.
        """
        return Parrot.BASE_SPEED


class AfricanParrot(Parrot):
    """An African parrot.

    Its speed is reduced by the number of coconuts attached to it.
    The load factor determines how much a single coconut slows the
    parrot down. The speed can never drop below zero.

    Attributes:
        LOAD_FACTOR (float): The penalty applied per coconut.
    """

    LOAD_FACTOR: float = 9.0

    def __init__(self, number_of_coconuts: int) -> None:
        """Initialize an African parrot.

        Args:
            number_of_coconuts: The number of coconuts attached to the
                parrot. Must be non-negative (not validated here).
        """
        self._number_of_coconuts = number_of_coconuts

    def cry(self) -> str:
        """Produce the African parrot's cry.

        Returns:
            str: The string "Sqaark!".
        """
        return "Sqaark!"

    def speed(self) -> float:
        """Calculate the African parrot's speed.

        The speed is the base speed minus the load factor multiplied
        by the number of coconuts. The result is clamped to zero so
        that it never becomes negative.

        Returns:
            float: The resulting speed, always >= 0.
        """
        return max(
            0.0,
            Parrot.BASE_SPEED
            - AfricanParrot.LOAD_FACTOR * self._number_of_coconuts,
        )


class NorwegianBlueParrot(Parrot):
    """A Norwegian blue parrot.

    Its speed depends on the voltage applied to it, but only if it is
    not nailed to a perch. A nailed parrot cannot move at all.

    Attributes:
        MAX_SPEED (float): The upper bound for the parrot's speed,
            regardless of voltage.
    """

    MAX_SPEED: float = 24.0

    def __init__(self, voltage: float, nailed: bool) -> None:
        """Initialize a Norwegian blue parrot.

        Args:
            voltage: The voltage applied to the parrot, in volts.
            nailed: Whether the parrot is nailed to its perch.
        """
        self._voltage = voltage
        self._nailed = nailed

    def cry(self) -> str:
        """Produce the Norwegian blue parrot's cry.

        Returns:
            str: "Bzzzzzz" if voltage is positive, otherwise "...".
        """
        return "Bzzzzzz" if self._voltage > 0 else "..."

    def speed(self) -> float:
        """Calculate the Norwegian blue parrot's speed.

        A nailed parrot or a parrot with negative voltage cannot move,
        so its speed is 0.0. Otherwise, the speed is the base speed
        scaled by the voltage, capped at MAX_SPEED.

        Returns:
            float: The resulting speed, always >= 0.
        """
        if self._nailed or self._voltage < 0:
            return 0.0
        return min(
            NorwegianBlueParrot.MAX_SPEED,
            self._voltage * Parrot.BASE_SPEED,
        )



def main() -> None:
    parrot1 = EuropeanParrot()
    print("European Parrot")
    print(f"Cry: {parrot1.cry()}")
    print(f"Speed: {parrot1.speed()}")

    parrot2 = AfricanParrot(number_of_coconuts=0)
    print("African Parrot (0 coconuts)")
    print(f"Cry: {parrot2.cry()}")
    print(f"Speed: {parrot2.speed()}")

    parrot3 = AfricanParrot(number_of_coconuts=1)
    print("African Parrot (1 coconut)")
    print(f"Cry: {parrot3.cry()}")
    print(f"Speed: {parrot3.speed()}")

    parrot4 = AfricanParrot(number_of_coconuts=2)
    print("African Parrot (2 coconuts)")
    print(f"Cry: {parrot4.cry()}")
    print(f"Speed: {parrot4.speed()}")

    parrot5 = NorwegianBlueParrot(voltage=1.5, nailed=False)
    print("Norwegian Blue Parrot (not nailed, 1.5V)")
    print(f"Cry: {parrot5.cry()}")
    print(f"Speed: {parrot5.speed()}")

    parrot6 = NorwegianBlueParrot(voltage=4.0, nailed=False)
    print("Norwegian Blue Parrot (not nailed, 4.0V)")
    print(f"Cry: {parrot6.cry()}")
    print(f"Speed: {parrot6.speed()}")

    parrot7 = NorwegianBlueParrot(voltage=1.5, nailed=True)
    print("Norwegian Blue Parrot (nailed)")
    print(f"Cry: {parrot7.cry()}")
    print(f"Speed: {parrot7.speed()}")

    parrot8 = NorwegianBlueParrot(voltage=0.0, nailed=False)
    print("Norwegian Blue Parrot (no voltage)")
    print(f"Cry: {parrot8.cry()}")
    print(f"Speed: {parrot8.speed()}")


if __name__ == "__main__":
    main()