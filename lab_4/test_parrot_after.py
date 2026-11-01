from parrot_after import (
    EuropeanParrot,
    AfricanParrot,
    NorwegianBlueParrot,
)


def test_speed_of_european_parrot() -> None:
    parrot = EuropeanParrot()
    assert parrot.speed() == 12.0


def test_cry_of_european_parrot() -> None:
    parrot = EuropeanParrot()
    assert parrot.cry() == "Sqoork!"


def test_speed_of_african_parrot_with_one_coconut() -> None:
    parrot = AfricanParrot(number_of_coconuts=1)
    assert parrot.speed() == 3.0


def test_cry_of_african_parrot() -> None:
    parrot = AfricanParrot(number_of_coconuts=1)
    assert parrot.cry() == "Sqaark!"


def test_speed_of_african_parrot_with_two_coconuts() -> None:
    parrot = AfricanParrot(number_of_coconuts=2)
    assert parrot.speed() == 0.0


def test_speed_of_african_parrot_with_no_coconuts() -> None:
    parrot = AfricanParrot(number_of_coconuts=0)
    assert parrot.speed() == 12.0


def test_speed_norwegian_blue_parrot_nailed() -> None:
    parrot = NorwegianBlueParrot(voltage=1.5, nailed=True)
    assert parrot.speed() == 0.0


def test_speed_norwegian_blue_parrot_not_nailed() -> None:
    parrot = NorwegianBlueParrot(voltage=1.5, nailed=False)
    assert parrot.speed() == 18.0


def test_speed_norwegian_blue_parrot_not_nailed_high_voltage() -> None:
    parrot = NorwegianBlueParrot(voltage=4.0, nailed=False)
    assert parrot.speed() == 24.0


def test_cry_norwegian_blue_parrot_high_voltage() -> None:
    parrot = NorwegianBlueParrot(voltage=4.0, nailed=False)
    assert parrot.cry() == "Bzzzzzz"


def test_cry_norwegian_blue_parrot_no_voltage() -> None:
    parrot = NorwegianBlueParrot(voltage=0.0, nailed=False)
    assert parrot.cry() == "..."