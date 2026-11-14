from parrot import ParrotType, create_parrot


def test_speed_of_european_parrot():
    parrot = create_parrot(ParrotType.EUROPEAN, 0, 0, False)
    assert parrot.speed() == 12.0


def test_cry_of_european_parrot():
    parrot = create_parrot(ParrotType.EUROPEAN, 0, 0, False)
    assert parrot.cry() == "Sqoork!"


def test_speed_of_african_parrot_with_one_coconut():
    parrot = create_parrot(ParrotType.AFRICAN, 1, 0, False)
    assert parrot.speed() == 3.0


def test_cry_of_african_parrot():
    parrot = create_parrot(ParrotType.AFRICAN, 1, 0, False)
    assert parrot.cry() == "Sqaark!"


def test_speed_of_african_parrot_with_two_coconuts():
    parrot = create_parrot(ParrotType.AFRICAN, 2, 0, False)
    assert parrot.speed() == 0.0


def test_speed_of_african_parrot_with_no_coconuts():
    parrot = create_parrot(ParrotType.AFRICAN, 0, 0, False)
    assert parrot.speed() == 12.0


def test_speed_norwegian_blue_parrot_nailed():
    parrot = create_parrot(ParrotType.NORWEGIAN_BLUE, 0, 1.5, True)
    assert parrot.speed() == 0.0


def test_speed_norwegian_blue_parrot_not_nailed():
    parrot = create_parrot(ParrotType.NORWEGIAN_BLUE, 0, 1.5, False)
    assert parrot.speed() == 18.0


def test_speed_norwegian_blue_parrot_not_nailed_high_voltage():
    parrot = create_parrot(ParrotType.NORWEGIAN_BLUE, 0, 4, False)
    assert parrot.speed() == 24.0


def test_cry_norwegian_blue_parrot_high_voltage():
    parrot = create_parrot(ParrotType.NORWEGIAN_BLUE, 0, 4, False)
    assert parrot.cry() == "Bzzzzzz"


def test_cry_norwegian_blue_parrot_no_voltage():
    parrot = create_parrot(ParrotType.NORWEGIAN_BLUE, 0, 0, False)
    assert parrot.cry() == "..."

def test_speed_of_african_parrot_never_negative():
    parrot = create_parrot(ParrotType.AFRICAN, 5, 0, False)
    assert parrot.speed() == 0.0

def test_speed_norwegian_blue_parrot_no_voltage():
    parrot = create_parrot(ParrotType.NORWEGIAN_BLUE, 0, 0, False)
    assert parrot.speed() == 0.0

def test_cry_norwegian_blue_parrot_nailed_with_voltage():
    parrot = create_parrot(ParrotType.NORWEGIAN_BLUE, 0, 4, True)
    assert parrot.cry() == "Bzzzzzz"

def test_european_parrot_ignores_other_parameters():
    parrot = create_parrot(ParrotType.EUROPEAN, 3, 2, True)
    assert parrot.speed() == 12.0
    assert parrot.cry() == "Sqoork!"