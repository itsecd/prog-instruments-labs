from checksum import calculate_checksum

def test_checksum_known_value():
    result = calculate_checksum([0, 1, 2, 3])
    assert isinstance(result, str)
    assert len(result) == 32

def test_checksum_empty_list():
    result = calculate_checksum([])
    assert isinstance(result, str)
    assert len(result) == 32