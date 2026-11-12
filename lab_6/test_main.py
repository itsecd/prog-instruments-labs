from main import build_url


def test_build_url():
    result = build_url("rain")

    assert result == "https://mixkit.co/free-sound-effects/rain/"