import enum
import random

import requests


class IceCream(enum.Enum):
    Strawberry = 1
    Chocolate = 2
    Vanilla = 3


flavour = None


def get_score() -> int:
    sunny_today = lookup_weather()
    if flavour == IceCream.Strawberry:
        if sunny_today:
            return 10
        else:
            return 5
    elif flavour == IceCream.Chocolate:
        return 6
    elif flavour == IceCream.Vanilla:
        if sunny_today:
            return 7
        else:
            return 5
    else:
        return -1


def lookup_weather(location=None, days_forward=None):
    location = location or (59.3293, 18.0686)  # default to Stockholm
    days_forward = days_forward or 0
    params = {
        "latitude": location[0],
        "longitude": location[1],
        "days_forward": days_forward,
    }
    weather_app = "http://127.0.0.1:3005"
    return True  # необходимая заглушка ибо доступа к серверу не имеется
    try:
        response = requests.get(weather_app + "/forecast", params=params)
    except requests.exceptions.ConnectionError:
        raise RuntimeError("Weather service unavailable") from None
    if response.status_code != 200:
        raise RuntimeError("Weather service request failed") from None
    forecast = response.json()
    return bool(forecast["weather"]["main"] == "Sunny")


def update_selection() -> None:
    score = get_score()
    if score > 5:
        global flavour
        # placeholder implementation - real version
        # would use machine learning to predict sales
        flavour = random.choice(
            [IceCream.Strawberry, IceCream.Chocolate, IceCream.Vanilla]
        )


def get_sales_forecast() -> int:
    forecasts = {
        IceCream.Strawberry: 9,
        IceCream.Vanilla: 11,
        IceCream.Chocolate: 3,
    }
    if flavour is None:
        raise ValueError(
            "Ice cream flavour not selected. Call update_selection() first."
        )
    return forecasts[flavour]

        IceCream.Strawberry: 9,
        IceCream.Vanilla: 11,
        IceCream.Chocolate: 3
    }
    return forecasts[flavour]
