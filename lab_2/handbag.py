import dataclasses

# Max handbag size in litres - larger than this and it's counted as luggage
MAX_HANDBAG_CAPACITY = 25.0


@dataclasses.dataclass
class Event:
    type: int
    size: int
    status_code: int
    checksum: int
    payload: str
    has_checksum: bool


@dataclasses.dataclass
class Handbag:
    compartments: int
    designer: str
    market_segment: str
    strap_type: str
    capacity: float


def predict_sales(handbag: Handbag):
    score = 0
    if handbag.compartments > 10:
        score -= 1
    if handbag.designer in ["H&M", "Zara", "Esprit"]:
        score += 3
    if handbag.market_segment == "Trendy":
        score += 1
    if handbag.market_segment == "Traveller":
        score -= 2
    if handbag.strap_type == "Clutch":
        score -= 1
    if handbag.strap_type == "Shoulder":
        score += 1
    if handbag.capacity < 1:
        score -= 1
    if handbag.capacity > 3:
        score += 1
    return score


def fits_desired_market_segment(handbag: Handbag):
    predicted_sales = predict_sales(handbag)
    return predicted_sales > 0


def has_recognized_brand(handbag: Handbag):
    return handbag.designer in ["H&M", "Zara", "Esprit"]


def has_dead_brand(handbag: Handbag):
    return handbag.designer in ["Debenhams", "Woolworths"]


def init_handbag_event(handbag: Handbag, event: Event) -> bool:
    result = False
    size = 0
    if handbag.capacity <= MAX_HANDBAG_CAPACITY and event is not None:
        size = 1
        if fits_desired_market_segment(handbag):
            if has_recognized_brand(handbag) or has_dead_brand(handbag):
                response = determine_event_payload(handbag, event)
                if response:
                    size = len(response)
                    result = True
            else:
                event.status_code = "Invalid brand"
                result = True
        else:
            event.status_code = "Invalid market segment"
            result = True
    else:
        # timeout
        pass

    if result:
        event.size = size
        event.checksum = calculate_checksum(event)
        event.has_checksum = True
        result = True
    return result


def determine_event_payload(handbag: Handbag, event: Event):
    event.payload = None
    if handbag.strap_type == "Shoulder":
        event.payload = "Shoulder Handbag"
        event.status_code = "Good"
        event.type = 1
    elif handbag.strap_type == "Clutch":
        event.payload = "Clutch Handbag"
        event.status_code = "Good"
        event.type = 2
    elif handbag.strap_type == "Fixed":
        event.payload = "Hard strap Handbag"
        event.status_code = "Good"
        event.type = 3
    else:
        event.status_code = "Unknown"
    return event.payload


def calculate_checksum(event):
    return (event.type + len(event.status_code) + event.size + len(event.payload)) % 10
