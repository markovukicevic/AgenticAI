from datetime import datetime
from zoneinfo import ZoneInfo


TIMEZONES = {
    "Berlin": "Europe/Berlin",
    "London": "Europe/London",
    "New York": "America/New_York",
    "Tokyo": "Asia/Tokyo",
}


def get_time(city: str) -> str:
    if city not in TIMEZONES:
        supported_cities = ", ".join(TIMEZONES.keys())

        raise ValueError(
            f"Unsupported city: {city}. "
            f"Supported cities are: {supported_cities}."
        )

    current_time = datetime.now(
        ZoneInfo(TIMEZONES[city])
    )

    return current_time.strftime("%H:%M:%S")


if __name__ == "__main__":
    print(get_time("Berlin"))