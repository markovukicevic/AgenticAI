from datetime import datetime
from zoneinfo import ZoneInfo

from schemas import TimeResult


TIMEZONES = {
    "Berlin": "Europe/Berlin",
    "London": "Europe/London",
    "New York": "America/New_York",
    "Tokyo": "Asia/Tokyo",
}


def get_time(city: str) -> TimeResult:
    if city not in TIMEZONES:
        supported_cities = ", ".join(TIMEZONES.keys())

        raise ValueError(
            f"Unsupported city: {city}. "
            f"Supported cities are: {supported_cities}."
        )

    timezone = TIMEZONES[city]

    current_time = datetime.now(
        ZoneInfo(timezone)
    )

    return TimeResult(
        city=city,
        time=current_time.strftime("%H:%M:%S"),
        timezone=timezone,
    )


if __name__ == "__main__":
    result = get_time("Berlin")

    print(result)
    print(result.model_dump())