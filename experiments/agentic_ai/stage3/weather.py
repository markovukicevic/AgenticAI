from schemas import WeatherResult


def get_weather(
    city: str,
    units: str = "celsius",
) -> WeatherResult:
    return WeatherResult(
        city=city,
        temperature=18,
        unit=units,
        condition="sunny",
    )


if __name__ == "__main__":
    result = get_weather("Berlin")

    print(result)
    print(type(result))
    print(result.model_dump())