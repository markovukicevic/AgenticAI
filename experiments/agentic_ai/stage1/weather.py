def get_weather(city: str, units: str = "celsius") -> str:
    return f"The weather in {city} is sunny ({units})."


if __name__ == "__main__":
    print(get_weather("Berlin"))
    print(get_weather("Berlin", "fahrenheit"))