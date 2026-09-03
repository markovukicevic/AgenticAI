def get_weather(city: str) -> str:
    return f"The weather in {city} is sunny."

if __name__ == "__main__":
    print(get_weather("Berlin"))