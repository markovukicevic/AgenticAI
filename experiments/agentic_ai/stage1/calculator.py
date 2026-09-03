def calculator(expression: str) -> float:
    return eval(expression)

if __name__ == "__main__":
    result = calculator("347 * 928")
    print(result)