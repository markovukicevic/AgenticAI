from schemas import CalculatorResult


def calculator(expression: str) -> CalculatorResult:
    result = eval(expression)

    return CalculatorResult(
        expression=expression,
        result=result,
    )


if __name__ == "__main__":
    result = calculator("25 * 12")

    print(result)
    print(result.model_dump())