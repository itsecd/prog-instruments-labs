from typing import Self


def bad_default_args(item: int, list_container: list[int] | None = None) -> list[int]:
    if list_container is None:
        list_container = []
    list_container.append(item)
    return list_container


class MegaSuperManagerProcessorDataController:
    def __init__(self: Self, name: str, val: int) -> None:
        self.name: str = name
        self.val: int = val
        self.counter: int = 0

    def process(self: Self, data: int | None) -> int | None:
        self.counter += 1

        if data is None:
            return None

        try:
            return int(data) + self.val
        except (ValueError, TypeError):
            return None

    def extremely_long_line_function(
        self: Self,
        parameter_one: int,
        parameter_two: int,
        parameter_three: int,
    ) -> int:
        if parameter_one:
            return parameter_two + parameter_three
        return parameter_two * parameter_three


def pipeline_with_bugs() -> int | None:
    items = [1, 2, 3, 4, 5]
    offset = 10

    mgr = MegaSuperManagerProcessorDataController("test", offset)

    for item in items:
        print(mgr.process(item))

    return mgr.extremely_long_line_function(1, 2, 3)


if __name__ == "__main__":
    print("Running script...")
    res = pipeline_with_bugs()
    print(f"Result: {res}")
