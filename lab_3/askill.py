import argparse
from src.askill import Askill


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="Askill",
        description="ASCII mini-language",
        usage="python askill.py file.ask",
    )

    parser.add_argument("filepath")
    parser.add_argument("-s", "--spaces", type=int)
    parser.add_argument("-e", "--enters", type=int)

    args = parser.parse_args()

    if not args.filepath.endswith(".ask"):
        raise RuntimeError("File must ends with '.ask'")

    return args


def main():
    args = parse_args()

    with open(args.filepath, "r") as file:
        text = file.read()

    ask = Askill.parse(text)

    ask.draw()

    spaces = args.spaces if args.spaces is not None else 1
    enters = args.enters if args.enters is not None else 1

    render = ask.render(spaces, enters)

    print(render)


if __name__ == "__main__":
    main()
