import argparse
from .src.askill import Askill


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="Askill",
        usage="python argparse.py file.ask"
    )

    parser.add_argument("filepath")

    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.filepath, "r") as file:
        text = file.read()

    ask = Askill.parse(text)

    ask.draw()

    print(ask.render())


if __name__ == "__main__":
    main()
