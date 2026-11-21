import argparse
import re
from collections import Counter
from pathlib import Path


def read_file(filename: str) -> str:
    """Reads the content of the given file."""
    file_path = Path(filename)
    try:
        return file_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"File '{filename}' not found.") from exc


def extract_cities(text: str) -> list[str]:
    """Extracts city names from the provided text using regex."""
    city_pattern = r"Город:\s*(?:г\.\s*)?([А-ЯЁа-яё\- ]+)(?=\n|$)"
    cities = re.findall(city_pattern, text)
    return [city.strip().replace("  ", " ") for city in cities]


def count_cities(cities: list[str]) -> list[tuple[str, int]]:
    """Counts and sorts the number of people in each city."""
    city_counts = Counter(cities)
    return sorted(city_counts.items(), key=lambda x: (-x[1], x[0]))


def save_results(results: list[tuple[str, int]], output_filename: str) -> None:
    """Saves city statistics to a text file."""
    result_lines = [f"{city}: {count}" for city, count in results]
    result_text = "\n".join(result_lines)

    output_path = Path(output_filename)
    output_path.write_text(result_text, encoding="utf-8")

    print("City statistics:")
    print(result_text)
    print(f"\nResults saved to file: {output_filename}")


def parse_arguments() -> argparse.Namespace:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(
        description="Calculate city statistics from survey data"
    )
    parser.add_argument(
        "filename",
        nargs="?",
        default="data.txt",
        type=str,
        help="Input file with data",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="cities_stats.txt",
        type=str,
        help="Output file for results",
    )
    return parser.parse_args()


def main() -> None:
    """Main function that coordinates the program flow."""
    args = parse_arguments()

    try:
        text = read_file(args.filename)
    except FileNotFoundError as exc:
        print(exc)
        return

    cities = extract_cities(text)

    if not cities:
        print("No cities found in the file.")
        return

    results = count_cities(cities)
    save_results(results, args.output)


if __name__ == "__main__":
    main()
