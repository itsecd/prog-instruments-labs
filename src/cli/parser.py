"""Module for command-line arguments parsing."""
import argparse

def parse_arguments(settings: dict) -> argparse.Namespace:
    """
    Parses command line arguments dynamically using settings and subparsers.

    Args:
        settings (dict): Application settings dictionary.

    Returns:
        argparse.Namespace: Parsed arguments object.

    Raises:
        Exception: If any error occurs during parsing.
    """
    try:
        allowed_bits = settings["hasher"]["allowed_bits"]
        default_bits = settings["hasher"]["default_bits"]
        default_exp = settings["experiments"]["default_count"]

        parser = argparse.ArgumentParser(description=settings["gui"]["window_title"])

        subparsers = parser.add_subparsers(dest='mode', required=True, help='Application mode (gui or cli)')

        subparsers.add_parser('gui', help='Launch Graphical User Interface')

        cli_parser = subparsers.add_parser('cli', help='Launch Command Line Interface')
        cli_parser.add_argument('--bits', type=int, choices=allowed_bits, default=default_bits,
                            help=f"Hash bits {allowed_bits}")
        cli_parser.add_argument('--experiments', type=int, default=default_exp,
                            help="Number of experiments for stats")
        cli_parser.add_argument('--out', type=str, default=None,
                            help="Path to save the results as a JSON file")

        return parser.parse_args()
    except Exception as e:
        raise Exception(f"CLI parsing error: {e}")