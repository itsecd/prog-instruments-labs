"""Main entry point for the application."""
import sys
from src.cli.parser import parse_arguments
from src.gui.main_window import start_gui
from src.hash_logic.statistics import run_experiments
from src.utils.file_manager import load_json, save_json

def main():
    """
    Main function to run the application in CLI or GUI mode.

    Args:
        None

    Returns:
        None

    Raises:
        ValueError: If arguments structure is unrecognized.
        Exception: If any top-level error occurs.
    """
    try:
        settings = load_json("settings.json")
        args = parse_arguments(settings)

        match vars(args):

            case {'mode': 'gui'}:
                start_gui(settings)

            case {'mode': 'cli', 'bits': bits, 'experiments': exps, 'out': out_file}:
                print(f"Running in CLI mode. Bits: {bits}, Experiments: {exps}")
                result = run_experiments(bits, exps, settings)

                print("\n--- RESULTS ---")
                print(f"Theory (Expected attempts): {result['theoretical_attempts']:.2f}")
                print(f"Practical (Average attempts): {result['average_attempts']:.2f}")
                for idx, col in enumerate(result['collisions'], 1):
                    print(f"[{idx}] {col['str1']} == {col['str2']} -> {col['hash']} (Attempts: {col['attempts']})")

                if out_file:
                    save_json(out_file, result)
                    print(f"\nResults successfully saved to JSON file: {out_file}")

            case _:
                raise ValueError("Unknown command-line arguments structure received.")

    except Exception as e:
        print(f"Critical error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()