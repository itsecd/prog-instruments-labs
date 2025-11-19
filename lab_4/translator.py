"""
Main entry point for translation application with new architecture.
"""

import sys
import argparse
from multiprocessing import Pool, freeze_support
from itertools import repeat
from typing import List

from config import TranslationConfig, BatchConfig
from translation_session import TranslationSession
from translation_orchestrator import TranslationOrchestrator


def setup_argparse() -> argparse.ArgumentParser:
    """
    Setup command line argument parser.

    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="Translate PO files using Google Translate",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Translate specific languages
  python translator.py --languages de fr es --locale-path /path/to/locale
  
  # Use configuration file
  python translator.py --config config.json
  
  # Single language with custom settings
  python translator.py --language ja --interface-language en --headless
        """
    )

    # Input options
    input_group = parser.add_argument_group('Input Options')
    input_group.add_argument(
        '--languages', '-l',
        nargs='+',
        help='Language codes to translate (e.g., de fr es)'
    )
    input_group.add_argument(
        '--language-file',
        help='File containing list of language codes (one per line)'
    )
    input_group.add_argument(
        '--config', '-c',
        help='JSON configuration file path'
    )

    # Path options
    path_group = parser.add_argument_group('Path Options')
    path_group.add_argument(
        '--driver-path', '-d',
        required='--config' not in sys.argv,
        help='Path to ChromeDriver executable'
    )
    path_group.add_argument(
        '--locale-path', '-p',
        required='--config' not in sys.argv,
        help='Path to locale directory'
    )

    # Translation options
    translation_group = parser.add_argument_group('Translation Options')
    translation_group.add_argument(
        '--interface-language', '-i',
        default='en',
        help='Google Translate interface language (default: en)'
    )
    translation_group.add_argument(
        '--source-language', '-s',
        default='en',
        help='Source language for translation (default: en)'
    )
    translation_group.add_argument(
        '--max-retries', '-r',
        type=int,
        default=3,
        help='Maximum retry attempts for failed translations (default: 3)'
    )

    # Execution options
    execution_group = parser.add_argument_group('Execution Options')
    execution_group.add_argument(
        '--headless',
        action='store_true',
        default=True,
        help='Run browser in headless mode (default: True)'
    )
    execution_group.add_argument(
        '--no-headless',
        action='store_false',
        dest='headless',
        help='Disable headless mode'
    )
    execution_group.add_argument(
        '--multi-process', '-m',
        action='store_true',
        help='Use multiprocessing for parallel translation'
    )
    execution_group.add_argument(
        '--max-processes',
        type=int,
        default=10,
        help='Maximum number of parallel processes (default: 10)'
    )

    return parser


def load_language_codes(args: argparse.Namespace) -> List[str]:
    """
    Load language codes from various sources.

    Args:
        args: Command line arguments

    Returns:
        List[str]: List of language codes

    Raises:
        ValueError: If no language codes provided
    """
    languages = []

    # From command line argument
    if args.languages:
        languages.extend(args.languages)

    # From file
    if args.language_file:
        try:
            with open(args.language_file, 'r', encoding='utf-8') as file:
                file_languages = [line.strip() for line in file if line.strip()]
                languages.extend(file_languages)
        except FileNotFoundError:
            print(f"[!] Language file not found: {args.language_file}")

    # Remove duplicates
    languages = list(set(languages))

    if not languages:
        raise ValueError("No language codes provided. Use --languages or --language-file")

    return languages


def create_config_from_args(args: argparse.Namespace) -> TranslationConfig:
    """
    Create configuration from command line arguments.

    Args:
        args: Command line arguments

    Returns:
        TranslationConfig: Created configuration
    """
    if args.config:
        return TranslationConfig.from_json_file(args.config)
    else:
        config = TranslationConfig(
            driver_path=args.driver_path,
            locale_path=args.locale_path,
            headless=args.headless,
            interface_language=args.interface_language,
            source_language=args.source_language,
            max_retries=args.max_retries,
            multi_process=args.multi_process,
            max_processes=args.max_processes
        )
        config.validate()
        return config


def translate_single_process(batch_config: BatchConfig):
    """
    Translate languages in single process mode.

    Args:
        batch_config: Batch configuration
    """
    session = TranslationSession(batch_config)
    session.start()

    for language_code in batch_config.language_codes:
        session.translate_language(language_code)
        session.print_progress()

    session.complete()


def translate_multi_process(batch_config: BatchConfig):
    """
    Translate languages in multiprocess mode.

    Args:
        batch_config: Batch configuration
    """
    freeze_support()

    config = batch_config.translation_config
    language_codes = batch_config.language_codes.copy()

    print(f"[+] Starting multiprocess translation with {config.max_processes} max processes")
    print(f"[+] Total languages: {len(language_codes)}")

    processed_count = 0

    while language_codes:
        current_batch = language_codes[:config.max_processes]
        language_codes = language_codes[config.max_processes:]

        print(f"[+] Processing batch: {current_batch}")

        with Pool(processes=len(current_batch)) as pool:
            results = pool.starmap(process_language_batch, [
                (code, config) for code in current_batch
            ])

        processed_count += len(current_batch)
        success_count = sum(1 for result in results if result)

        print(f"[+] Batch completed: {success_count}/{len(current_batch)} successful")
        print(f"[+] Overall progress: {processed_count}/{len(batch_config.language_codes)}")

    print(f"[+] Multiprocess translation completed")


def process_language_batch(language_code: str, config: TranslationConfig) -> bool:
    """
    Process a single language in multiprocessing pool.

    Args:
        language_code: Language code to translate
        config: Translation configuration

    Returns:
        bool: True if translation was successful
    """
    try:
        with TranslationOrchestrator(config) as orchestrator:
            return orchestrator.translate_language(language_code)
    except Exception as error:
        print(f"[!] Multiprocess error for {language_code}: {error}")
        return False


def main():
    """Main application entry point."""
    parser = setup_argparse()
    args = parser.parse_args()

    try:
        # Load configuration
        config = create_config_from_args(args)

        # Load language codes
        language_codes = load_language_codes(args)

        # Create batch configuration
        batch_config = BatchConfig(language_codes, config)
        batch_config.validate()

        print("[+] Translation Configuration:")
        print(f"    Languages: {', '.join(language_codes)}")
        print(f"    Interface: {config.interface_language}")
        print(f"    Source: {config.source_language}")
        print(f"    Headless: {config.headless}")
        print(f"    Multiprocess: {config.multi_process}")
        print(f"    Max Processes: {config.max_processes}")
        print(f"    Max Retries: {config.max_retries}")

        # Execute translation
        if config.multi_process:
            translate_multi_process(batch_config)
        else:
            translate_single_process(batch_config)

        print("[+] Translation process completed successfully")

    except ValueError as e:
        print(f"[!] Configuration error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[!] Translation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"[!] Unexpected error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()