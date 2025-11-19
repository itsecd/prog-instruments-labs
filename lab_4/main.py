"""
Main application entry point with advanced features.
"""

import sys
import argparse
import signal
import time
from pathlib import Path
from typing import List, Dict, Any

from config import TranslationConfig, BatchConfig
from translation_session import TranslationSession
from advanced_logger import setup_logging, get_logger
from metrics_collector import get_metrics_collector
from health_monitor import get_health_monitor


class TranslationApplication:
    """
    Main application class with advanced features.
    """

    def __init__(self):
        """Initialize translation application."""
        self.logger = None
        self.metrics = get_metrics_collector()
        self.health_monitor = get_health_monitor()
        self.is_running = False

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        print(f"\n[!] Received signal {signum}, shutting down gracefully...")
        self.is_running = False

    def run(self, args: argparse.Namespace) -> bool:
        """
        Run the translation application.

        Args:
            args: Command line arguments

        Returns:
            bool: True if application completed successfully
        """
        self.is_running = True

        try:
            # Setup logging
            self.logger = setup_logging(
                log_level=getattr(args, 'log_level', 'INFO'),
                log_dir=getattr(args, 'log_dir', 'logs'),
                enable_file_logging=not getattr(args, 'no_file_logging', False)
            )

            logger_instance = self.logger.get_logger()
            logger_instance.info("Starting translation application")

            # Load configuration and languages
            config = self._load_configuration(args)
            language_codes = self._load_language_codes(args)

            if not language_codes:
                logger_instance.error("No language codes provided")
                return False

            # Create batch configuration
            batch_config = BatchConfig(language_codes, config)
            batch_config.validate()

            # Start health monitoring
            if getattr(args, 'enable_health_monitoring', True):
                self.health_monitor.start_monitoring()
                logger_instance.info("Health monitoring started")

            # Log startup information
            self._log_startup_info(batch_config)

            # Execute translation
            success = self._execute_translation(batch_config)

            # Log completion
            self._log_completion_info(success)

            return success

        except KeyboardInterrupt:
            logger_instance.warning("Application interrupted by user")
            return False
        except Exception as e:
            if self.logger:
                self.logger.get_logger().error(f"Application error: {e}", exc_info=True)
            else:
                print(f"[!] Application error: {e}")
            return False
        finally:
            self._cleanup()

    def _load_configuration(self, args: argparse.Namespace) -> TranslationConfig:
        """
        Load translation configuration.

        Args:
            args: Command line arguments

        Returns:
            TranslationConfig: Loaded configuration
        """
        if hasattr(args, 'config') and args.config:
            config = TranslationConfig.from_json_file(args.config)
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

    def _load_language_codes(self, args: argparse.Namespace) -> List[str]:
        """
        Load language codes from various sources.

        Args:
            args: Command line arguments

        Returns:
            List[str]: List of language codes
        """
        languages = []

        # From command line
        if getattr(args, 'languages', None):
            languages.extend(args.languages)

        # From file
        if getattr(args, 'language_file', None):
            try:
                with open(args.language_file, 'r', encoding='utf-8') as file:
                    file_languages = [line.strip() for line in file if line.strip()]
                    languages.extend(file_languages)
            except FileNotFoundError:
                self.logger.get_logger().error(f"Language file not found: {args.language_file}")

        # Remove duplicates and validate
        languages = list(set(languages))
        valid_languages = [lang for lang in languages if len(lang) == 2]

        if len(valid_languages) != len(languages):
            invalid = set(languages) - set(valid_languages)
            self.logger.get_logger().warning(f"Invalid language codes: {invalid}")

        return valid_languages

    def _log_startup_info(self, batch_config: BatchConfig):
        """
        Log startup information.

        Args:
            batch_config: Batch configuration
        """
        config_dict = batch_config.translation_config.to_dict()

        self.logger.log_translation_start(
            len(batch_config.language_codes),
            config_dict
        )

        # Print startup banner
        print("\n" + "="*60)
        print("          PO FILE TRANSLATION TOOL")
        print("="*60)
        print(f"Languages:    {', '.join(batch_config.language_codes)}")
        print(f"Total:        {len(batch_config.language_codes)} languages")
        print(f"Interface:    {batch_config.translation_config.interface_language}")
        print(f"Source:       {batch_config.translation_config.source_language}")
        print(f"Mode:         {'Multi-process' if batch_config.translation_config.multi_process else 'Single-process'}")
        print(f"Headless:     {batch_config.translation_config.headless}")
        print("="*60 + "\n")

    def _execute_translation(self, batch_config: BatchConfig) -> bool:
        """
        Execute translation based on configuration.

        Args:
            batch_config: Batch configuration

        Returns:
            bool: True if translation completed successfully
        """
        # Start metrics collection
        session_id = self.metrics.start_session(batch_config.language_codes)
        self.logger.get_logger().info(f"Started translation session: {session_id}")

        try:
            if batch_config.translation_config.multi_process:
                return self._execute_multi_process(batch_config)
            else:
                return self._execute_single_process(batch_config)
        finally:
            self.metrics.end_session()

    def _execute_single_process(self, batch_config: BatchConfig) -> bool:
        """
        Execute translation in single process mode.

        Args:
            batch_config: Batch configuration

        Returns:
            bool: True if translation completed successfully
        """
        session = TranslationSession(batch_config)
        session.start()

        for language_code in batch_config.language_codes:
            if not self.is_running:
                break

            success = session.translate_language(language_code)

            # Log progress
            progress = session.progress
            self.logger.get_logger().info(
                f"Progress: {progress:.1f}% - {session.completed_count}/{len(batch_config.language_codes)}"
            )

            # Print health summary every 5 languages
            if session.completed_count % 5 == 0:
                self._print_health_summary()

        session.complete()

        # Log session results
        summary = session.get_summary()
        self.logger.get_logger().info("Translation session completed",
                                    extra={'extra_data': summary})

        return session.failed_count == 0

    def _execute_multi_process(self, batch_config: BatchConfig) -> bool:
        """
        Execute translation in multi-process mode.

        Args:
            batch_config: Batch configuration

        Returns:
            bool: True if translation completed successfully
        """
        # Import here to avoid circular imports
        from main import translate_multi_process

        self.logger.get_logger().info(
            f"Starting multi-process translation with {batch_config.translation_config.max_processes} processes"
        )

        try:
            translate_multi_process(batch_config)
            return True
        except Exception as e:
            self.logger.get_logger().error(f"Multi-process translation failed: {e}")
            return False

    def _print_health_summary(self):
        """Print system health summary."""
        health_summary = self.health_monitor.get_health_summary()
        if health_summary:
            cpu_msg = f"CPU: {health_summary['current_cpu_percent']:.1f}%"
            memory_msg = f"Memory: {health_summary['current_memory_percent']:.1f}%"
            print(f"[Health] {cpu_msg} | {memory_msg}")

    def _log_completion_info(self, success: bool):
        """
        Log application completion information.

        Args:
            success: Whether application completed successfully
        """
        # Metrics summary
        metrics_summary = self.metrics.get_session_summary()
        historical_stats = self.metrics.get_historical_stats()

        # Health summary
        health_summary = self.health_monitor.get_health_summary()

        completion_data = {
            'success': success,
            'session_metrics': metrics_summary,
            'historical_stats': historical_stats,
            'health_summary': health_summary
        }

        if success:
            self.logger.get_logger().info("Application completed successfully",
                                        extra={'extra_data': completion_data})
        else:
            self.logger.get_logger().error("Application completed with errors",
                                         extra={'extra_data': completion_data})

        # Print summary to console
        print("\n" + "="*60)
        print("                 TRANSLATION SUMMARY")
        print("="*60)

        if metrics_summary:
            print(f"Languages:     {metrics_summary['successful']} successful, "
                  f"{metrics_summary['failed']} failed")
            print(f"Success Rate:  {metrics_summary['success_rate_percent']}%")
            print(f"Duration:      {metrics_summary['duration_seconds']:.1f} seconds")

            if metrics_summary['average_speed_cps']:
                print(f"Speed:         {metrics_summary['average_speed_cps']} chars/sec")

        if health_summary:
            print(f"Avg CPU:       {health_summary.get('average_cpu_percent', 0):.1f}%")
            print(f"Avg Memory:    {health_summary.get('average_memory_percent', 0):.1f}%")

        print("="*60)

    def _cleanup(self):
        """Cleanup resources."""
        self.health_monitor.stop_monitoring()
        self.is_running = False


def main():
    """Main application entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()

    # Create and run application
    app = TranslationApplication()
    success = app.run(args)

    # Exit with appropriate code
    sys.exit(0 if success else 1)


def create_argument_parser() -> argparse.ArgumentParser:
    """
    Create argument parser with all options.

    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="Advanced PO File Translation Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with specific languages
  python main.py -l de fr es -d /path/to/chromedriver -p /path/to/locale
  
  # Use configuration file
  python main.py -c config.json
  
  # Multi-process with custom settings
  python main.py -l de fr ja -d /path/to/chromedriver -p /path/to/locale -m --max-processes 5
  
  # With advanced logging
  python main.py -c config.json --log-level DEBUG --log-dir /var/log/translator
        """
    )

    # Input options
    input_group = parser.add_argument_group('Input Options')
    input_group.add_argument(
        '-l', '--languages',
        nargs='+',
        help='Language codes to translate (e.g., de fr es)'
    )
    input_group.add_argument(
        '--language-file',
        help='File containing list of language codes (one per line)'
    )
    input_group.add_argument(
        '-c', '--config',
        help='JSON configuration file path'
    )

    # Required path options (if no config file)
    path_group = parser.add_argument_group('Path Options')
    path_group.add_argument(
        '-d', '--driver-path',
        help='Path to ChromeDriver executable'
    )
    path_group.add_argument(
        '-p', '--locale-path',
        help='Path to locale directory'
    )

    # Translation options
    translation_group = parser.add_argument_group('Translation Options')
    translation_group.add_argument(
        '-i', '--interface-language',
        default='en',
        help='Google Translate interface language (default: en)'
    )
    translation_group.add_argument(
        '-s', '--source-language',
        default='en',
        help='Source language for translation (default: en)'
    )
    translation_group.add_argument(
        '-r', '--max-retries',
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
        '-m', '--multi-process',
        action='store_true',
        help='Use multiprocessing for parallel translation'
    )
    execution_group.add_argument(
        '--max-processes',
        type=int,
        default=