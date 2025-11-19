"""
Advanced logging configuration with structured logging and file handlers.
"""

import logging
import logging.handlers
import sys
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path


class StructuredFormatter(logging.Formatter):
    """
    Formatter for structured JSON logging.
    """

    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as structured JSON.

        Args:
            record: Log record to format

        Returns:
            str: JSON formatted log entry
        """
        log_entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }

        # Add extra fields if present
        if hasattr(record, 'extra_data'):
            log_entry.update(record.extra_data)

        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)

        return json.dumps(log_entry, ensure_ascii=False)


class ColorFormatter(logging.Formatter):
    """
    Color formatter for console output.
    """

    COLORS = {
        'DEBUG': '\033[36m',  # Cyan
        'INFO': '\033[32m',  # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',  # Red
        'CRITICAL': '\033[41m',  # Red background
        'RESET': '\033[0m'  # Reset
    }

    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record with colors.

        Args:
            record: Log record to format

        Returns:
            str: Colored log message
        """
        log_message = super().format(record)
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        return f"{color}{log_message}{self.COLORS['RESET']}"


class TranslationLogger:
    """
    Advanced logger for translation application with multiple handlers.
    """

    def __init__(self, name: str = "translator", log_level: str = "INFO",
                 log_dir: str = "logs", enable_file_logging: bool = True):
        """
        Initialize translation logger.

        Args:
            name: Logger name
            log_level: Logging level
            log_dir: Directory for log files
            enable_file_logging: Whether to enable file logging
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))

        # Avoid duplicate handlers
        if not self.logger.handlers:
            self._setup_handlers(log_dir, enable_file_logging)

    def _setup_handlers(self, log_dir: str, enable_file_logging: bool):
        """
        Setup console and file handlers.

        Args:
            log_dir: Directory for log files
            enable_file_logging: Whether to enable file logging
        """
        # Create log directory if it doesn't exist
        if enable_file_logging:
            Path(log_dir).mkdir(exist_ok=True)

        # Console handler with colors
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = ColorFormatter(
            '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        if enable_file_logging:
            # File handler with structured JSON
            log_file = Path(log_dir) / f"translation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_formatter = StructuredFormatter()
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)

            # Rotating file handler for detailed logs
            debug_log_file = Path(log_dir) / "translation_debug.log"
            rotating_handler = logging.handlers.RotatingFileHandler(
                debug_log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding='utf-8'
            )
            debug_formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(name)s - %(module)s:%(lineno)d - %(message)s'
            )
            rotating_handler.setFormatter(debug_formatter)
            rotating_handler.setLevel(logging.DEBUG)
            self.logger.addHandler(rotating_handler)

    def log_translation_start(self, language_count: int, config: Dict[str, Any]):
        """
        Log translation session start.

        Args:
            language_count: Number of languages to translate
            config: Translation configuration
        """
        self.logger.info(
            "Translation session started",
            extra={'extra_data': {
                'languages_count': language_count,
                'config': config,
                'session_id': datetime.now().strftime('%Y%m%d_%H%M%S')
            }}
        )

    def log_translation_success(self, language_code: str, source_text: str,
                                translated_text: str, duration: float):
        """
        Log successful translation.

        Args:
            language_code: Target language code
            source_text: Original text
            translated_text: Translated text
            duration: Translation duration in seconds
        """
        self.logger.info(
            f"Translated to {language_code}",
            extra={'extra_data': {
                'language': language_code,
                'source_length': len(source_text),
                'translation_length': len(translated_text),
                'duration_seconds': round(duration, 2)
            }}
        )

    def log_translation_error(self, language_code: str, error: str,
                              retry_count: int = 0):
        """
        Log translation error.

        Args:
            language_code: Target language code
            error: Error message
            retry_count: Number of retry attempts
        """
        self.logger.error(
            f"Translation failed for {language_code}",
            extra={'extra_data': {
                'language': language_code,
                'error': error,
                'retry_count': retry_count
            }}
        )

    def log_file_operation(self, operation: str, file_path: str,
                           success: bool, details: Optional[Dict] = None):
        """
        Log file operation.

        Args:
            operation: Type of operation (read, write, etc.)
            file_path: File path
            success: Whether operation was successful
            details: Additional details
        """
        level = logging.INFO if success else logging.ERROR
        extra_data = {
            'operation': operation,
            'file_path': file_path,
            'success': success
        }

        if details:
            extra_data.update(details)

        self.logger.log(
            level,
            f"File {operation} {'succeeded' if success else 'failed'}",
            extra={'extra_data': extra_data}
        )

    def get_logger(self) -> logging.Logger:
        """
        Get the underlying logger instance.

        Returns:
            logging.Logger: Logger instance
        """
        return self.logger


# Global logger instance
_default_logger: Optional[TranslationLogger] = None


def setup_logging(log_level: str = "INFO", log_dir: str = "logs",
                  enable_file_logging: bool = True) -> TranslationLogger:
    """
    Setup and return global logger instance.

    Args:
        log_level: Logging level
        log_dir: Directory for log files
        enable_file_logging: Whether to enable file logging

    Returns:
        TranslationLogger: Configured logger instance
    """
    global _default_logger
    if _default_logger is None:
        _default_logger = TranslationLogger(
            log_level=log_level,
            log_dir=log_dir,
            enable_file_logging=enable_file_logging
        )
    return _default_logger


def get_logger() -> TranslationLogger:
    """
    Get global logger instance.

    Returns:
        TranslationLogger: Logger instance

    Raises:
        RuntimeError: If logger not setup
    """
    if _default_logger is None:
        raise RuntimeError("Logger not setup. Call setup_logging first.")
    return _default_logger