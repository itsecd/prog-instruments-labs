"""
Configuration management for translation application.
Uses dataclasses for type-safe configuration.
"""

from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
import json
import os


@dataclass
class TranslationConfig:
    """
    Main configuration for translation operations.
    """
    driver_path: str
    locale_path: str
    headless: bool = True
    interface_language: str = 'en'
    source_language: str = 'en'
    max_retries: int = 3
    multi_process: bool = False
    max_processes: int = 10
    timeout_per_translation: int = 30
    user_agent_rotation: bool = True

    def validate(self) -> None:
        """
        Validate configuration parameters.

        Raises:
            ValueError: If any configuration parameter is invalid
        """
        if not os.path.exists(self.driver_path):
            raise ValueError(f"Driver path does not exist: {self.driver_path}")

        if not os.path.exists(self.locale_path):
            raise ValueError(f"Locale path does not exist: {self.locale_path}")

        if self.max_retries < 0:
            raise ValueError("Max retries cannot be negative")

        if self.max_processes < 1:
            raise ValueError("Max processes must be at least 1")

        if self.timeout_per_translation < 5:
            raise ValueError("Timeout must be at least 5 seconds")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TranslationConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)

    @classmethod
    def from_json_file(cls, file_path: str) -> 'TranslationConfig':
        """
        Load configuration from JSON file.

        Args:
            file_path: Path to JSON configuration file

        Returns:
            TranslationConfig: Loaded configuration

        Raises:
            FileNotFoundError: If configuration file doesn't exist
            ValueError: If configuration is invalid
        """
        with open(file_path, 'r', encoding='utf-8') as file:
            config_dict = json.load(file)

        config = cls.from_dict(config_dict)
        config.validate()
        return config

    def save_to_json(self, file_path: str):
        """
        Save configuration to JSON file.

        Args:
            file_path: Path to save configuration
        """
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(self.to_dict(), file, indent=2, ensure_ascii=False)


@dataclass
class LanguageConfig:
    """
    Configuration for specific language translation.
    """
    language_code: str
    config: TranslationConfig

    @property
    def po_file_path(self) -> str:
        """Get full path to PO file for this language."""
        return os.path.join(
            self.config.locale_path,
            self.language_code,
            'LC_MESSAGES',
            'django.po'
        )

    def validate(self) -> None:
        """Validate language-specific configuration."""
        if not self.language_code or len(self.language_code) != 2:
            raise ValueError(f"Invalid language code: {self.language_code}")


@dataclass
class BatchConfig:
    """
    Configuration for batch translation operations.
    """
    language_codes: List[str]
    translation_config: TranslationConfig

    def validate(self) -> None:
        """Validate batch configuration."""
        if not self.language_codes:
            raise ValueError("Language codes list cannot be empty")

        for code in self.language_codes:
            if len(code) != 2:
                raise ValueError(f"Invalid language code: {code}")

        self.translation_config.validate()

    def get_language_configs(self) -> List[LanguageConfig]:
        """Get language configurations for all codes in batch."""
        return [
            LanguageConfig(code, self.translation_config)
            for code in self.language_codes
        ]