"""
Manages translation sessions with state tracking and progress monitoring.
"""

import time
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

from config import BatchConfig, LanguageConfig
from translation_orchestrator import TranslationOrchestrator


class TranslationStatus(Enum):
    """Status of translation operation."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class TranslationResult:
    """Result of a single language translation."""
    language_code: str
    status: TranslationStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    file_path: Optional[str] = None
    changes_made: bool = False

    @property
    def duration(self) -> Optional[float]:
        """Get translation duration in seconds."""
        if self.end_time and self.start_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

    def mark_completed(self, changes_made: bool = False):
        """Mark translation as completed."""
        self.status = TranslationStatus.COMPLETED
        self.end_time = datetime.now()
        self.changes_made = changes_made

    def mark_failed(self, error: str):
        """Mark translation as failed."""
        self.status = TranslationStatus.FAILED
        self.end_time = datetime.now()
        self.error_message = error


class TranslationSession:
    """
    Manages a complete translation session with progress tracking.
    """

    def __init__(self, batch_config: BatchConfig):
        """
        Initialize translation session.

        Args:
            batch_config: Batch configuration for translation
        """
        self.batch_config = batch_config
        self.results: Dict[str, TranslationResult] = {}
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None

        # Initialize results for all languages
        for lang_config in batch_config.get_language_configs():
            self.results[lang_config.language_code] = TranslationResult(
                language_code=lang_config.language_code,
                status=TranslationStatus.PENDING,
                start_time=datetime.now(),
                file_path=lang_config.po_file_path
            )

    def start(self):
        """Start the translation session."""
        self.start_time = datetime.now()
        print(f"[+] Translation session started at {self.start_time}")
        print(f"[+] Processing {len(self.results)} languages")

    def complete(self):
        """Mark session as completed."""
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds() if self.start_time else 0

        successful = self.successful_count
        failed = self.failed_count

        print(f"[+] Translation session completed at {self.end_time}")
        print(f"[+] Results: {successful} successful, {failed} failed, {self.skipped_count} skipped")
        print(f"[+] Total duration: {duration:.2f} seconds")

    def translate_language(self, language_code: str) -> bool:
        """
        Translate a single language and track result.

        Args:
            language_code: Language code to translate

        Returns:
            bool: True if translation was successful
        """
        result = self.results[language_code]
        result.status = TranslationStatus.IN_PROGRESS

        print(f"[+] Translating {language_code}...")

        try:
            # Create orchestrator and translate
            with TranslationOrchestrator(self.batch_config.translation_config) as orchestrator:
                success = orchestrator.translate_language(language_code)

            if success:
                result.mark_completed()
                print(f"[+] Completed translation for {language_code}")
            else:
                result.mark_failed("Translation failed without specific error")
                print(f"[-] Failed translation for {language_code}")

            return success

        except Exception as error:
            result.mark_failed(str(error))
            print(f"[-] Error translating {language_code}: {error}")
            return False

    @property
    def completed_count(self) -> int:
        """Get number of completed translations."""
        return len([r for r in self.results.values()
                    if r.status == TranslationStatus.COMPLETED])

    @property
    def successful_count(self) -> int:
        """Get number of successful translations."""
        return len([r for r in self.results.values()
                    if r.status == TranslationStatus.COMPLETED and r.changes_made])

    @property
    def failed_count(self) -> int:
        """Get number of failed translations."""
        return len([r for r in self.results.values()
                    if r.status == TranslationStatus.FAILED])

    @property
    def skipped_count(self) -> int:
        """Get number of skipped translations."""
        return len([r for r in self.results.values()
                    if r.status == TranslationStatus.SKIPPED])

    @property
    def progress(self) -> float:
        """Get session progress as percentage."""
        total = len(self.results)
        if total == 0:
            return 100.0

        completed = self.completed_count + self.failed_count + self.skipped_count
        return (completed / total) * 100

    def get_summary(self) -> Dict[str, Any]:
        """Get session summary."""
        return {
            "total_languages": len(self.results),
            "successful": self.successful_count,
            "failed": self.failed_count,
            "skipped": self.skipped_count,
            "progress": self.progress,
            "duration": self.duration,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
        }

    @property
    def duration(self) -> Optional[float]:
        """Get session duration in seconds."""
        if self.end_time and self.start_time:
            return (self.end_time - self.start_time).total_seconds()
        elif self.start_time:
            return (datetime.now() - self.start_time).total_seconds()
        return None

    def print_progress(self):
        """Print current progress."""
        progress = self.progress
        print(f"[Progress] {progress:.1f}% - {self.completed_count}/{len(self.results)} languages")

        if self.failed_count > 0:
            failed_codes = [code for code, result in self.results.items()
                            if result.status == TranslationStatus.FAILED]
            print(f"[!] Failed languages: {', '.join(failed_codes)}")