"""
Metrics collection and performance monitoring for translation operations.
"""

import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from contextlib import contextmanager
import statistics


@dataclass
class TranslationMetrics:
    """Metrics for a single translation operation."""
    language_code: str
    start_time: datetime
    end_time: Optional[datetime] = None
    source_text_length: int = 0
    translated_text_length: int = 0
    success: bool = False
    error_message: Optional[str] = None
    retry_count: int = 0

    @property
    def duration(self) -> Optional[float]:
        """Get translation duration in seconds."""
        if self.end_time and self.start_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

    @property
    def characters_per_second(self) -> Optional[float]:
        """Get translation speed in characters per second."""
        if self.duration and self.duration > 0:
            return self.source_text_length / self.duration
        return None


@dataclass
class SessionMetrics:
    """Metrics for a complete translation session."""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_languages: int = 0
    successful_translations: int = 0
    failed_translations: int = 0
    total_characters_translated: int = 0
    individual_metrics: List[TranslationMetrics] = field(default_factory=list)

    @property
    def duration(self) -> Optional[float]:
        """Get session duration in seconds."""
        if self.end_time and self.start_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

    @property
    def success_rate(self) -> float:
        """Get success rate as percentage."""
        if self.total_languages == 0:
            return 0.0
        return (self.successful_translations / self.total_languages) * 100

    @property
    def average_speed(self) -> Optional[float]:
        """Get average translation speed in characters per second."""
        successful_metrics = [m for m in self.individual_metrics if m.success]
        if not successful_metrics:
            return None

        speeds = [m.characters_per_second for m in successful_metrics
                  if m.characters_per_second is not None]
        return statistics.mean(speeds) if speeds else None

    def get_summary(self) -> Dict[str, Any]:
        """Get session summary."""
        return {
            'session_id': self.session_id,
            'duration_seconds': self.duration,
            'total_languages': self.total_languages,
            'successful': self.successful_translations,
            'failed': self.failed_translations,
            'success_rate_percent': round(self.success_rate, 2),
            'total_characters': self.total_characters_translated,
            'average_speed_cps': round(self.average_speed, 2) if self.average_speed else None,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None
        }


class MetricsCollector:
    """
    Collects and analyzes metrics for translation operations.
    """

    def __init__(self):
        """Initialize metrics collector."""
        self.current_session: Optional[SessionMetrics] = None
        self.sessions: List[SessionMetrics] = []

    def start_session(self, language_codes: List[str]) -> str:
        """
        Start a new translation session.

        Args:
            language_codes: List of language codes in session

        Returns:
            str: Session ID
        """
        session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.current_session = SessionMetrics(
            session_id=session_id,
            start_time=datetime.now(),
            total_languages=len(language_codes)
        )
        return session_id

    def end_session(self):
        """End current translation session."""
        if self.current_session:
            self.current_session.end_time = datetime.now()
            self.sessions.append(self.current_session)
            self.current_session = None

    @contextmanager
    def track_translation(self, language_code: str, source_text: str):
        """
        Context manager to track translation metrics.

        Args:
            language_code: Language being translated to
            source_text: Text being translated
        """
        if not self.current_session:
            raise RuntimeError("No active session. Call start_session first.")

        metrics = TranslationMetrics(
            language_code=language_code,
            start_time=datetime.now(),
            source_text_length=len(source_text)
        )

        try:
            yield metrics
            metrics.success = True
        except Exception as e:
            metrics.success = False
            metrics.error_message = str(e)
            raise
        finally:
            metrics.end_time = datetime.now()
            self.current_session.individual_metrics.append(metrics)

            if metrics.success:
                self.current_session.successful_translations += 1
                self.current_session.total_characters_translated += metrics.source_text_length
            else:
                self.current_session.failed_translations += 1

    def get_session_summary(self) -> Optional[Dict[str, Any]]:
        """
        Get current session summary.

        Returns:
            Optional[Dict]: Session summary or None if no active session
        """
        if self.current_session:
            return self.current_session.get_summary()
        return None

    def get_historical_stats(self) -> Dict[str, Any]:
        """
        Get historical statistics across all sessions.

        Returns:
            Dict: Historical statistics
        """
        if not self.sessions:
            return {}

        total_sessions = len(self.sessions)
        total_translations = sum(s.successful_translations + s.failed_translations
                                 for s in self.sessions)
        total_successful = sum(s.successful_translations for s in self.sessions)
        total_characters = sum(s.total_characters_translated for s in self.sessions)

        durations = [s.duration for s in self.sessions if s.duration]
        avg_duration = statistics.mean(durations) if durations else 0

        success_rates = [s.success_rate for s in self.sessions]
        avg_success_rate = statistics.mean(success_rates) if success_rates else 0

        return {
            'total_sessions': total_sessions,
            'total_translations': total_translations,
            'total_successful': total_successful,
            'overall_success_rate': round((total_successful / total_translations * 100)
                                          if total_translations else 0, 2),
            'total_characters_translated': total_characters,
            'average_session_duration_seconds': round(avg_duration, 2),
            'average_success_rate_percent': round(avg_success_rate, 2)
        }


# Global metrics collector
_metrics_collector = MetricsCollector()


def get_metrics_collector() -> MetricsCollector:
    """
    Get global metrics collector instance.

    Returns:
        MetricsCollector: Metrics collector instance
    """
    return _metrics_collector