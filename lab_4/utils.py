"""
Utility functions for the translation application.
"""

import random
import time
from constants import TimeConstants


def random_pause() -> None:
    """
    Pause execution for a random duration between MIN_PAUSE and MAX_PAUSE.

    This helps avoid being detected as a bot and adds human-like behavior.
    """
    pause_duration = round(
        random.uniform(TimeConstants.MIN_PAUSE, TimeConstants.MAX_PAUSE),
        TimeConstants.PAUSE_DECIMALS
    )
    time.sleep(pause_duration)


def create_substitution_pattern(index: int) -> str:
    """
    Create a substitution pattern for temporary text replacement.

    Args:
        index: Pattern index

    Returns:
        str: Substitution pattern
    """
    from constants import TranslatorConstants
    return TranslatorConstants.SUBSTITUTION_PATTERN % (index,)


def format_translation_url(interface_lang: str, source_lang: str, target_lang: str) -> str:
    """
    Format Google Translate URL with specified language parameters.

    Args:
        interface_lang: Interface language code
        source_lang: Source language code
        target_lang: Target language code

    Returns:
        str: Formatted URL
    """
    from constants import TranslatorConstants
    return TranslatorConstants.TRANSLATE_URL % {
        'lang_interface': interface_lang,
        'from_lang': source_lang,
        'to_lang': target_lang
    }