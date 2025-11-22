"""
Constants for PO file translation application.
Centralizes all magic numbers and string literals.
"""

class POFileConstants:
    """Constants related to PO file format."""
    MSGID_PREFIX = 'msgid "'
    MSGSTR_PREFIX = 'msgstr "'
    QUOTE = '"'
    NEWLINE = '\n'
    EMPTY_MSGID_LENGTH = 9  # len('msgid ""\n')
    EMPTY_MSGSTR_LENGTH = 10  # len('msgstr ""\n')
    ESCAPED_QUOTE = r'\"'
    FORMAT_PYTHON_NAMED = '%('
    FORMAT_PYTHON_UNNAMED = '%s'
    HTML_CLASS_PREFIX = 'class="'


class TranslatorConstants:
    """Constants for translation service."""
    INPUT_SELECTOR = 'textarea.er8xn'
    PRIMARY_OUTPUT_SELECTOR = 'span.ryNqvb'
    FALLBACK_OUTPUT_SELECTOR = 'span.HwtZe'
    SUBSTITUTION_PATTERN = "{%s}"
    TEMP_SUBSTITUTION = "{+}"
    TRANSLATE_URL = 'https://translate.google.com/?hl=%(lang_interface)s&sl=%(from_lang)s&tl=%(to_lang)s&op=translate'


class TimeConstants:
    """Constants for timing and delays."""
    MIN_PAUSE = 1.8
    MAX_PAUSE = 2.1
    PAUSE_DECIMALS = 2


class LogMessages:
    """Standardized log messages."""
    FILE_OPENED = "{path} - opened!"
    FILE_NOT_FOUND = "[!] FAIL {path} doesn't exists"
    FILE_SAVED = "{path} - Saved!"
    NO_CHANGES = "{path} - Without changes!"
    TRANSLATION_SUCCESS = "[+] {source} - {translation}"
    TRANSLATION_RETRY = "[!] FAIL -> {text} | retry={retry} ({error})"
    TRANSLATION_FAILED = "[!] No attempts left for -> {text}"
    SYNTAX_ERROR = "[!] SyntaxError at {line} line in {file}"