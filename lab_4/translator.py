"""
Main translation functionality using Selenium WebDriver.
Integrates all components for PO file translation.
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import NoSuchElementException
from multiprocessing import Pool, freeze_support
from itertools import repeat
from fake_useragent import UserAgent

from constants import POFileConstants, TranslatorConstants, LogMessages
from po_parser import POParser
from text_processor import TextProcessor
from utils import random_pause, format_translation_url


def translate(driver: webdriver.Chrome, text: str, last_translation: str, max_retries: int) -> str:
    """
    Translate text using Google Translate via Selenium WebDriver.

    Args:
        driver: Selenium WebDriver instance
        text: Text to translate
        last_translation: Previous translation for comparison
        max_retries: Maximum number of retry attempts

    Returns:
        str: Translated text or empty string if translation failed
    """
    # Preprocess text for translation
    processed_text, variables, html_classes, has_service_chars, has_unnamed_format = (
        TextProcessor.preprocess_text(text)
    )

    def get_translated_text() -> str:
        """
        Extract translated text from Google Translate interface.

        Returns:
            str: Extracted translation text
        """
        random_pause()

        try:
            output_elements = driver.find_elements(By.CSS_SELECTOR, TranslatorConstants.PRIMARY_OUTPUT_SELECTOR)
        except NoSuchElementException:
            driver.refresh()
            output_elements = driver.find_elements(By.CSS_SELECTOR, TranslatorConstants.PRIMARY_OUTPUT_SELECTOR)

        # Handle multiple translations (some languages show alternatives)
        if not output_elements:
            output_elements = driver.find_elements(By.CSS_SELECTOR, TranslatorConstants.FALLBACK_OUTPUT_SELECTOR)[-1:]

        translation_parts = [element.text for element in output_elements]
        return ''.join(translation_parts)

    try:
        # Input text and get translation
        input_field = driver.find_element(By.CSS_SELECTOR, TranslatorConstants.INPUT_SELECTOR)
        input_field.clear()
        input_field.send_keys(processed_text)

        translation_result = get_translated_text()

        # Retry if translation is same as previous (possible delay)
        if translation_result == last_translation:
            translation_result = get_translated_text()

        # Postprocess translation to restore original patterns
        final_translation = TextProcessor.postprocess_text(
            translation_result, variables, html_classes, has_service_chars, has_unnamed_format
        )

        print(LogMessages.TRANSLATION_SUCCESS.format(source=text, translation=final_translation))
        return final_translation

    except Exception as error:
        if max_retries > 0:
            print(LogMessages.TRANSLATION_RETRY.format(text=text, retry=max_retries, error=error))
            return translate(driver, text, last_translation, max_retries - 1)
        else:
            print(LogMessages.TRANSLATION_FAILED.format(text=text))
            return ""


def translator(
        language_code: str,
        driver_path: str,
        locale_path: str,
        headless: bool = True,
        interface_language: str = 'en',
        source_language: str = 'en',
        max_retries: int = 3
) -> None:
    """
    Translate PO file for a specific language.

    Args:
        language_code: Target language code (e.g., 'de', 'fr')
        driver_path: Path to ChromeDriver executable
        locale_path: Path to locale directory containing PO files
        headless: Whether to run browser in headless mode
        interface_language: Google Translate interface language
        source_language: Source language for translation
        max_retries: Maximum translation retry attempts
    """
    # Setup WebDriver
    user_agent = UserAgent(verify_ssl=False)
    service = Service(executable_path=driver_path)
    options = webdriver.ChromeOptions()
    options.add_argument(f'user-agent={user_agent.random}')
    options.add_argument('--disable-blink-features=AutomationControlled')

    if headless:
        options.add_argument('--headless')

    driver = webdriver.Chrome(service=service, options=options)

    try:
        # Navigate to Google Translate
        url = format_translation_url(interface_language, source_language, language_code)
        driver.get(url)

        # Process PO file
        po_file_path = f'{locale_path}/{language_code}/LC_MESSAGES/django.po'
        _process_po_file(driver, po_file_path, max_retries)

    finally:
        driver.quit()


def _process_po_file(driver: webdriver.Chrome, file_path: str, max_retries: int) -> None:
    """
    Process and translate strings in a PO file.

    Args:
        driver: WebDriver instance
        file_path: Path to PO file
        max_retries: Maximum translation retry attempts
    """
    try:
        with open(file_path, 'r', encoding='UTF-8') as file:
            print(LogMessages.FILE_OPENED.format(path=file_path))
            file_content = file.read()
            translated_content = _translate_po_content(driver, file_content, max_retries)

        # Save if changes were made
        if translated_content != file_content:
            with open(file_path, 'w', encoding='UTF-8') as file:
                file.write(translated_content)
            print(LogMessages.FILE_SAVED.format(path=file_path))
        else:
            print(LogMessages.NO_CHANGES.format(path=file_path))

    except FileNotFoundError:
        print(LogMessages.FILE_NOT_FOUND.format(path=file_path))


def _translate_po_content(driver: webdriver.Chrome, content: str, max_retries: int) -> str:
    """
    Translate untranslated strings in PO file content.

    Args:
        driver: WebDriver instance
        content: PO file content as string
        max_retries: Maximum translation retry attempts

    Returns:
        str: Translated content
    """
    lines = content.splitlines(True)
    result_lines = []

    current_text = ''
    is_translating = False
    is_complex_text = False
    should_save_complex = False
    complex_text_parts = []
    current_translation = None
    last_translation = None

    for i, line in enumerate(lines):
        # Skip already translated strings
        if POParser.is_translated(lines, i, line):
            result_lines.append(line)
            continue

        if line.startswith(POFileConstants.MSGID_PREFIX):
            # Start new translation unit
            text_content = POParser.extract_text_content(line)

            if text_content:  # Simple string translation
                current_translation = translate(
                    driver, text_content, last_translation, max_retries
                )
                last_translation = current_translation
                is_translating = True
            else:  # Complex multi-line string
                is_complex_text = True

            result_lines.append(line)

        elif line.startswith(POFileConstants.QUOTE) and is_complex_text and len(line) > 2:
            # Collect complex text parts
            text_content = POParser.extract_text_content(line)
            complex_text_parts.append(text_content)
            result_lines.append(line)

            # Check if next line starts msgstr (end of complex text)
            try:
                if lines[i + 1].startswith(POFileConstants.MSGSTR_PREFIX):
                    is_complex_text = False
                    should_save_complex = True
                    is_translating = True
            except IndexError:
                print(LogMessages.SYNTAX_ERROR.format(line=i + 1, file="PO file"))

        elif line.startswith(POFileConstants.MSGSTR_PREFIX) and is_translating:
            # Write translation
            if should_save_complex:
                full_text = ' '.join(complex_text_parts)
                translation = translate(driver, full_text, last_translation, max_retries)
                last_translation = translation
                result_lines.append(f'msgstr ""\n"{translation}"\n')
                should_save_complex = False
                complex_text_parts.clear()
            else:
                result_lines.append(f'msgstr "{current_translation}"\n')

            is_translating = False
            current_translation = None
        else:
            result_lines.append(line)

    return ''.join(result_lines)


# Manager function remains similar but uses new constants
def manager(
        language_codes: list,
        driver_path: str,
        locale_path: str,
        headless: bool = True,
        multi_process: bool = False,
        max_processes: int = 10,
        interface_language: str = 'en',
        source_language: str = 'en',
        max_retries: int = 3
) -> None:
    """
    Manage translation process for multiple languages.

    Args:
        language_codes: List of language codes to translate
        driver_path: Path to ChromeDriver executable
        locale_path: Path to locale directory
        headless: Whether to run browser in headless mode
        multi_process: Whether to use multiprocessing
        max_processes: Maximum number of parallel processes
        interface_language: Google Translate interface language
        source_language: Source language for translation
        max_retries: Maximum translation retry attempts
    """
    if multi_process:
        freeze_support()
        codes_to_process = language_codes.copy()

        while codes_to_process:
            current_batch = codes_to_process[:max_processes]
            codes_to_process = codes_to_process[max_processes:]

            with Pool(processes=len(current_batch)) as pool:
                pool.starmap(translator, zip(
                    current_batch,
                    repeat(driver_path),
                    repeat(locale_path),
                    repeat(headless),
                    repeat(interface_language),
                    repeat(source_language),
                    repeat(max_retries)
                ))
    else:
        for code in language_codes:
            translator(code, driver_path, locale_path, headless,
                       interface_language, source_language, max_retries)


if __name__ == '__main__':
    manager(
        language_codes=['de', 'fr', 'ja', 'tr', 'ru', 'uk'],
        driver_path='/DJTranslator/chromedriver',
        locale_path='/DJAuth/locale',
        multi_process=True,
        interface_language='ru',
    )