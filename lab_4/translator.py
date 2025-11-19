"""
Main translation coordinator integrating all components.
"""

from multiprocessing import Pool, freeze_support
from itertools import repeat

from translation_driver import TranslationDriver
from translation_service import TranslationService
from po_file_processor import POFileProcessor
from file_manager import FileManager


def translate_language(
        language_code: str,
        driver_path: str,
        locale_path: str,
        headless: bool = True,
        interface_language: str = 'en',
        source_language: str = 'en',
        max_retries: int = 3
) -> None:
    """
    Translate PO file for a specific language using the new architecture.

    Args:
        language_code: Target language code
        driver_path: Path to ChromeDriver executable
        locale_path: Path to locale directory
        headless: Whether to run browser in headless mode
        interface_language: Google Translate interface language
        source_language: Source language for translation
        max_retries: Maximum translation retry attempts
    """
    # Setup translation pipeline
    driver = TranslationDriver(driver_path, headless)

    try:
        # Initialize components
        driver.navigate_to_translator(interface_language, source_language, language_code)
        translation_service = TranslationService(driver, max_retries)
        file_processor = POFileProcessor(translation_service)

        # Process PO file
        po_file_path = FileManager.get_po_file_path(locale_path, language_code)

        try:
            original_content = FileManager.read_po_file(po_file_path)
            translated_content = file_processor.process_file_content(original_content)

            # Save if changes were made
            if FileManager.file_has_changes(original_content, translated_content):
                FileManager.write_po_file(po_file_path, translated_content)
            else:
                FileManager.log_no_changes(po_file_path)

        except FileNotFoundError:
            # Error already logged by FileManager
            pass

    finally:
        driver.close()


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
                pool.starmap(translate_language, zip(
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
            translate_language(
                language_code=code,
                driver_path=driver_path,
                locale_path=locale_path,
                headless=headless,
                interface_language=interface_language,
                source_language=source_language,
                max_retries=max_retries
            )


if __name__ == '__main__':
    manager(
        language_codes=['de', 'fr', 'ja', 'tr', 'ru', 'uk'],
        driver_path='/DJTranslator/chromedriver',
        locale_path='/DJAuth/locale',
        multi_process=True,
        interface_language='ru',
    )