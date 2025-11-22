"""
Драйвер для автоматизации Google Translate.
"""
"""
Selenium WebDriver management and translation automation.
Handles browser setup, navigation, and translation operations.
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import NoSuchElementException
from fake_useragent import UserAgent

from constants import TranslatorConstants, TimeConstants
from utils import random_pause


class TranslationDriver:
    """
    Manages browser automation for Google Translate operations.
    Handles driver setup, navigation, and text translation.
    """

    def __init__(self, driver_path: str, headless: bool = True):
        """
        Initialize translation driver.

        Args:
            driver_path: Path to ChromeDriver executable
            headless: Whether to run browser in headless mode
        """
        self.driver_path = driver_path
        self.headless = headless
        self.driver = None
        self.last_translation = None

    def start(self):
        """Start the WebDriver and configure browser options."""
        service = Service(executable_path=self.driver_path)
        options = self._create_browser_options()
        self.driver = webdriver.Chrome(service=service, options=options)

    def _create_browser_options(self) -> webdriver.ChromeOptions:
        """Create and configure Chrome browser options."""
        options = webdriver.ChromeOptions()
        user_agent = UserAgent(verify_ssl=False).random

        options.add_argument(f'user-agent={user_agent}')
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_experimental_option('excludeSwitches', ['enable-automation'])
        options.add_experimental_option('useAutomationExtension', False)

        if self.headless:
            options.add_argument('--headless')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')

        return options

    def navigate_to_translator(self, interface_language: str, source_language: str, target_language: str):
        """
        Navigate to Google Translate with specified language settings.

        Args:
            interface_language: Google Translate interface language
            source_language: Source language for translation
            target_language: Target language for translation
        """
        from utils import format_translation_url

        if not self.driver:
            self.start()

        url = format_translation_url(interface_language, source_language, target_language)
        self.driver.get(url)

        # Wait for page to load
        random_pause()
        try:
            self.driver.find_element(By.CSS_SELECTOR, 'textarea.er8xn')
            print(f"[+] Переводчик готов для {target_language}")
        except:
            print(f"[!] Ошибка загрузки переводчика для {target_language}")

    def translate_text(self, text: str, max_retries: int = 3) -> str:
        """
        Translate text using Google Translate.

        Args:
            text: Text to translate
            max_retries: Maximum number of retry attempts

        Returns:
            str: Translated text or empty string if translation failed
        """
        if not self.driver:
            raise RuntimeError("Driver not started. Call navigate_to_translator first.")

        return self._perform_translation_with_retry(text, max_retries)

    def _perform_translation_with_retry(self, text: str, retries_left: int) -> str:
        """
        Perform translation with retry logic for failures.

        Args:
            text: Text to translate
            retries_left: Number of retry attempts remaining

        Returns:
            str: Translated text
        """
        try:
            self._input_text_for_translation(text)
            translation = self._get_translation_output()

            # Handle potential translation delays
            if translation == self.last_translation:
                translation = self._get_translation_output()

            self.last_translation = translation
            return translation

        except Exception as error:
            if retries_left > 0:
                return self._perform_translation_with_retry(text, retries_left - 1)
            else:
                from constants import LogMessages
                print(LogMessages.TRANSLATION_FAILED.format(text=text))
                return ""

    def _input_text_for_translation(self, text: str):
        """
        Input text into Google Translate input field.

        Args:
            text: Text to input for translation
        """
        input_field = self.driver.find_element(By.CSS_SELECTOR, TranslatorConstants.INPUT_SELECTOR)
        input_field.clear()
        input_field.send_keys(text)

    def _get_translation_output(self) -> str:
        """
        Extract translation result from Google Translate output field.

        Returns:
            str: Extracted translation text
        """
        random_pause()

        try:
            # Try primary output selector
            output_elements = self.driver.find_elements(
                By.CSS_SELECTOR, TranslatorConstants.PRIMARY_OUTPUT_SELECTOR
            )
        except NoSuchElementException:
            # Refresh and retry if elements not found
            self.driver.refresh()
            random_pause()
            output_elements = self.driver.find_elements(
                By.CSS_SELECTOR, TranslatorConstants.PRIMARY_OUTPUT_SELECTOR
            )

        # Handle cases with multiple translation options
        if not output_elements:
            output_elements = self.driver.find_elements(
                By.CSS_SELECTOR, TranslatorConstants.FALLBACK_OUTPUT_SELECTOR
            )[-1:]  # Take only the last element

        # Combine all translation parts
        translation_parts = [element.text for element in output_elements]
        return ''.join(translation_parts)

    def refresh_page(self):
        """Refresh the current page and wait for load."""
        if self.driver:
            self.driver.refresh()
            random_pause()

    def close(self):
        """Close the WebDriver and cleanup resources."""
        if self.driver:
            self.driver.quit()
            self.driver = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with resource cleanup."""
        self.close()