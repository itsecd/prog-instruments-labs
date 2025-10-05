import pytest
import json
from unittest.mock import MagicMock, patch, mock_open
from io import BytesIO
from ..app import (
    perform_conversion,
    extract_text_from_pdf,
    extract_text_from_docx,
    convert_image_format,
    setup_logging
)


class TestPerformConversionAdvanced:
    """Продвинутые тесты основной функции конвертации"""

    @pytest.mark.parametrize("input_fmt,output_fmt,expected_content", [
        ("csv", "json", "name"),  # CSV → JSON
        ("csv", "xml", "<root>"),  # CSV → XML
        ("json", "csv", "name,age"),  # JSON → CSV
        ("json", "xml", "<root>"),  # JSON → XML
        ("txt", "pdf", "%PDF"),  # TXT → PDF (бинарный)
    ])
    def test_perform_conversion_parametrized(self, input_fmt, output_fmt, expected_content, sample_csv_content,
                                             sample_json_content):
        """ПАРАМЕТРИЗОВАННЫЙ ТЕСТ: различные комбинации конвертаций"""

        # Выбираем тестовые данные в зависимости от входного формата
        test_content = sample_csv_content if input_fmt == "csv" else sample_json_content

        result = perform_conversion(test_content, input_fmt, output_fmt)

        # Для бинарных форматов проверяем сигнатуру
        if output_fmt == "pdf" and expected_content == "%PDF":
            assert result.startswith(b'%PDF') or expected_content in str(result)
        else:
            assert expected_content in result

    def test_perform_conversion_unsupported_format(self):
        """Тест обработки неподдерживаемого формата"""
        with pytest.raises(ValueError) as exc_info:
            perform_conversion("content", "invalid_format", "json")

        assert "Unsupported input format" in str(exc_info.value)


class TestPDFExtractionWithMocks:
    """Тесты извлечения PDF с использованием МОКОВ"""

    def test_extract_text_from_pdf_success_with_mock(self):
        """МОК-ТЕСТ: успешное извлечение текста из PDF"""
        mock_file = MagicMock()

        # Мокаем pdfplumber
        with patch('app.pdfplumber.open') as mock_pdf_open:
            # Настраиваем мок PDF документа
            mock_pdf = MagicMock()
            mock_page = MagicMock()
            mock_page.extract_text.return_value = "Извлеченный текст из PDF"
            mock_pdf.pages = [mock_page]
            mock_pdf_open.return_value.__enter__.return_value = mock_pdf

            # Вызываем тестируемую функцию
            result = extract_text_from_pdf(mock_file)

            # Проверяем результат
            assert "Извлеченный текст из PDF" in result
            # Проверяем что мок был вызван
            mock_pdf_open.assert_called_once()

    def test_extract_text_from_pdf_fallback_with_mocks(self):
        """МОК-ТЕСТ: тест fallback механизма при ошибке PDF"""
        mock_file = MagicMock()

        # Мокаем pdfplumber чтобы он бросал исключение
        with patch('app.pdfplumber.open', side_effect=Exception("PDF error")):
            # Мокаем PyMuPDF (fitz) для fallback
            with patch('app.fitz') as mock_fitz:
                mock_doc = MagicMock()
                mock_page = MagicMock()
                mock_page.get_text.return_value = "Fallback текст"
                mock_doc.__enter__.return_value = [mock_page]
                mock_fitz.open.return_value = mock_doc

                # Вызываем тестируемую функцию
                result = extract_text_from_pdf(mock_file)

                # Проверяем что использовался fallback
                assert "Fallback текст" in result
                mock_fitz.open.assert_called_once()


class TestImageConversionWithMocks:
    """Тесты конвертации изображений с МОКАМИ"""

    def test_convert_image_format_with_mock(self):
        """МОК-ТЕСТ: конвертация формата изображения"""
        mock_file = MagicMock()

        with patch('app.Image.open') as mock_image_open:
            with patch('app.BytesIO') as mock_bytes_io:
                # Настраиваем моки
                mock_img = MagicMock()
                mock_img.format = 'PNG'
                mock_img.mode = 'RGB'
                mock_img.size = (100, 100)
                mock_image_open.return_value = mock_img

                mock_buffer = MagicMock()
                mock_buffer.getvalue.return_value = b'converted_image_data'
                mock_bytes_io.return_value = mock_buffer

                # Вызываем тестируемую функцию
                result = convert_image_format(mock_file, 'JPEG')

                # Проверяем что методы были вызваны правильно
                mock_image_open.assert_called_once_with(mock_file)
                mock_img.save.assert_called_once_with(mock_buffer, format='JPEG')
                assert result == b'converted_image_data'

    def test_convert_image_rgba_to_rgb_conversion(self):
        """Тест конвертации RGBA в RGB для JPEG"""
        mock_file = MagicMock()

        with patch('app.Image.open') as mock_image_open:
            with patch('app.BytesIO') as mock_bytes_io:
                mock_img = MagicMock()
                mock_img.format = 'PNG'
                mock_img.mode = 'RGBA'  # Альфа-канал
                mock_img.size = (100, 100)

                converted_img = MagicMock()
                mock_img.convert.return_value = converted_img
                mock_image_open.return_value = mock_img

                mock_buffer = MagicMock()
                mock_buffer.getvalue.return_value = b'image_data'
                mock_bytes_io.return_value = mock_buffer

                # Конвертируем в JPEG (не поддерживает RGBA)
                result = convert_image_format(mock_file, 'JPEG')

                # Проверяем что конвертация RGBA→RGB была выполнена
                mock_img.convert.assert_called_once_with('RGB')
                converted_img.save.assert_called_once()


class TestLogging:
    """Тесты системы логирования"""

    def test_setup_logging(self):
        """Тест настройки логирования"""
        logger = setup_logging()

        assert logger is not None
        assert logger.name == 'file_converter'
        assert logger.level == 10  # DEBUG