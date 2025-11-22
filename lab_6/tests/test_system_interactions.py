#!/usr/bin/env python3
"""
Unit tests for system interactions and subprocess calls
"""

import sys
import os
import pytest
import subprocess
from unittest.mock import MagicMock, call

# Добавляем родительскую директорию в путь для импорта основного модуля
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from nmap_gui_scan import build_nmap_command, NmapGUI


class TestSystemInteractions:
    """Тесты системных взаимодействий и subprocess вызовов"""

    def test_nmap_subprocess_execution(self, gui_instance, mock_subprocess):
        """Тест выполнения nmap через subprocess с моками"""
        mock_popen, mock_process = mock_subprocess

        gui = gui_instance
        gui.stop_requested = MagicMock()
        gui.stop_requested.is_set.return_value = False

        test_cmd = ["nmap", "-sS", "192.168.1.1"]
        test_output_path = "/tmp/test_scan"

        gui.run_nmap(test_cmd, test_output_path)

        # Проверяем вызовы
        mock_popen.assert_called_once()

    def test_nmap_subprocess_with_stop_requested(self, mocker):
        """Тест прерывания выполнения nmap по запросу остановки"""
        mock_popen = mocker.patch('nmap_gui_scan.subprocess.Popen')
        mock_event = mocker.patch('nmap_gui_scan.threading.Event')

        # Настраиваем моки для остановки
        mock_process = MagicMock()
        mock_process.stdout = ["Starting Nmap\n", "Scanning...\n"]
        mock_process.returncode = 1
        mock_popen.return_value = mock_process
        mock_event.return_value.is_set.return_value = True  # Stop requested

        root_mock = MagicMock()
        gui = NmapGUI(root_mock)
        gui.append_log = MagicMock()
        gui.start_btn = MagicMock()
        gui.stop_btn = MagicMock()
        gui.status_var = MagicMock()
        gui.stop_requested = mock_event.return_value

        test_cmd = ["nmap", "-sS", "192.168.1.1"]
        gui.run_nmap(test_cmd, "/tmp/test")

        # Проверяем что процесс был завершен
        mock_process.terminate.assert_called_once()
        gui.append_log.assert_any_call("\nScan stopped by user.\n")

    def test_nmap_not_found_error(self, mocker):
        """Тест обработки ошибки когда nmap не найден"""
        mock_popen = mocker.patch('nmap_gui_scan.subprocess.Popen')
        mock_popen.side_effect = FileNotFoundError("nmap not found")

        root_mock = MagicMock()
        gui = NmapGUI(root_mock)
        gui.append_log = MagicMock()
        gui.start_btn = MagicMock()
        gui.stop_btn = MagicMock()
        gui.status_var = MagicMock()
        gui.stop_requested = MagicMock()
        gui.stop_requested.is_set.return_value = False

        test_cmd = ["nmap", "-sS", "192.168.1.1"]
        gui.run_nmap(test_cmd, "/tmp/test")

        gui.append_log.assert_any_call("\nError: nmap not found. Ensure nmap is installed and in PATH.\n")
        gui.start_btn.configure.assert_called_with(state='normal')

    def test_nmap_general_exception_handling(self, mocker):
        """Тест обработки общих исключений при выполнении nmap"""
        mock_popen = mocker.patch('nmap_gui_scan.subprocess.Popen')
        mock_popen.side_effect = Exception("General error")

        root_mock = MagicMock()
        gui = NmapGUI(root_mock)
        gui.append_log = MagicMock()
        gui.start_btn = MagicMock()
        gui.stop_btn = MagicMock()
        gui.status_var = MagicMock()
        gui.stop_requested = MagicMock()
        gui.stop_requested.is_set.return_value = False

        test_cmd = ["nmap", "-sS", "192.168.1.1"]
        gui.run_nmap(test_cmd, "/tmp/test")

        gui.append_log.assert_any_call("\nError running nmap: General error\n")
        gui.start_btn.configure.assert_called_with(state='normal')

    def test_output_folder_selection(self, mocker):
        """Тест выбора папки для сохранения результатов"""
        mock_filedialog = mocker.patch('nmap_gui_scan.filedialog.askdirectory')
        mock_filedialog.return_value = "/custom/output/folder"

        root_mock = MagicMock()
        gui = NmapGUI(root_mock)
        gui.output_folder = "/original/folder"
        gui.output_folder_label = MagicMock()

        # Вызываем метод выбора папки
        gui.choose_output_folder()

        # Проверяем что папка была обновлена
        assert gui.output_folder == "/custom/output/folder"
        gui.output_folder_label.configure.assert_called_with(text="Output folder: /custom/output/folder")

    def test_output_folder_selection_cancelled(self, mocker):
        """Тест отмены выбора папки"""
        mock_filedialog = mocker.patch('nmap_gui_scan.filedialog.askdirectory')
        mock_filedialog.return_value = ""  # Пользователь отменил выбор

        root_mock = MagicMock()
        gui = NmapGUI(root_mock)
        original_folder = "/original/folder"
        gui.output_folder = original_folder
        gui.output_folder_label = MagicMock()

        gui.choose_output_folder()

        # Папка не должна измениться
        assert gui.output_folder == original_folder
        gui.output_folder_label.configure.assert_not_called()

    def test_append_log_functionality(self, gui_instance):
        """Тест функциональности добавления логов"""
        gui = gui_instance

        test_message = "Test log message"

        gui.append_log(test_message)

        # Проверяем что метод был вызван
        gui.append_log.assert_called_once_with(test_message)

    def test_build_command_with_mocked_datetime(self, mocker):
        """Тест построения команды с моком datetime для предсказуемого имени файла"""
        mock_datetime = mocker.patch('nmap_gui_scan.datetime.datetime')
        mock_datetime.now.return_value.strftime.return_value = "20231225_120000"

        cmd = build_nmap_command(
            target="192.168.1.1",
            port_option="common",
            custom_ports="",
            syn_scan=True,
            enable_A=False,
            timing="T3",
            extra_args="",
            output_basename="scan_20231225_120000"
        )

        expected = ["nmap", "-sS", "-oA", "scan_20231225_120000", "192.168.1.1"]
        assert cmd == expected

    def test_stop_request_handling(self, gui_instance):
        """Тест обработки запроса остановки"""
        gui = gui_instance

        gui.stop_requested = MagicMock()
        gui.append_log = MagicMock()
        gui.status_var = MagicMock()

        # Вызываем запрос остановки
        gui.request_stop()

        # Проверяем что флаг установлен и логи добавлены
        gui.stop_requested.set.assert_called_once()
        gui.append_log.assert_called_with("\nStop requested. Attempting to terminate nmap...\n")
        gui.status_var.set.assert_called_with("Stopping...")

    def test_nmap_process_cleanup_on_error(self, mocker):
        """Тест очистки процесса nmap при ошибках"""
        mock_popen = mocker.patch('nmap_gui_scan.subprocess.Popen')
        mock_process = MagicMock()
        mock_process.stdout = ["Starting Nmap\n"]
        mock_process.wait.side_effect = subprocess.TimeoutExpired("nmap", 10)
        mock_popen.return_value = mock_process

        root_mock = MagicMock()
        gui = NmapGUI(root_mock)
        gui.append_log = MagicMock()
        gui.start_btn = MagicMock()
        gui.stop_btn = MagicMock()
        gui.status_var = MagicMock()
        gui.stop_requested = MagicMock()
        gui.stop_requested.is_set.return_value = False

        test_cmd = ["nmap", "-sS", "192.168.1.1"]
        gui.run_nmap(test_cmd, "/tmp/test")

        # Проверяем что кнопки были сброшены даже при ошибке
        gui.start_btn.configure.assert_called_with(state='normal')
        gui.stop_btn.configure.assert_called_with(state='disabled')