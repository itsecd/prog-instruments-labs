#!/usr/bin/env python3
"""
Unit tests for GUI logic and business rules
"""

import sys
import os
import pytest
import tkinter as tk
from unittest.mock import MagicMock, patch, call

# Добавляем родительскую директорию в путь для импорта основного модуля
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from nmap_gui_scan import NmapGUI, build_nmap_command, sanitize_arg


class TestGUILogic:
    """Тесты бизнес-логики GUI"""

    def test_gui_initialization(self, gui_instance):
        """Тест инициализации GUI компонентов"""
        # Проверяем что основные переменные инициализированы
        assert hasattr(gui_instance, 'target_var')
        assert hasattr(gui_instance, 'port_choice')
        assert hasattr(gui_instance, 'syn_var')
        assert hasattr(gui_instance, 'adv_var')
        assert hasattr(gui_instance, 'timing_var')
        assert hasattr(gui_instance, 'agree_var')
        assert hasattr(gui_instance, 'basename_var')

        # Проверяем начальные значения
        assert gui_instance.target_var.get() == "192.168.1.0/24"
        assert gui_instance.port_choice.get() == "common"
        assert gui_instance.syn_var.get() == True
        assert gui_instance.adv_var.get() == False
        assert gui_instance.timing_var.get() == "T3"
        assert gui_instance.agree_var.get() == False

    def test_start_scan_permission_check(self, gui_instance, mocker):
        """Тест проверки разрешения перед запуском сканирования"""
        # Мокаем messagebox
        mock_messagebox = mocker.patch('nmap_gui_scan.messagebox.showwarning')

        # Устанавливаем что согласие не дано
        gui_instance.agree_var.set(False)

        # Вызываем start_scan
        gui_instance.start_scan()

        # Проверяем что было показано предупреждение
        mock_messagebox.assert_called_once_with("Permission required",
                                                "You must confirm you have permission to scan the targets.")

    def test_start_scan_missing_target(self, gui_instance, mocker):
        """Тест проверки отсутствия цели сканирования"""
        mock_messagebox = mocker.patch('nmap_gui_scan.messagebox.showwarning')

        # Устанавливаем согласие но пустую цель
        gui_instance.agree_var.set(True)
        gui_instance.target_var.set("")

        gui_instance.start_scan()

        mock_messagebox.assert_called_once_with("Missing target", "Enter a target or network to scan.")

    @pytest.mark.parametrize("target_input,expected_output", [
        ("192.168.1.1", "192.168.1.1"),
        ("  example.com  ", "example.com"),
        ("", ""),
    ])
    def test_target_sanitization_in_start_scan(self, gui_instance, mocker, target_input, expected_output):
        """Тест очистки ввода цели в start_scan"""
        mock_messagebox = mocker.patch('nmap_gui_scan.messagebox')
        mock_build_command = mocker.patch('nmap_gui_scan.build_nmap_command')

        gui_instance.agree_var.set(True)
        gui_instance.target_var.set(target_input)

        # Если цель пустая, тест завершится на проверке missing target
        if not expected_output:
            mock_messagebox.showwarning.assert_called()
            return

        # Для непустых целей продолжаем тест
        gui_instance.start_scan()

        # Проверяем что sanitize_arg был вызван с правильным значением
        # (функция вызывается внутри start_scan)
        mock_build_command.assert_called_once()
        call_args = mock_build_command.call_args[1]
        assert call_args['target'] == expected_output

    def test_start_scan_successful_flow(self, gui_instance, mocker):
        """Тест успешного запуска сканирования"""
        # Мокаем все зависимости
        mock_messagebox = mocker.patch('nmap_gui_scan.messagebox')
        mock_build_command = mocker.patch('nmap_gui_scan.build_nmap_command')
        mock_threading = mocker.patch('nmap_gui_scan.threading.Thread')

        # Настраиваем моки
        mock_messagebox.askokcancel.return_value = True  # User confirms
        mock_build_command.return_value = ["nmap", "-sS", "192.168.1.1"]

        # Настраиваем GUI состояние
        gui_instance.agree_var.set(True)
        gui_instance.target_var.set("192.168.1.1")
        gui_instance.port_choice.set("common")
        gui_instance.syn_var.set(True)
        gui_instance.adv_var.set(False)
        gui_instance.timing_var.set("T3")
        gui_instance.extra_args_var.set("")
        gui_instance.basename_var.set("test_scan")
        gui_instance.output_folder = "/tmp"

        gui_instance.append_log = MagicMock()
        gui_instance.status_var = MagicMock()
        gui_instance.stop_requested = MagicMock()
        gui_instance.stop_requested.is_set.return_value = False
        gui_instance.log = MagicMock()

        # Вызываем start_scan
        gui_instance.start_scan()

        # Проверяем вызовы
        mock_build_command.assert_called_once()
        mock_messagebox.askokcancel.assert_called_once()

        # Проверяем что поток был запущен
        mock_threading.assert_called_once()
        mock_threading.return_value.start.assert_called_once()

        # Проверяем обновление состояния GUI
        gui_instance.start_btn.configure.assert_called_with(state='disabled')
        gui_instance.stop_btn.configure.assert_called_with(state='normal')
        gui_instance.status_var.set.assert_called_with("Running scan...")

    def test_start_scan_user_cancels_confirmation(self, gui_instance, mocker):
        """Тест отмены пользователем подтверждения сканирования"""
        mock_messagebox = mocker.patch('nmap_gui_scan.messagebox')
        mock_messagebox.askokcancel.return_value = False  # User cancels

        gui_instance.agree_var.set(True)
        gui_instance.target_var.set("192.168.1.1")

        gui_instance.start_scan()

        # Проверяем что сканирование не было запущено
        mock_messagebox.askokcancel.assert_called_once()
        gui_instance.start_btn.configure.assert_not_called()  # Кнопка не должна меняться

    def test_basename_generation_with_datetime(self, gui_instance, mocker):
        """Тест генерации имени файла с временной меткой"""
        mock_datetime = mocker.patch('nmap_gui_scan.datetime.datetime')
        mock_datetime.now.return_value.strftime.return_value = "20231225_120000"

        mock_messagebox = mocker.patch('nmap_gui_scan.messagebox')
        mock_messagebox.askokcancel.return_value = True
        mock_build_command = mocker.patch('nmap_gui_scan.build_nmap_command')
        mocker.patch('nmap_gui_scan.threading.Thread')

        gui_instance.agree_var.set(True)
        gui_instance.target_var.set("192.168.1.1")
        gui_instance.basename_var.set("")  # Пустое имя файла

        gui_instance.append_log = MagicMock()
        gui_instance.status_var = MagicMock()
        gui_instance.log = MagicMock()

        gui_instance.start_scan()

        # Проверяем что было сгенерировано имя с временной меткой
        call_args = mock_build_command.call_args[1]
        assert "nmap_scan_20231225_120000" in call_args['output_basename']

    def test_custom_ports_handling(self, gui_instance, mocker):
        """Тест обработки пользовательских портов"""
        mock_messagebox = mocker.patch('nmap_gui_scan.messagebox')
        mock_messagebox.askokcancel.return_value = True
        mock_build_command = mocker.patch('nmap_gui_scan.build_nmap_command')
        mocker.patch('nmap_gui_scan.threading.Thread')

        gui_instance.agree_var.set(True)
        gui_instance.target_var.set("192.168.1.1")
        gui_instance.port_choice.set("custom")
        gui_instance.custom_ports_var.set("22,80,443,8000-8100")

        gui_instance.append_log = MagicMock()
        gui_instance.status_var = MagicMock()
        gui_instance.log = MagicMock()

        gui_instance.start_scan()

        # Проверяем что custom ports были переданы правильно
        call_args = mock_build_command.call_args[1]
        assert call_args['port_option'] == "custom"
        assert call_args['custom_ports'] == "22,80,443,8000-8100"

    def test_advanced_scan_options(self, gui_instance, mocker):
        """Тест расширенных опций сканирования"""
        mock_messagebox = mocker.patch('nmap_gui_scan.messagebox')
        mock_messagebox.askokcancel.return_value = True
        mock_build_command = mocker.patch('nmap_gui_scan.build_nmap_command')
        mocker.patch('nmap_gui_scan.threading.Thread')

        gui_instance.agree_var.set(True)
        gui_instance.target_var.set("192.168.1.1")
        gui_instance.adv_var.set(True)  # Enable advanced options
        gui_instance.timing_var.set("T5")  # Insane timing
        gui_instance.extra_args_var.set("-v --reason")

        gui_instance.append_log = MagicMock()
        gui_instance.status_var = MagicMock()
        gui_instance.log = MagicMock()

        gui_instance.start_scan()

        # Проверяем что расширенные опции были переданы
        call_args = mock_build_command.call_args[1]
        assert call_args['enable_A'] == True
        assert call_args['timing'] == "T5"
        assert call_args['extra_args'] == "-v --reason"

    def test_tcp_connect_scan_option(self, gui_instance, mocker):
        """Тест опции TCP connect сканирования"""
        mock_messagebox = mocker.patch('nmap_gui_scan.messagebox')
        mock_messagebox.askokcancel.return_value = True
        mock_build_command = mocker.patch('nmap_gui_scan.build_nmap_command')
        mocker.patch('nmap_gui_scan.threading.Thread')

        gui_instance.agree_var.set(True)
        gui_instance.target_var.set("192.168.1.1")
        gui_instance.syn_var.set(False)  # Use TCP connect instead of SYN

        gui_instance.append_log = MagicMock()
        gui_instance.status_var = MagicMock()
        gui_instance.log = MagicMock()

        gui_instance.start_scan()

        # Проверяем что был выбран TCP connect scan
        call_args = mock_build_command.call_args[1]
        assert call_args['syn_scan'] == False

    def test_output_path_construction(self, gui_instance, mocker):
        """Тест построения пути для сохранения результатов"""
        mock_messagebox = mocker.patch('nmap_gui_scan.messagebox')
        mock_messagebox.askokcancel.return_value = True
        mock_build_command = mocker.patch('nmap_gui_scan.build_nmap_command')
        mocker.patch('nmap_gui_scan.threading.Thread')

        gui_instance.agree_var.set(True)
        gui_instance.target_var.set("192.168.1.1")
        gui_instance.basename_var.set("my_custom_scan")
        gui_instance.output_folder = "/custom/output"

        gui_instance.append_log = MagicMock()
        gui_instance.status_var = MagicMock()
        gui_instance.log = MagicMock()

        gui_instance.start_scan()

        # Проверяем что путь был построен правильно
        call_args = mock_build_command.call_args[1]
        expected_path = "/custom/output/my_custom_scan"
        assert call_args['output_basename'] == expected_path

    def test_nmap_not_found_during_gui_init(self, mocker):
        """Тест обработки отсутствия nmap при инициализации GUI"""
        mock_which = mocker.patch('nmap_gui_scan.shutil.which')
        mock_which.return_value = None  # nmap not found
        mock_messagebox = mocker.patch('nmap_gui_scan.messagebox.showerror')

        root_mock = MagicMock()
        gui = NmapGUI(root_mock)

        # Проверяем что было показано сообщение об ошибке
        mock_messagebox.assert_called_once_with("nmap not found",
                                                "nmap executable not found in PATH. Please install nmap and ensure it's in your PATH.")

        # Проверяем что кнопка запуска отключена
        gui.start_btn.configure.assert_called_with(state='disabled')