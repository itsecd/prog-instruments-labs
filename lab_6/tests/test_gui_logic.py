#!/usr/bin/env python3
"""
Unit tests for GUI logic and business rules
"""

import sys
import os
import pytest
import tkinter as tk
from unittest.mock import MagicMock, patch
import shlex

# Добавляем родительскую директорию в путь для импорта основного модуля
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from nmap_gui_scan import NmapGUI, build_nmap_command, sanitize_arg


class MockTkVariable:
    """Mock class for Tkinter variables"""

    def __init__(self, initial_value=None):
        self._value = initial_value

    def get(self):
        return self._value

    def set(self, value):
        self._value = value

# Добавляем родительскую директорию в путь для импорта основного модуля
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from nmap_gui_scan import NmapGUI, build_nmap_command, sanitize_arg


class TestGUILogic:
    """Тесты бизнес-логики GUI"""

    def test_gui_initialization(self, gui_instance):
        """Тест инициализации GUI компонентов"""
        gui = gui_instance

        # Проверяем что основные переменные инициализированы
        assert hasattr(gui, 'target_var')
        assert hasattr(gui, 'port_choice')
        assert hasattr(gui, 'syn_var')
        assert hasattr(gui, 'adv_var')
        assert hasattr(gui, 'timing_var')
        assert hasattr(gui, 'agree_var')
        assert hasattr(gui, 'basename_var')

        # Проверяем начальные значения
        assert gui.target_var.get() == "192.168.1.0/24"
        assert gui.port_choice.get() == "common"
        assert gui.syn_var.get() == True
        assert gui.adv_var.get() == False
        assert gui.timing_var.get() == "T3"
        assert gui.agree_var.get() == False

    def test_start_scan_permission_check(self, gui_instance, monkeypatch):
        """Тест проверки разрешения перед запуском сканирования"""
        # Мокаем messagebox
        mock_messagebox = MagicMock()
        monkeypatch.setattr('nmap_gui_scan.messagebox.showwarning', mock_messagebox)

        # Мокаем shlex.split чтобы избежать проблем
        monkeypatch.setattr('nmap_gui_scan.shlex.split', lambda x: [])

        gui = gui_instance
        gui.agree_var.set(False)

        # Вызываем start_scan
        gui.start_scan()

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
    def test_target_sanitization_in_start_scan(self, gui_instance, monkeypatch, target_input, expected_output):
        """Тест очистки ввода цели в start_scan"""
        mock_messagebox = MagicMock()
        monkeypatch.setattr('nmap_gui_scan.messagebox.showwarning', mock_messagebox)

        # Мокаем shlex.split чтобы избежать проблем
        monkeypatch.setattr('nmap_gui_scan.shlex.split', lambda x: [])

        gui = gui_instance
        gui.agree_var.set(True)
        gui.target_var.set(target_input)

        # Если цель пустая, тест должен завершиться на проверке missing target
        if not expected_output:
            gui.start_scan()
            # Проверяем что было показано предупреждение о пустой цели
            mock_messagebox.assert_called_with("Missing target", "Enter a target or network to scan.")
            return

        # Для непустых целей мокаем дальнейшие вызовы
        mock_build_command = MagicMock(return_value=["nmap", "-sS", expected_output])
        monkeypatch.setattr('nmap_gui_scan.build_nmap_command', mock_build_command)

        mock_askokcancel = MagicMock(return_value=False)  # User cancels to stop early
        monkeypatch.setattr('nmap_gui_scan.messagebox.askokcancel', mock_askokcancel)

        gui.start_scan()

        # Проверяем что build_nmap_command был вызван с правильным значением
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

    def test_output_path_construction(self, gui_instance, monkeypatch):
        """Тест построения пути для сохранения результатов"""
        mock_messagebox = MagicMock()
        mock_messagebox.askokcancel.return_value = True
        monkeypatch.setattr('nmap_gui_scan.messagebox', mock_messagebox)

        mock_build_command = MagicMock()
        monkeypatch.setattr('nmap_gui_scan.build_nmap_command', mock_build_command)

        monkeypatch.setattr('nmap_gui_scan.threading.Thread', MagicMock())

        gui = gui_instance
        gui.agree_var.set(True)
        gui.target_var.set("192.168.1.1")
        gui.basename_var.set("my_custom_scan")
        gui.output_folder = "/custom/output"

        # Используем реальные MockTkVariable объекты
        gui.basename_var = MockTkVariable("my_custom_scan")

        gui.append_log = MagicMock()
        gui.status_var = MagicMock()
        gui.log = MagicMock()

        gui.start_scan()

        # Проверяем что путь был построен правильно
        mock_build_command.assert_called_once()
        call_args = mock_build_command.call_args[1]

        # Ожидаемый путь должен быть объединением output_folder и basename
        expected_path = os.path.join("/custom/output", "my_custom_scan")
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

    def test_start_scan_permission_check(self, gui_instance, mocker):
        """Тест проверки разрешения перед запуском сканирования"""
        # Мокаем messagebox
        mock_messagebox = mocker.patch('nmap_gui_scan.messagebox.showwarning')

        # Мокаем shlex.split чтобы избежать проблем с MagicMock
        mocker.patch('nmap_gui_scan.shlex.split', return_value=[])

        # Устанавливаем что согласие не дано
        gui_instance.agree_var.set(False)

        # Вызываем start_scan
        gui_instance.start_scan()

        # Проверяем что было показано предупреждение
        mock_messagebox.assert_called_once_with("Permission required",
                                                "You must confirm you have permission to scan the targets.")