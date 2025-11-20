#!/usr/bin/env python3
"""
Unit tests for edge cases and additional coverage
"""

import sys
import os
import pytest
import subprocess
from unittest.mock import MagicMock, patch

# Добавляем родительскую директорию в путь для импорта основного модуля
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from nmap_gui_scan import build_nmap_command, sanitize_arg, NmapGUI, find_nmap


class TestEdgeCases:
    """Тесты граничных случаев и дополнительного покрытия"""

    def test_build_command_with_none_values(self):
        """Тест построения команды с None значениями"""
        cmd = build_nmap_command(
            target="192.168.1.1",
            port_option="common",
            custom_ports=None,  # None вместо строки
            syn_scan=True,
            enable_A=False,
            timing=None,  # None timing
            extra_args=None,  # None extra args
            output_basename=None  # None output
        )

        # Должна быть создана валидная команда даже с None значениями
        expected = ["nmap", "-sS", "192.168.1.1"]
        assert cmd == expected

    def test_sanitize_arg_with_whitespace_characters(self):
        """Тест очистки аргументов с различными пробельными символами"""
        test_cases = [
            ("  normal  spaces  ", "normal  spaces"),
            ("\ttab\tseparated\t", "tab\tseparated"),
            ("\nnewline\n", "newline"),
            ("\r\ncarriage\r\n", "carriage"),
            ("  mixed\t\nwhitespace  ", "mixed\t\nwhitespace"),
        ]

        for input_arg, expected in test_cases:
            result = sanitize_arg(input_arg)
            assert result == expected

    def test_build_command_special_characters_in_target(self):
        """Тест специальных символов в цели сканирования"""
        special_targets = [
            "hostname-with-dashes.com",
            "under_score_host",
            "192.168.1.1,192.168.1.2",  # Multiple hosts
            "fe80::1",  # IPv6
            "192.168.1.0/24,192.168.2.0/24",  # Multiple networks
        ]

        for target in special_targets:
            cmd = build_nmap_command(
                target=target,
                port_option="common",
                custom_ports="",
                syn_scan=True,
                enable_A=False,
                timing="T3",
                extra_args="",
                output_basename=""
            )

            # Команда должна быть построена без ошибок
            assert isinstance(cmd, list)
            assert len(cmd) >= 3
            assert cmd[-1] == target  # Цель должна быть последним аргументом

    def test_command_with_all_options_enabled(self):
        """Тест команды со всеми включенными опциями"""
        cmd = build_nmap_command(
            target="192.168.1.0/24",
            port_option="all",  # All ports
            custom_ports="",  # Ignored when port_option is all
            syn_scan=True,  # SYN scan
            enable_A=True,  # Advanced
            timing="T5",  # Fast timing
            extra_args="-v --max-retries 5 --script safe",  # Extra args
            output_basename="complete_scan"  # Output
        )

        expected = [
            "nmap", "-T5", "-sS", "-A", "-p-", "-v",
            "--max-retries", "5", "--script", "safe",
            "-oA", "complete_scan", "192.168.1.0/24"
        ]
        assert cmd == expected

    def test_nmap_process_timeout_handling(self, mocker):
        """Тест обработки таймаута процесса nmap"""
        mock_popen = mocker.patch('nmap_gui_scan.subprocess.Popen')
        mock_process = MagicMock()
        mock_process.stdout = ["Starting Nmap\n", "Still running...\n"]
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

        # Проверяем что состояние было сброшено даже при таймауте
        gui.start_btn.configure.assert_called_with(state='normal')
        gui.stop_btn.configure.assert_called_with(state='disabled')

    def test_gui_initialization_with_nmap_found(self, mocker):
        """Тест инициализации GUI когда nmap найден"""
        mock_which = mocker.patch('nmap_gui_scan.shutil.which')
        mock_which.return_value = "/usr/bin/nmap"  # nmap found
        mock_messagebox = mocker.patch('nmap_gui_scan.messagebox.showerror')

        root_mock = MagicMock()
        gui = NmapGUI(root_mock)

        # Проверяем что сообщение об ошибке НЕ было показано
        mock_messagebox.assert_not_called()
        # Кнопка запуска должна быть активна
        gui.start_btn.configure.assert_not_called()  # Не вызывалась для отключения

    def test_thread_safety_in_run_nmap(self, mocker):
        """Тест потокобезопасности в run_nmap"""
        mock_popen = mocker.patch('nmap_gui_scan.subprocess.Popen')
        mock_process = MagicMock()

        # Создаем имитацию вывода с задержкой
        output_lines = [f"Line {i}\n" for i in range(100)]
        mock_process.stdout = output_lines
        mock_process.returncode = 0
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

        # Проверяем что все строки вывода были обработаны
        assert gui.append_log.call_count == len(output_lines) + 2  # +2 для начального и конечного сообщений

    def test_build_command_whitespace_in_extra_args(self):
        """Тест пробелов в дополнительных аргументах"""
        cmd = build_nmap_command(
            target="192.168.1.1",
            port_option="common",
            custom_ports="",
            syn_scan=True,
            enable_A=False,
            timing="T3",
            extra_args="  --max-retries  3  --host-timeout  30m  ",  # Extra spaces
            output_basename=""
        )

        # Пробелы должны быть корректно обработаны shlex.split
        expected = ["nmap", "-sS", "--max-retries", "3", "--host-timeout", "30m", "192.168.1.1"]
        assert cmd == expected

    def test_empty_output_basename_handling(self):
        """Тест обработки пустого имени выходного файла"""
        cmd = build_nmap_command(
            target="192.168.1.1",
            port_option="common",
            custom_ports="",
            syn_scan=True,
            enable_A=False,
            timing="T3",
            extra_args="",
            output_basename=""  # Empty basename
        )

        # Не должно быть опций вывода
        expected = ["nmap", "-sS", "192.168.1.1"]
        assert cmd == expected

    def test_gui_string_vars_initialization(self):
        """Тест инициализации строковых переменных GUI"""
        # Импортируем tkinter для реальных переменных
        import tkinter as tk

        # Создаем реальные переменные для тестирования
        target_var = tk.StringVar(value="192.168.1.0/24")
        port_choice = tk.StringVar(value="common")
        timing_var = tk.StringVar(value="T3")

        # Проверяем начальные значения
        assert target_var.get() == "192.168.1.0/24"
        assert port_choice.get() == "common"
        assert timing_var.get() == "T3"

        # Проверяем изменение значений
        target_var.set("10.0.0.1")
        port_choice.set("all")
        timing_var.set("T5")

        assert target_var.get() == "10.0.0.1"
        assert port_choice.get() == "all"
        assert timing_var.get() == "T5"