#!/usr/bin/env python3
"""
Pytest configuration and fixtures for nmap_gui_scan tests
"""

import pytest
import sys
import os
from unittest.mock import MagicMock

# Добавляем родительскую директорию в путь для импорта
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


@pytest.fixture
def mock_tkinter(mocker):
    """Фикстура для мока Tkinter компонентов"""
    # Мокаем Tkinter чтобы тесты могли работать без GUI
    mocker.patch('nmap_gui_scan.tk.Tk', return_value=MagicMock())
    mocker.patch('nmap_gui_scan.ttk.Frame', return_value=MagicMock())
    mocker.patch('nmap_gui_scan.ttk.Label', return_value=MagicMock())
    mocker.patch('nmap_gui_scan.ttk.Entry', return_value=MagicMock())
    mocker.patch('nmap_gui_scan.ttk.Button', return_value=MagicMock())
    mocker.patch('nmap_gui_scan.ttk.Checkbutton', return_value=MagicMock())
    mocker.patch('nmap_gui_scan.ttk.Radiobutton', return_value=MagicMock())
    mocker.patch('nmap_gui_scan.ttk.Combobox', return_value=MagicMock())
    mocker.patch('nmap_gui_scan.ttk.Labelframe', return_value=MagicMock())
    mocker.patch('nmap_gui_scan.scrolledtext.ScrolledText', return_value=MagicMock())
    mocker.patch('nmap_gui_scan.tk.BooleanVar', return_value=MagicMock())
    mocker.patch('nmap_gui_scan.tk.StringVar', return_value=MagicMock())


@pytest.fixture
def gui_instance(mock_tkinter):
    """Фикстура для создания экземпляра NmapGUI с моками"""
    from nmap_gui_scan import NmapGUI

    root_mock = MagicMock()
    gui = NmapGUI(root_mock)

    # Дополнительные моки для часто используемых атрибутов
    gui.append_log = MagicMock()
    gui.start_btn = MagicMock()
    gui.stop_btn = MagicMock()
    gui.status_var = MagicMock()
    gui.stop_requested = MagicMock()
    gui.log = MagicMock()

    return gui


@pytest.fixture
def sample_nmap_command():
    """Фикстура с примером команды nmap для тестирования"""
    return ["nmap", "-sS", "-p", "22,80,443", "192.168.1.1"]


@pytest.fixture
def sample_scan_parameters():
    """Фикстура с примером параметров сканирования"""
    return {
        "target": "192.168.1.1",
        "port_option": "custom",
        "custom_ports": "22,80,443",
        "syn_scan": True,
        "enable_A": False,
        "timing": "T3",
        "extra_args": "",
        "output_basename": "test_scan"
    }


@pytest.fixture
def gui_with_real_vars(mock_tkinter):
    """Фикстура для GUI с реальными Tkinter переменными"""
    from nmap_gui_scan import NmapGUI
    import tkinter as tk

    root_mock = MagicMock()
    gui = NmapGUI(root_mock)

    # Заменяем моки на реальные Tkinter переменные для некоторых тестов
    gui.target_var = tk.StringVar(value="192.168.1.0/24")
    gui.port_choice = tk.StringVar(value="common")
    gui.syn_var = tk.BooleanVar(value=True)
    gui.adv_var = tk.BooleanVar(value=False)
    gui.timing_var = tk.StringVar(value="T3")
    gui.agree_var = tk.BooleanVar(value=False)
    gui.custom_ports_var = tk.StringVar(value="")
    gui.extra_args_var = tk.StringVar(value="")
    gui.basename_var = tk.StringVar(value="")

    # Моки для остальных компонентов
    gui.append_log = MagicMock()
    gui.start_btn = MagicMock()
    gui.stop_btn = MagicMock()
    gui.status_var = MagicMock()
    gui.stop_requested = MagicMock()
    gui.log = MagicMock()

    return gui