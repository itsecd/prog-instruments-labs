#!/usr/bin/env python3
"""
Pytest configuration and fixtures for nmap_gui_scan tests
"""

import pytest
import sys
import os
from unittest.mock import MagicMock, patch

# Добавляем родительскую директорию в путь для импорта
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class MockTkVariable:
    """Mock class for Tkinter variables"""

    def __init__(self, initial_value=None):
        self._value = initial_value

    def get(self):
        return self._value

    def set(self, value):
        self._value = value


@pytest.fixture(autouse=True)
def mock_tkinter():
    """Automatically mock all Tkinter components for every test"""
    with patch('nmap_gui_scan.tk.Tk') as mock_tk, \
            patch('nmap_gui_scan.tk.BooleanVar') as mock_bool_var, \
            patch('nmap_gui_scan.tk.StringVar') as mock_str_var, \
            patch('nmap_gui_scan.ttk.Frame') as mock_frame, \
            patch('nmap_gui_scan.ttk.Label') as mock_label, \
            patch('nmap_gui_scan.ttk.Entry') as mock_entry, \
            patch('nmap_gui_scan.ttk.Button') as mock_button, \
            patch('nmap_gui_scan.ttk.Checkbutton') as mock_check, \
            patch('nmap_gui_scan.ttk.Radiobutton') as mock_radio, \
            patch('nmap_gui_scan.ttk.Combobox') as mock_combo, \
            patch('nmap_gui_scan.ttk.Labelframe') as mock_labelframe, \
            patch('nmap_gui_scan.scrolledtext.ScrolledText') as mock_text, \
            patch('nmap_gui_scan.filedialog.askdirectory') as mock_filedialog:
        # Configure mocks to return our custom variable classes
        def create_bool_var(value=False):
            return MockTkVariable(value)

        def create_str_var(value=""):
            return MockTkVariable(value)

        mock_bool_var.side_effect = create_bool_var
        mock_str_var.side_effect = create_str_var

        # Configure other mocks
        mock_filedialog.return_value = ""

        yield


@pytest.fixture
def gui_instance():
    """Фикстура для создания экземпляра NmapGUI с моками"""
    from nmap_gui_scan import NmapGUI

    # Mock the root window
    root_mock = MagicMock()

    # Create GUI instance
    gui = NmapGUI(root_mock)

    # Ensure all Tkinter variables are properly mocked
    gui.target_var = MockTkVariable("192.168.1.0/24")
    gui.port_choice = MockTkVariable("common")
    gui.syn_var = MockTkVariable(True)
    gui.adv_var = MockTkVariable(False)
    gui.timing_var = MockTkVariable("T3")
    gui.agree_var = MockTkVariable(False)
    gui.custom_ports_var = MockTkVariable("")
    gui.extra_args_var = MockTkVariable("")
    gui.basename_var = MockTkVariable("")

    # Mock other required attributes
    gui.append_log = MagicMock()
    gui.start_btn = MagicMock()
    gui.stop_btn = MagicMock()
    gui.status_var = MagicMock()
    gui.stop_requested = MagicMock()
    gui.log = MagicMock()
    gui.output_folder = os.getcwd()
    gui.output_folder_label = MagicMock()

    return gui


@pytest.fixture
def sample_nmap_command():
    """Фикстура с примером команды nmap для тестирования"""
    return ["nmap", "-sS", "-p", "22,80,443", "192.168.1.1"]


@pytest.fixture
def mock_subprocess():
    """Фикстура для мока subprocess"""
    with patch('nmap_gui_scan.subprocess.Popen') as mock_popen:
        mock_process = MagicMock()
        mock_process.stdout = iter(["Output line 1\n", "Output line 2\n"])
        mock_process.returncode = 0
        mock_popen.return_value = mock_process
        yield mock_popen, mock_process