#!/usr/bin/env python3
"""
Unit tests for nmap command building functionality
"""

import sys
import os
import pytest

# Добавляем родительскую директорию в путь для импорта основного модуля
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from nmap_gui_scan import build_nmap_command


class TestCommandBuilding:
    """Тесты построения команд nmap"""

    def test_basic_command_defaults(self):
        """Тест базовой команды с параметрами по умолчанию"""
        cmd = build_nmap_command(
            target="192.168.1.1",
            port_option="common",
            custom_ports="",
            syn_scan=True,
            enable_A=False,
            timing="T3",
            extra_args="",
            output_basename=""
        )

        expected = ["nmap", "-sS", "192.168.1.1"]
        assert cmd == expected

    def test_command_with_all_ports(self):
        """Тест команды со сканированием всех портов"""
        cmd = build_nmap_command(
            target="192.168.1.0/24",
            port_option="all",
            custom_ports="",
            syn_scan=True,
            enable_A=False,
            timing="T3",
            extra_args="",
            output_basename=""
        )

        expected = ["nmap", "-sS", "-p-", "192.168.1.0/24"]
        assert cmd == expected

    def test_command_with_custom_ports(self):
        """Тест команды с пользовательскими портами"""
        cmd = build_nmap_command(
            target="example.com",
            port_option="custom",
            custom_ports="22,80,443,8000-8100",
            syn_scan=False,  # TCP connect scan
            enable_A=False,
            timing="T3",
            extra_args="",
            output_basename=""
        )

        expected = ["nmap", "-sT", "-p", "22,80,443,8000-8100", "example.com"]
        assert cmd == expected

    def test_command_with_advanced_options(self):
        """Тест команды с расширенными опциями"""
        cmd = build_nmap_command(
            target="10.0.0.1",
            port_option="common",
            custom_ports="",
            syn_scan=True,
            enable_A=True,  # Enable OS detection, version detection, etc.
            timing="T4",  # Aggressive timing
            extra_args="",
            output_basename="scan_results"
        )

        expected = ["nmap", "-T4", "-sS", "-A", "-oA", "scan_results", "10.0.0.1"]
        assert cmd == expected

    def test_command_with_extra_args(self):
        """Тест команды с дополнительными аргументами"""
        cmd = build_nmap_command(
            target="192.168.1.1-100",
            port_option="common",
            custom_ports="",
            syn_scan=True,
            enable_A=False,
            timing="T2",  # Polite timing
            extra_args="--max-retries 3 --host-timeout 30m",
            output_basename=""
        )

        expected = ["nmap", "-T2", "-sS", "--max-retries", "3", "--host-timeout", "30m", "192.168.1.1-100"]
        assert cmd == expected

    def test_command_tcp_connect_scan(self):
        """Тест команды с TCP connect scan (без привилегий)"""
        cmd = build_nmap_command(
            target="localhost",
            port_option="common",
            custom_ports="",
            syn_scan=False,  # TCP connect
            enable_A=False,
            timing="T3",
            extra_args="",
            output_basename=""
        )

        expected = ["nmap", "-sT", "localhost"]
        assert cmd == expected

    def test_command_empty_custom_ports(self):
        """Тест команды с пустыми пользовательскими портами"""
        cmd = build_nmap_command(
            target="192.168.1.1",
            port_option="custom",
            custom_ports="",  # Empty custom ports
            syn_scan=True,
            enable_A=False,
            timing="T3",
            extra_args="",
            output_basename=""
        )

        # Should not include -p when custom ports are empty
        expected = ["nmap", "-sS", "192.168.1.1"]
        assert cmd == expected

    def test_command_with_output_basename_only(self):
        """Тест команды только с именем выходного файла"""
        cmd = build_nmap_command(
            target="192.168.1.1",
            port_option="common",
            custom_ports="",
            syn_scan=True,
            enable_A=False,
            timing="T3",
            extra_args="",
            output_basename="my_scan"
        )

        expected = ["nmap", "-sS", "-oA", "my_scan", "192.168.1.1"]
        assert cmd == expected

    def test_command_complex_extra_args(self):
        """Тест команды со сложными дополнительными аргументами"""
        cmd = build_nmap_command(
            target="192.168.1.0/24",
            port_option="common",
            custom_ports="",
            syn_scan=True,
            enable_A=True,
            timing="T5",  # Insane timing
            extra_args='-sV --script "http-* and safe"',
            output_basename="full_scan"
        )

        expected = ["nmap", "-T5", "-sS", "-A", "-sV", "--script", "http-* and safe", "-oA", "full_scan",
                    "192.168.1.0/24"]
        assert cmd == expected