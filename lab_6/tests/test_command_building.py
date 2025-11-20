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

    # Параметризованные тесты для различных комбинаций портов
    @pytest.mark.parametrize("port_option,custom_ports,expected_args", [
        ("common", "", []),  # No port args for common
        ("all", "", ["-p-"]),  # All ports
        ("custom", "80,443", ["-p", "80,443"]),  # Custom ports
        ("custom", "22,80,443,8000-8100", ["-p", "22,80,443,8000-8100"]),  # Range
        ("custom", "", []),  # Empty custom ports
    ])
    def test_port_options_parametrized(self, port_option, custom_ports, expected_args):
        """Параметризованный тест различных опций портов"""
        cmd = build_nmap_command(
            target="192.168.1.1",
            port_option=port_option,
            custom_ports=custom_ports,
            syn_scan=True,
            enable_A=False,
            timing="T3",
            extra_args="",
            output_basename=""
        )

        expected = ["nmap", "-sS"] + expected_args + ["192.168.1.1"]
        assert cmd == expected

    # Параметризованные тесты для timing templates
    @pytest.mark.parametrize("timing,expected_args", [
        ("T0", ["-T0"]),  # Paranoid
        ("T1", ["-T1"]),  # Sneaky
        ("T2", ["-T2"]),  # Polite
        ("T3", []),  # Normal (default, not included)
        ("T4", ["-T4"]),  # Aggressive
        ("T5", ["-T5"]),  # Insane
        ("", []),  # Empty timing
        (None, []),  # None timing
    ])
    def test_timing_options_parametrized(self, timing, expected_args):
        """Параметризованный тест timing templates"""
        cmd = build_nmap_command(
            target="192.168.1.1",
            port_option="common",
            custom_ports="",
            syn_scan=True,
            enable_A=False,
            timing=timing,
            extra_args="",
            output_basename=""
        )

        expected = ["nmap"] + expected_args + ["-sS", "192.168.1.1"]
        assert cmd == expected

    # Параметризованные тесты для scan types и advanced options
    @pytest.mark.parametrize("syn_scan,enable_A,expected_scan_type,expected_advanced", [
        (True, False, "-sS", []),  # SYN scan only
        (False, False, "-sT", []),  # TCP connect only
        (True, True, "-sS", ["-A"]),  # SYN + Advanced
        (False, True, "-sT", ["-A"]),  # TCP + Advanced
    ])
    def test_scan_types_advanced_parametrized(self, syn_scan, enable_A, expected_scan_type, expected_advanced):
        """Параметризованный тест типов сканирования и расширенных опций"""
        cmd = build_nmap_command(
            target="192.168.1.1",
            port_option="common",
            custom_ports="",
            syn_scan=syn_scan,
            enable_A=enable_A,
            timing="T3",
            extra_args="",
            output_basename=""
        )

        expected = ["nmap"] + expected_advanced + [expected_scan_type, "192.168.1.1"]
        assert cmd == expected

    # Параметризованные тесты для различных целей сканирования
    @pytest.mark.parametrize("target,description", [
        ("192.168.1.1", "Single IP"),
        ("192.168.1.0/24", "Network CIDR"),
        ("192.168.1.1-100", "IP range"),
        ("example.com", "Hostname"),
        ("localhost", "Localhost"),
        ("8.8.8.8,8.8.4.4", "Multiple IPs"),
    ])
    def test_different_targets_parametrized(self, target, description):
        """Параметризованный тест различных типов целей"""
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

        expected = ["nmap", "-sS", target]
        assert cmd == expected, f"Failed for target: {description}"

    # Параметризованные тесты для extra args
    @pytest.mark.parametrize("extra_args,expected_extra", [
        ("", []),
        ("--max-retries 3", ["--max-retries", "3"]),
        ("-v --reason", ["-v", "--reason"]),
        ('--script "default and safe"', ["--script", "default and safe"]),
        ("-O --osscan-limit", ["-O", "--osscan-limit"]),
    ])
    def test_extra_args_parametrized(self, extra_args, expected_extra):
        """Параметризованный тест дополнительных аргументов"""
        cmd = build_nmap_command(
            target="192.168.1.1",
            port_option="common",
            custom_ports="",
            syn_scan=True,
            enable_A=False,
            timing="T3",
            extra_args=extra_args,
            output_basename=""
        )

        expected = ["nmap", "-sS"] + expected_extra + ["192.168.1.1"]
        assert cmd == expected

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
