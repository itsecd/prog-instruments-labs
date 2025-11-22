"""
System health monitoring and resource management.
"""

import psutil
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from threading import Thread, Event
import warnings

# Suppress psutil deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


@dataclass
class SystemHealth:
    """System health metrics."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_usage_percent: float
    open_files: int
    threads_count: int


class HealthMonitor:
    """
    Monitors system health and resources during translation.
    """

    def __init__(self, check_interval: float = 5.0):
        """
        Initialize health monitor.

        Args:
            check_interval: Health check interval in seconds
        """
        self.check_interval = check_interval
        self.monitoring = False
        self.monitor_thread: Optional[Thread] = None
        self.stop_event = Event()
        self.health_history: List[SystemHealth] = []
        self.warning_thresholds = {
            'cpu_percent': 80.0,
            'memory_percent': 85.0,
            'disk_usage_percent': 90.0
        }

    def start_monitoring(self):
        """Start health monitoring in background thread."""
        if self.monitoring:
            return

        self.monitoring = True
        self.stop_event.clear()
        self.monitor_thread = Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

    def stop_monitoring(self):
        """Stop health monitoring."""
        self.monitoring = False
        self.stop_event.set()
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)

    def _monitor_loop(self):
        """Main monitoring loop."""
        while not self.stop_event.is_set():
            try:
                health_data = self._collect_health_data()
                self.health_history.append(health_data)
                self._check_thresholds(health_data)
            except Exception as e:
                print(f"[Health Monitor] Error collecting health data: {e}")

            self.stop_event.wait(self.check_interval)

    def _collect_health_data(self) -> SystemHealth:
        """
        Collect current system health data.

        Returns:
            SystemHealth: Collected health data
        """
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)

        # Memory usage
        memory = psutil.virtual_memory()

        # Disk usage (current directory)
        disk = psutil.disk_usage('.')

        # Process info
        process = psutil.Process()
        open_files = len(process.open_files())
        threads_count = process.num_threads()

        return SystemHealth(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_mb=memory.used / 1024 / 1024,
            memory_available_mb=memory.available / 1024 / 1024,
            disk_usage_percent=disk.percent,
            open_files=open_files,
            threads_count=threads_count
        )

    def _check_thresholds(self, health_data: SystemHealth):
        """
        Check health metrics against warning thresholds.

        Args:
            health_data: Current health data to check
        """
        warnings = []

        if health_data.cpu_percent > self.warning_thresholds['cpu_percent']:
            warnings.append(f"High CPU usage: {health_data.cpu_percent:.1f}%")

        if health_data.memory_percent > self.warning_thresholds['memory_percent']:
            warnings.append(f"High memory usage: {health_data.memory_percent:.1f}%")

        if health_data.disk_usage_percent > self.warning_thresholds['disk_usage_percent']:
            warnings.append(f"High disk usage: {health_data.disk_usage_percent:.1f}%")

        if warnings:
            warning_msg = " | ".join(warnings)
            print(f"[Health Warning] {warning_msg}")

    def get_current_health(self) -> Optional[SystemHealth]:
        """
        Get most recent health data.

        Returns:
            Optional[SystemHealth]: Most recent health data or None
        """
        if self.health_history:
            return self.health_history[-1]
        return None

    def get_health_summary(self) -> Dict[str, Any]:
        """
        Get health monitoring summary.

        Returns:
            Dict: Health summary
        """
        if not self.health_history:
            return {}

        recent_data = self.health_history[-10:]  # Last 10 samples

        cpu_values = [h.cpu_percent for h in recent_data]
        memory_values = [h.memory_percent for h in recent_data]

        return {
            'samples_count': len(self.health_history),
            'current_cpu_percent': cpu_values[-1] if cpu_values else 0,
            'average_cpu_percent': sum(cpu_values) / len(cpu_values) if cpu_values else 0,
            'max_cpu_percent': max(cpu_values) if cpu_values else 0,
            'current_memory_percent': memory_values[-1] if memory_values else 0,
            'average_memory_percent': sum(memory_values) / len(memory_values) if memory_values else 0,
            'max_memory_percent': max(memory_values) if memory_values else 0,
            'monitoring_duration_seconds': (self.health_history[-1].timestamp -
                                            self.health_history[0].timestamp) if len(self.health_history) > 1 else 0
        }

    def set_warning_thresholds(self, cpu: float = 80.0, memory: float = 85.0,
                               disk: float = 90.0):
        """
        Set warning thresholds for health metrics.

        Args:
            cpu: CPU usage threshold percentage
            memory: Memory usage threshold percentage
            disk: Disk usage threshold percentage
        """
        self.warning_thresholds = {
            'cpu_percent': cpu,
            'memory_percent': memory,
            'disk_usage_percent': disk
        }


# Global health monitor
_health_monitor = HealthMonitor()


def get_health_monitor() -> HealthMonitor:
    """
    Get global health monitor instance.

    Returns:
        HealthMonitor: Health monitor instance
    """
    return _health_monitor