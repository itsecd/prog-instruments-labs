import pytest
from unittest.mock import Mock
import datetime
from main import Employee, Timesheet


# Тесты для Employee
def test_employee_work_and_calculate_pay():
    emp = Employee("John", 20, datetime.date(2020, 1, 1))
    emp.work(8)
    emp.work(4)
    assert emp.calculate_pay() == 240

def test_employee_assign_task():
    emp = Employee("John", 20, datetime.date(2020, 1, 1))
    emp.assign_task("Task 1")
    emp.assign_task("Task 2")
    assert emp.completed_tasks == ["Task 1", "Task 2"]

def test_employee_performance_review():
    emp = Employee("John", 20, datetime.date(2020, 1, 1))
    for i in range(12):
        emp.assign_task(f"Task {i}")
    assert emp.performance_review() == "Outstanding"

def test_employee_tenure():
    emp = Employee("John", 20, datetime.date(2020, 1, 1))
    assert emp.tenure() >= 4  # Assuming the current year is 2024


# Тесты для Timesheet
def test_timesheet_log_and_get_hours():
    ts = Timesheet()
    ts.log_time("John", datetime.date(2024, 12, 6), 8)
    ts.log_time("John", datetime.date(2024, 12, 6), 4)
    assert ts.get_logged_hours("John", datetime.date(2024, 12, 6)) == 12

def test_timesheet_total_hours_for_employee():
    ts = Timesheet()
    ts.log_time("John", datetime.date(2024, 12, 6), 8)
    ts.log_time("John", datetime.date(2024, 12, 7), 6)
    assert ts.total_hours_for_employee("John") == 14

@pytest.mark.parametrize("date,expected", [
    (datetime.date(2024, 12, 6), ["John"]),
    (datetime.date(2024, 12, 7), ["John", "Alice"]),
    (datetime.date(2024, 12, 8), []),
])
def test_timesheet_employees_with_hours_on_date(date, expected):
    ts = Timesheet()
    ts.log_time("John", datetime.date(2024, 12, 6), 8)
    ts.log_time("Alice", datetime.date(2024, 12, 7), 4)
    ts.log_time("John", datetime.date(2024, 12, 7), 2)
    assert ts.employees_with_hours_on_date(date) == expected

def test_timesheet_most_active_employee():
    ts = Timesheet()
    ts.log_time("John", datetime.date(2024, 12, 6), 8)
    ts.log_time("Alice", datetime.date(2024, 12, 6), 10)
    assert ts.most_active_employee() == "Alice"

def test_timesheet_invalid_date_logging():
    ts = Timesheet()
    with pytest.raises(TypeError):
        ts.log_time("John", "invalid-date", 8)

def test_timesheet_log_time_with_mock():
    ts = Timesheet()
    ts.log_time = Mock()
    ts.log_time("John", datetime.date(2024, 12, 6), 8)
    ts.log_time("Alice", datetime.date(2024, 12, 6), 10)
    ts.log_time.assert_any_call("John", datetime.date(2024, 12, 6), 8)
    ts.log_time.assert_any_call("Alice", datetime.date(2024, 12, 6), 10)
    assert ts.log_time.call_count == 2

