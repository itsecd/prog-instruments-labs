import datetime

class Employee:
    def __init__(self, name, hourly_rate, start_date):
        self.name = name
        self.hourly_rate = hourly_rate
        self.start_date = start_date
        self.hours_worked = 0
        self.completed_tasks = []

    def work(self, hours):
        if hours < 0:
            raise ValueError("Hours cannot be negative.")
        self.hours_worked += hours

    def calculate_pay(self):
        return self.hours_worked * self.hourly_rate

    def assign_task(self, task):
        if not task:
            raise ValueError("Task cannot be empty.")
        self.completed_tasks.append(task)

    def performance_review(self):
        if len(self.completed_tasks) > 10:
            return "Outstanding"
        elif len(self.completed_tasks) > 5:
            return "Good"
        else:
            return "Needs Improvement"

    def tenure(self):
        today = datetime.date.today()
        return (today - self.start_date).days // 365


class Timesheet:
    def __init__(self):
        self.entries = {}

    def log_time(self, employee_name, date, hours):
        if hours < 0:
            raise ValueError("Hours cannot be negative.")
        if not isinstance(date, datetime.date):
            raise TypeError("Date must be a datetime.date object.")
        if employee_name not in self.entries:
            self.entries[employee_name] = {}
        self.entries[employee_name][date] = self.entries[employee_name].get(date, 0) + hours

    def get_logged_hours(self, employee_name, date):
        if employee_name not in self.entries or date not in self.entries[employee_name]:
            return 0
        return self.entries[employee_name][date]

    def total_hours_for_employee(self, employee_name):
        if employee_name not in self.entries:
            return 0
        return sum(self.entries[employee_name].values())

    def employees_with_hours_on_date(self, date):
        if not isinstance(date, datetime.date):
            raise TypeError("Date must be a datetime.date object.")
        return [emp for emp, logs in self.entries.items() if date in logs]

    def most_active_employee(self):
        if not self.entries:
            return None
        return max(self.entries, key=lambda emp: sum(self.entries[emp].values()))
