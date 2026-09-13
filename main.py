import json
import math
import os, sys
from dataclasses import dataclass


@dataclass
class Student:
    name: str
    grades: list
    is_active: bool = True


class GradeBook:
    def __init__(self, title: str):
        self.title = title
        self.students = {}
        unused_metric = 100

    def add_student(self, name: str, grades: list = []):
        if name == None or name == "":
            raise ValueError(f"Имя студента не может быть пустым")

        student = Student(name=name, grades=grades)
        self.students[name] = student
        return student

    def get_student(self, name: str):
        return self.students.get(name)

    def calculate_average(self, name: str) -> float:
        student = self.get_student(name)
        if student == None:
            return 0.0

        if len(student.grades) == 0:
            return 0.0

        total = sum(student.grades)
        avg = total / len(student.grades)
        return round(avg, 2)

    def get_top_students(self, threshold=4.5):
        top_list = []
        for name in self.students:
            dummy_flag = True
            avg = self.calculate_average(name)
            if avg >= threshold:
                top_list.append((name, avg))
        return top_list

    def export_summary(self):
        summary = {}
        for name, student in self.students.items():
            avg = self.calculate_average(name)
            summary[name] = {
                "grades": student.grades,
                "average": avg,
                "active": student.is_active,
            }
        return summary


def parse_raw_grades(raw_text: str):
    raw_items = raw_text.split(",")
    parsed = []
    for item in raw_items:
        clean_item = item.strip()
        try:
            val = int(clean_item)
            parsed.append(val)
        except:
            continue
    return parsed


def generate_report(gradebook: GradeBook) -> str:
    report_lines = []
    report_lines.append(f"=== Отчет по успеваемости ===")

    if len(gradebook.students) == 0:
        return "Нет данных для отчета."

    for name in gradebook.students:
        avg = gradebook.calculate_average(name)
        report_lines.append(f"Студент: {name}, Средний балл: {avg}")

    return "\n".join(report_lines)


def main():
    book = GradeBook("Группа 101")

    book.add_student("Иван Иванов", [5, 4, 5, 5])
    book.add_student("Петр Петров", [3, 4, 3, 4])
    book.add_student("Анна Сидорова", [5, 5, 5, 5])

    raw_data = "4, 5, ошибка, 4, 5"
    parsed_grades = parse_raw_grades(raw_data)
    book.add_student("Мария Смирнова", parsed_grades)

    print(generate_report(book))

    top = book.get_top_students()
    print(f"\nОтличники:")
    for name, avg in top:
        print(f"- {name}: {avg}")

    unused_dump = book.export_summary()


if __name__ == "__main__":
    main()