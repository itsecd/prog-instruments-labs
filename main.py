"""Модуль для учета успеваемости студентов учебной группы."""

from dataclasses import dataclass


@dataclass
class Student:
    """Модель данных студента.

    Attributes:
        name: Полное имя студента.
        grades: Список полученных оценок.
        is_active: Статус активности (обучается ли студент в данный момент).

    """

    name: str
    grades: list
    is_active: bool = True


class GradeBook:
    """Журнал успеваемости для управления списком студентов и их оценками."""

    def __init__(self, title: str) -> None:
        """Инициализирует журнал успеваемости.

        Args:
            title: Название группы или курса.

        """
        self.title = title
        self.students = {}

    def add_student(self, name: str, grades: list[int] | None = None) -> Student:
        """Добавляет нового студента в журнал.

        Args:
            name: Имя студента.
            grades: Начальный список оценок (по умолчанию пустой список).

        Returns:
            Созданный экземпляр студента.

        Raises:
            ValueError: Если передано пустое имя.

        """
        if not name:
            msg = "Имя студента не может быть пустым"
            raise ValueError(msg)

        student = Student(name=name, grades=grades)
        self.students[name] = student
        return student

    def get_student(self, name: str) -> Student | None:
        """Возвращает объект студента по имени.

        Args:
            name: Имя студента.

        Returns:
            Экземпляр Student или None, если студент не найден.

        """
        return self.students.get(name)

    def calculate_average(self, name: str) -> float:
        """Вычисляет средний балл оценок конкретного студента.

        Args:
            name: Имя студента.

        Returns:
            Средний балл, округленный до сотых, либо 0.0 при отсутствии оценок.

        """
        student = self.get_student(name)
        if student is None:
            return 0.0

        if len(student.grades) == 0:
            return 0.0

        total = sum(student.grades)
        avg = total / len(student.grades)
        return round(avg, 2)

    def get_top_students(self, threshold: float  = 4.5) -> list[tuple[str, float]]:
        """Возвращает список студентов, средний балл которых выше или равен порогу.

        Args:
            threshold: Минимальный средний балл для включения в список отличников.

        Returns:
            Список кортежей вида (имя_студента, средний_балл).

        """
        top_list = []
        for name in self.students:
            avg = self.calculate_average(name)
            if avg >= threshold:
                top_list.append((name, avg))
        return top_list

    def export_summary(self) -> dict[str, dict[str]]:
        """Формирует словарь со сводной информацией по всем студентам.

        Returns:
            Словарь с оценками, средним баллом и статусом каждого студента.

        """
        summary = {}
        for name, student in self.students.items():
            avg = self.calculate_average(name)
            summary[name] = {
                "grades": student.grades,
                "average": avg,
                "active": student.is_active,
            }
        return summary


def parse_raw_grades(raw_text: str) -> list[int]:
    """Парсит строку с оценками, разделенными запятыми, игнорируя некорректные записи.

    Args:
        raw_text: Сырая строка с оценками (например, "4, 5, test, 3").

    Returns:
        Список успешно распарсенных целочисленных оценок.

    """
    raw_items = raw_text.split(",")
    parsed = []
    for item in raw_items:
        clean_item = item.strip()
        try:
            val = int(clean_item)
            parsed.append(val)
        except ValueError:
            continue
    return parsed


def generate_report(gradebook: GradeBook) -> str:
    """Генерирует текстовый отчет по успеваемости всех студентов группы.

    Args:
        gradebook: Экземпляр журнала успеваемости.

    Returns:
        Многострочная строка с форматированным отчетом.

    """
    report_lines = []
    report_lines.append("=== Отчет по успеваемости ===")

    if len(gradebook.students) == 0:
        return "Нет данных для отчета."

    for name in gradebook.students:
        avg = gradebook.calculate_average(name)
        report_lines.append(f"Студент: {name}, Средний балл: {avg}")

    return "\n".join(report_lines)


def main() -> None:
    """Основная функция для демонстрации работы журнала успеваемости."""
    book = GradeBook("Группа 101")

    book.add_student("Иван Иванов", [5, 4, 5, 5])
    book.add_student("Петр Петров", [3, 4, 3, 4])
    book.add_student("Анна Сидорова", [5, 5, 5, 5])

    raw_data = "4, 5, ошибка, 4, 5"
    parsed_grades = parse_raw_grades(raw_data)
    book.add_student("Мария Смирнова", parsed_grades)

    print(generate_report(book))

    top = book.get_top_students()
    print("\nОтличники:")
    for name, avg in top:
        print(f"- {name}: {avg}")


if __name__ == "__main__":
    main()
