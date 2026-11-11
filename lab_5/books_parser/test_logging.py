"""Проверки формата журнала и логирования работы приложения."""

import io
import json
import logging
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import main as app
from logging_config import APP_NAME, HOSTNAME, configure_logging

DATA_DIR = Path(__file__).parent


class LoggingTests(unittest.TestCase):
    """Проверяет журнал во временной папке."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

        self.root = Path(self.directory.name)
        self.log_path = self.root / "logs" / "app.log"

        self.logger = logging.getLogger(APP_NAME)
        self.old_level = self.logger.level
        self.old_propagate = self.logger.propagate
        self.addCleanup(self.close_logger)

        configure_logging(self.log_path)

    def close_logger(self):
        """Закрывает файл до удаления временной папки."""
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
            handler.close()

        self.logger.setLevel(self.old_level)
        self.logger.propagate = self.old_propagate

    def read_records(self):
        """Проверяет шесть полей и разбирает JSON каждой записи."""
        columns = [
            "datetime", "level", "event", "service", "hostname", "data"
        ]
        records = []

        for line in self.log_path.read_text(
            encoding="utf-8"
        ).splitlines():
            values = line.split("\t")
            self.assertEqual(len(values), len(columns))

            record = dict(zip(columns, values))
            record["data"] = json.loads(record["data"])
            records.append(record)

        return records

    def test_format_preserves_special_characters(self):
        data = {"text": 'Книга\t"Название"\nВторая строка'}
        self.logger.debug("FORMAT_CHECK", extra={"data": data})

        records = self.read_records()
        self.assertEqual(len(records), 1)

        record = records[0]
        self.assertEqual(record["level"], "DEBUG")
        self.assertEqual(record["event"], "FORMAT_CHECK")
        self.assertEqual(record["service"], APP_NAME)
        self.assertEqual(record["hostname"], HOSTNAME)
        self.assertEqual(record["data"], data)

        timestamp = datetime.fromisoformat(record["datetime"])
        self.assertEqual(timestamp.utcoffset(), timedelta(0))

    def test_reconfiguration_keeps_history_without_duplicates(self):
        self.logger.info("FIRST_EVENT")

        configure_logging(self.log_path)
        self.logger.info("SECOND_EVENT")

        events = [record["event"] for record in self.read_records()]
        self.assertEqual(events, ["FIRST_EVENT", "SECOND_EVENT"])

    def test_success_logs_events_and_preserves_csv(self):
        (self.root / "catalogue.html").write_bytes(
            (DATA_DIR / "catalogue.html").read_bytes()
        )

        with patch.object(app, "__file__", str(self.root / "main.py")):
            with redirect_stdout(io.StringIO()):
                exit_code = app.main()

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            (self.root / "books.csv").read_bytes(),
            (DATA_DIR / "books.csv").read_bytes(),
        )

        records = self.read_records()
        events = [record["event"] for record in records]

        self.assertEqual(events[0], "APP_STARTED")
        self.assertEqual(events[-1], "APP_FINISHED")
        self.assertEqual(events.count("BOOK_PARSED"), 20)
        self.assertNotIn("APP_FAILED", events)

        for event in (
            "INPUT_READ", "CARDS_FOUND", "CSV_WRITE_STARTED", "CSV_SAVED"
        ):
            self.assertIn(event, events)

        self.assertEqual(records[-1]["data"]["records"], 20)

    def test_failure_logs_traceback_and_preserves_previous_csv(self):
        (self.root / "catalogue.html").write_text(
            "<html>Нет карточек</html>", encoding="utf-8"
        )
        csv_path = self.root / "books.csv"
        csv_path.write_bytes(b"previous result")
        stderr = io.StringIO()

        with patch.object(app, "__file__", str(self.root / "main.py")):
            with redirect_stderr(stderr):
                exit_code = app.main()

        self.assertEqual(exit_code, 1)
        self.assertIn("Ошибка:", stderr.getvalue())
        self.assertEqual(csv_path.read_bytes(), b"previous result")

        records = self.read_records()
        failures = [
            record for record in records
            if record["event"] == "APP_FAILED"
        ]
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0]["level"], "ERROR")
        self.assertEqual(failures[0]["data"]["error_type"], "ValueError")
        self.assertIn("ValueError", failures[0]["data"]["traceback"])

        events = [record["event"] for record in records]
        self.assertNotIn("CSV_SAVED", events)
        self.assertNotIn("APP_FINISHED", events)


if __name__ == "__main__":
    unittest.main()