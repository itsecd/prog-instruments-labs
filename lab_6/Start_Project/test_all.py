
import json
import logging
import sqlite3
import sys
import unittest.mock as mock

import pytest

from handler import SQLiteHandler
from log_parser import main
from logger_config import setup_logger
from parser import _parse_line, _save_stats, parse_logs
from patterns import LOG_PATTERN


VALID_LINES = [
    pytest.param(
        '192.168.1.10 - - [28/Aug/2025:14:32:10 +0800] "GET /index.html HTTP/1.1" 200 1024',
        {
            "ip": "192.168.1.10", "identd": "-", "userid": "-",
            "date": "28/Aug/2025:14:32:10 +0800", "method": "GET",
            "url": "/index.html", "status": "200", "size": "1024",
        },
        id="get-index",
    ),
    pytest.param(
        '1.1.1.1 testuser testid [01/Jan/2026:00:00:00 -0000] "PUT /x HTTP/2.0" 201 10',
        {
            "ip": "1.1.1.1", "identd": "testuser", "userid": "testid",
            "date": "01/Jan/2026:00:00:00 -0000", "method": "PUT",
            "url": "/x", "status": "201", "size": "10",
        },
        id="non-dash-identd",
    ),
]

BROKEN_LINES = [
    pytest.param("BROKEN_LINE_1", id="garbage"),
    pytest.param(
        '192.168.1.10 - - 28/Aug/2025:14:42:00 +0800 "GET /no-brackets HTTP/1.1" 200 100',
        id="no-date-brackets",
    ),
    pytest.param(
        '192.168.1.10 - - [28/Aug/2025:14:32:10 +0800] "123 /index.html HTTP/1.1" 200 1024',
        id="method-digits",
    ),
]

@pytest.mark.parametrize("line,expected_fields", VALID_LINES)
def test_valid_line_parses_all_fields(line, expected_fields):
    """Валидная строка матчится регуляркой, поля извлекаются верно."""
    match = LOG_PATTERN.match(line)
    assert match is not None
    assert match.groupdict() == expected_fields


@pytest.mark.parametrize("line", BROKEN_LINES)
def test_broken_line_does_not_match(line):
    """Битые строки не матчатся регуляркой."""
    assert LOG_PATTERN.match(line) is None


VALID_LINE = '192.168.1.10 - - [28/Aug/2025:14:32:10 +0800] "GET /index.html HTTP/1.1" 200 1024'

def test_parse_line_returns_dict_or_none():
    """_parse_line возвращает словарь для валидной строки и None для битой."""
    data = _parse_line(VALID_LINE)
    assert data is not None
    assert data["ip"] == "192.168.1.10"
    assert _parse_line("BROKEN_LINE") is None


def test_parse_logs_collects_correct_stats(tmp_path):
    """parse_logs считает валидные/битые строки и статус-коды."""
    input_file = tmp_path / "input.txt"
    input_file.write_text(
        VALID_LINE + "\nBROKEN_LINE_1\n"
        '8.8.8.8 - - [28/Aug/2025:14:33:01 +0800] "GET /contact HTTP/1.1" 404 150\n',
        encoding="utf-8"
    )

    stats = parse_logs(
        str(input_file), str(tmp_path / "valid.csv"),
        str(tmp_path / "broken.txt"), str(tmp_path / "stats.json"),
    )

    assert stats["valid_lines"] == 2
    assert stats["broken_lines"] == 1
    assert stats["status_codes"] == {"200": 1, "404": 1}


def test_parse_logs_raises_on_missing_file(tmp_path):
    """Отсутствие входного файла даёт FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        parse_logs(
            str(tmp_path / "nonexistent.txt"), str(tmp_path / "valid.csv"),
            str(tmp_path / "broken.txt"), str(tmp_path / "stats.json"),
        )


def test_parse_logs_propagates_unexpected_error(tmp_path):
    """Непредвиденное исключение пробрасывается."""
    input_file = tmp_path / "input.txt"
    input_file.write_text(VALID_LINE, encoding="utf-8")

    with mock.patch("parser._process_lines", side_effect=RuntimeError("boom")):
        with pytest.raises(RuntimeError, match="boom"):
            parse_logs(
                str(input_file), str(tmp_path / "valid.csv"),
                str(tmp_path / "broken.txt"), str(tmp_path / "stats.json"),
            )


def test_save_stats_writes_valid_json(tmp_path):
    """_save_stats сериализует данные в JSON."""
    stats_json = tmp_path / "stats.json"
    stats = {"valid_lines": 1, "broken_lines": 2}
    _save_stats(stats, str(stats_json))
    assert json.loads(stats_json.read_text(encoding="utf-8")) == stats


def _make_record(msg="hello", level=logging.INFO, name="test_logger"):
    return logging.LogRecord(
        name=name, level=level, pathname=__file__, lineno=1,
        msg=msg, args=(), exc_info=None,
    )


def test_init_db_creates_table_and_is_idempotent(tmp_path):
    """Создается таблица, повторный вызов не ломает данные."""
    db = tmp_path / "test.db"
    handler = SQLiteHandler(str(db))
    handler.emit(_make_record(msg="first"))
    
    SQLiteHandler(str(db))  

    with sqlite3.connect(str(db)) as conn:
        cols = [row[1] for row in conn.execute("PRAGMA table_info(logs)").fetchall()]
        count = conn.execute("SELECT COUNT(*) FROM logs").fetchone()[0]
        
    assert cols == ["id", "timestamp", "level", "logger_name", "message"]
    assert count == 1


def test_emit_writes_row_with_format_args(tmp_path):
    """emit() вставляет запись и подставляет аргументы через getMessage()."""
    db = tmp_path / "test.db"
    handler = SQLiteHandler(str(db))
    record = logging.LogRecord(
        name="t", level=logging.INFO, pathname=__file__, lineno=1,
        msg="user %s", args=("alice",), exc_info=None,
    )
    handler.emit(record)

    with sqlite3.connect(str(db)) as conn:
        (message,) = conn.execute("SELECT message FROM logs").fetchone()
    assert message == "user alice"


def test_emit_swallows_errors_and_calls_handle_error(tmp_path, monkeypatch):
    """Если sqlite3.connect падает — emit() вызывает handleError."""
    db = tmp_path / "test.db"
    handler = SQLiteHandler(str(db))
    record = _make_record()

    monkeypatch.setattr(
        "handler.sqlite3.connect",
        lambda *a, **kw: (_ for _ in ()).throw(sqlite3.OperationalError("boom")),
    )
    called = {}
    monkeypatch.setattr(handler, "handleError", lambda r: called.setdefault("ok", r))

    handler.emit(record)
    assert called["ok"] is record


def test_setup_logger_configures_handlers_correctly(tmp_path):
    """Логгер имеет верные уровни и ровно два хендлера."""
    logger = setup_logger(db_path=str(tmp_path / "logs.db"))

    assert logger.name == "log_parser"
    assert logger.level == logging.DEBUG
    assert len(logger.handlers) == 2
    
    console = next(h for h in logger.handlers if isinstance(h, logging.StreamHandler) and not isinstance(h, SQLiteHandler))
    db = next(h for h in logger.handlers if isinstance(h, SQLiteHandler))
    assert console.level == logging.INFO
    assert db.level == logging.DEBUG


def test_setup_logger_writes_debug_to_db_not_console(tmp_path, capsys):
    """DEBUG уходит в БД, но не в stdout."""
    logger = setup_logger(db_path=str(tmp_path / "logs.db"))
    logger.debug("debug-only")
    logger.info("info-visible")

    captured = capsys.readouterr()
    assert "debug-only" not in captured.out
    assert "info-visible" in captured.out


def test_main_without_args_exits():
    """Без аргументов argparse завершается с кодом 2."""
    with mock.patch.object(sys, "argv", ["log_parser"]):
        with pytest.raises(SystemExit) as exc:
            main()
    assert exc.value.code == 2


def test_main_missing_file_logs_error(tmp_path, caplog):
    """Отсутствие файла не роняет main(): ERROR + return."""
    argv = ["log_parser", str(tmp_path / "missing.txt")]
    with mock.patch.object(sys, "argv", argv):
        with caplog.at_level(logging.ERROR, logger="log_parser"):
            main()

    assert any(r.levelno == logging.ERROR and "Файл не найден" in r.getMessage() for r in caplog.records)


def test_main_calls_parse_logs_with_args(tmp_path):
    """main() пробрасывает CLI-аргументы в parse_logs."""
    input_file = tmp_path / "in.txt"
    input_file.write_text("", encoding="utf-8")

    argv = ["log_parser", str(input_file), "-o", str(tmp_path / "out.csv")]
    fake_stats = {"total_lines_processed": 0, "valid_lines": 0, "broken_lines": 0, "status_codes": {}}

    with mock.patch.object(sys, "argv", argv), \
         mock.patch("log_parser.parse_logs", return_value=fake_stats) as mocked:
        main()

    mocked.assert_called_once_with(str(input_file), str(tmp_path / "out.csv"), "broken_logs.txt", "stats.json")