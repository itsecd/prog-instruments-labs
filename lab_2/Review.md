# Code Review: `check_error_utils.py`

## Общая информация

**Файл:** `check_error_utils.py`
**Назначение:** Модуль для централизованной обработки ошибок в GUI-приложении на `tkinter`. Содержит обёртки над `messagebox.*`, декораторы для автоматической обработки исключений (`handle_errors`), подтверждения действий (`confirm_action`) и безопасных файловых/криптографических операций.
---

## Замечание 1: Неиспользуемый параметр `operation_name`

**Файл:** `check_error_utils.py`
**Строка:** 140
**Тип проблемы:** Мёртвый код (неиспользуемый параметр), нарушение YAGNI
**Важность:** Квадрант III (низкая важность, низкая конфликтность)

**Описание:**
В сигнатуре `safe_encryption_operation` объявлен параметр `operation_name: str`, но в теле функции он не используется. Это вводит в заблуждение вызывающий код: создаётся впечатление, что название операции как-то влияет на поведение, хотя это не так. В docstring прямо сказано: «Название операции (не используется)».

**Предложение:**
Удалить `operation_name` из сигнатуры.

**Пример кода:**
```python
def safe_encryption_operation(status_callback: Optional[Callable] = None) -> Callable:
    return handle_errors(
        error_title="Криптографическая ошибка",
        show_traceback=True,
        status_callback=status_callback
    )
```

---

## Замечание 2: Отсутствие модульных тестов

**Файл:** `check_error_utils.py`
**Вся структура**
**Тип проблемы:** Отсутствие модульных тестов
**Важность:** Квадрант I (высокая важность, низкая конфликтность)

**Описание:**
Модуль не покрыт тестами. Декораторы — идеальный кандидат для юнит-тестов, так как их поведение легко проверить изолированно: можно подменить `show_error` и `show_question` через `unittest.mock` и проверить, что декоратор возвращает `fallback_return`, вызывает `status_callback` и корректно обрабатывает отказ пользователя.

**Предложение:**
Добавить тесты с использованием `unittest.mock`.

**Пример кода:**
```python
from unittest.mock import MagicMock, patch
from check_error_utils import handle_errors, confirm_action


def test_handle_errors_returns_fallback():
    @handle_errors(fallback_return=42)
    def failing():
        raise ValueError("test")

    assert failing() == 42


def test_handle_errors_calls_status_callback():
    callback = MagicMock()

    @handle_errors(status_callback=callback)
    def failing():
        raise ValueError("test")

    failing()
    callback.assert_called_once()


def test_confirm_action_returns_none_on_reject():
    with patch('check_error_utils.show_question', return_value=False):
        @confirm_action("Test", "Are you sure?")
        def do_something():
            return 42

        assert do_something() is None


def test_confirm_action_calls_function_on_accept():
    with patch('check_error_utils.show_question', return_value=True):
        @confirm_action("Test", "Are you sure?")
        def do_something():
            return 42

        assert do_something() == 42
```

---

## Замечание 3 (nit): Дублирование в `safe_file_operation` и `safe_encryption_operation`

**Файл:** `check_error_utils.py`
**Строки:** 127–142
**Тип проблемы:** Дублирование кода (nit)
**Важность:** Квадрант II (низкая важность, высокая конфликтность)

**Описание:**
`safe_file_operation` и `safe_encryption_operation` — почти идентичные обёртки над `handle_errors`. Различаются только заголовком ошибки. Это не критично, но если обёрток станет больше, дублирование начнёт мешать.

**Предложение:**
Можно заменить обе функции одной универсальной обёрткой:

```python
def safe_operation(error_title: str, status_callback: Optional[Callable] = None) -> Callable:
    return handle_errors(
        error_title=error_title,
        show_traceback=True,
        status_callback=status_callback
    )
```

Тогда использование будет явным:

```python
@safe_operation("Ошибка при чтении файла")
def read_file(...):
    ...
```

**nit:** Это вопрос вкуса. Текущий вариант с двумя функциями тоже допустим — он даёт более говорящие имена.

---

## Замечание 4 (nit): `import traceback` вместо `from traceback import format_exc`

**Файл:** `check_error_utils.py`
**Строка:** 1
**Тип проблемы:** Стилистика (nit)
**Важность:** Квадрант II (низкая важность, высокая конфликтность)

**Описание:**
Используется `import traceback` и вызов `traceback.format_exc()`. Можно сократить до `from traceback import format_exc` и вызывать `format_exc()`. Однако это **не влияет** на производительность и не является ошибкой: `traceback` — часть стандартной библиотеки, и разница в потреблении памяти ничтожна. Более того, `import traceback` + `traceback.format_exc()` часто читается лучше, так как явно показывает источник функции.

**Предложение:**
Оставить как есть или заменить на `from traceback import format_exc`, если в проекте принят стиль «импортировать только нужные функции».

**nit:** Это не проблема, а вкусовщина. По лекции: «Избегаем вкусовщины!»

---

## Замечание 5: `handle_errors` глотает исключения без возможности повторного выброса

**Файл:** `check_error_utils.py`
**Строки:** 78–80
**Тип проблемы:** Сокрытие ошибок, сложность отладки
**Важность:** Квадрант I (высокая важность, низкая конфликтность)

**Описание:**
Декоратор перехватывает **все** исключения (`except Exception`) и возвращает `fallback_return`. Это удобно для GUI, но в процессе разработки и отладки такое поведение мешает: исключение не пробрасывается дальше, и разработчик не может использовать отладчик или `pytest.raises`.

**Предложение:**
Добавить флаг `reraise: bool = False`. При `True` исключение логируется, показывается пользователю и пробрасывается дальше.

**Пример кода:**
```python
def handle_errors(
    error_title: str = "Ошибка",
    show_traceback: bool = True,
    fallback_return: Any = None,
    status_callback: Optional[Callable] = None,
    reraise: bool = False
) -> Callable:
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_message = str(e) if str(e) else f"Ошибка в {func.__name__}"
                logger.exception("Ошибка в %s", func.__name__)
                details = traceback.format_exc() if show_traceback else None
                show_error(error_title, error_message, details)
                if status_callback:
                    try:
                        status_callback(f"Ошибка в {func.__name__}: {error_message}")
                    except Exception as callback_error:
                        logger.warning("Ошибка в status_callback: %s", callback_error)
                if reraise:
                    raise
                return fallback_return
        return wrapper
    return decorator
```

---

## Замечание 6: `confirm_action` возвращает `None` при отказе — неявный контракт

**Файл:** `check_error_utils.py`
**Строки:** 102–124
**Тип проблемы:** Неявное поведение, потенциальная ошибка логики
**Важность:** Квадрант IV (высокая важность, высокая конфликтность)

**Описание:**
Если пользователь нажал «Нет», декоратор `confirm_action` возвращает `None`. Это может быть проблемой, если декорируемая функция должна возвращать, например, `bool` или список. Вызывающий код может не ожидать `None` и упасть с `AttributeError` или `TypeError`. При этом в docstring это поведение не описано.

**Предложение:**
Добавить параметр `fallback_return` и задокументировать поведение при отказе.

**Пример кода:**
```python
def confirm_action(
    title: str,
    message: str,
    fallback_return: Any = None
) -> Callable:
    """
    Декоратор для подтверждения действия перед выполнением функции.

    Args:
        title: Заголовок окна подтверждения.
        message: Текст вопроса для пользователя.
        fallback_return: Значение, возвращаемое при отказе пользователя.

    Returns:
        Декорированная функция. Если пользователь отказался — возвращает fallback_return.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if show_question(title, message):
                return func(*args, **kwargs)
            return fallback_return
        return wrapper
    return decorator
```

---

## Замечание 7: Неинформативный вызов `status_callback`

**Файл:** `check_error_utils.py`
**Строки:** 88–92
**Тип проблемы:** Неинформативное сообщение об ошибке, недостаточная гибкость
**Важность:** Квадрант I (высокая важность, низкая конфликтность)

**Описание:**
В `handle_errors` при возникновении исключения всегда вызывается `status_callback("Ошибка")`. Это неинформативно: пользователь видит только слово «Ошибка» и не понимает, что именно пошло не так. При этом в блоке `except` уже есть переменная `error_message`, которую можно передать в колбэк.

**Предложение:**
Передавать в `status_callback` текст ошибки и имя функции, в которой она произошла.

**Пример кода:**
```python
if status_callback:
    try:
        status_callback(f"Ошибка в {func.__name__}: {error_message}")
    except Exception as callback_error:
        logger.warning("Ошибка в status_callback: %s", callback_error)
```

---

## Замечание 8: `handle_errors` не логирует ошибки

**Файл:** `check_error_utils.py`
**Строки:** 62–99
**Тип проблемы:** Недостаточность логирования
**Важность:** Квадрант I (высокая важность, низкая конфликтность)

**Описание:**
Модуль отвечает за обработку ошибок, но нигде не логирует их. Если пользователь закрыл диалог и не сообщил о проблеме, информация об ошибке теряется навсегда. В GUI-приложении это может быть приемлемо для мелких ошибок, но для критических — нет.

**Предложение:**
Добавить `logging` и логировать все ошибки с уровнем `ERROR`.

**Пример кода:**
```python
import logging

logger = logging.getLogger(__name__)


def handle_errors(...):
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_message = str(e) if str(e) else f"Ошибка в {func.__name__}"
                logger.exception("Ошибка в %s", func.__name__)
                details = traceback.format_exc() if show_traceback else None
                show_error(error_title, error_message, details)
                if status_callback:
                    try:
                        status_callback(f"Ошибка в {func.__name__}: {error_message}")
                    except Exception as callback_error:
                        logger.warning("Ошибка в status_callback: %s", callback_error)
                return fallback_return
        return wrapper
    return decorator
```

---




# Code Review: `file_utils.py`

## Общая информация

**Файл:** `file_utils.py`
**Назначение:** Утилиты для чтения/записи файлов (JSON, бинарные данные) с обёрткой исключений в кастомное `FileOperationError`.

---

## Замечание 1: `write_binary_file` падает, если `file_path` без директории

**Файл:** `file_utils.py`
**Строка:** 76
**Тип проблемы:** Ошибка логики
**Важность:** Квадрант IV (высокая важность, высокая конфликтность)

**Описание:**
`os.path.dirname(file_path)` возвращает пустую строку, если в `file_path` нет директории (например, `"data.bin"`). `os.makedirs("", exist_ok=True)` выбрасывает `FileNotFoundError`. То есть запись файла в текущую директорию с `create_dirs=True` (по умолчанию!) сломается.

**Предложение:** Проверять, что `dirname` не пустой, перед созданием директорий.

**Пример кода:**
```python
if create_dirs:
    dir_path = os.path.dirname(file_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
```

---

## Замечание 2: `file_exists` не различает файл и директорию

**Файл:** `file_utils.py`
**Строка:** 98
**Тип проблемы:** Вводящее в заблуждение поведение
**Важность:** Квадрант I (высокая важность, низкая конфликтность)

**Описание:**
`os.path.exists(file_path)` возвращает `True` и для директорий. Функция называется `file_exists`, но по факту проверяет «существует ли что-либо по этому пути». Это может привести к багам: вызывающий код думает, что это файл, и пытается его открыть, а это директория.

**Предложение:** Использовать `os.path.isfile`, если нужно проверить именно файл. Если нужна проверка «существует ли путь» — переименовать в `path_exists`.

**Пример кода:**
```python
def file_exists(file_path: str) -> bool:
    """
    Проверяет, существует ли файл (не директория).
    """
    return os.path.isfile(file_path)
```

---

## Замечание 3: `read_json_file` не проверяет, что результат — словарь

**Файл:** `file_utils.py`
**Строка:** 24
**Тип проблемы:** Несоответствие контракта и реальности
**Важность:** Квадрант I (высокая важность, низкая конфликтность)

**Описание:**
Аннотация возвращает `Dict[str, Any]`, но `json.load` может вернуть список, строку, число, `None`. Если в файле лежит `[1, 2, 3]` или `"hello"`, функция вернёт это без ошибок, а вызывающий код упадёт с `AttributeError` при попытке `result["key"]`.

**Предложение:** Проверять тип после парсинга и выбрасывать `FileOperationError`, если это не словарь. Либо изменить аннотацию на `Any`, если словарь не гарантируется.

**Пример кода:**
```python
data = json.load(f)
if not isinstance(data, dict):
    raise FileOperationError(
        f"Ожидался JSON-объект (dict), получен {type(data).__name__}: {file_path}"
    )
return data
```

---

## Замечание 4: Дублирование логики обработки исключений

**Файл:** `file_utils.py`
**Строки:** 24–40, 51–63, 76–90
**Тип проблемы:** Дублирование кода (DRY)
**Важность:** Квадрант III (низкая важность, низкая конфликтность)

**Описание:**
Каждая из трёх функций (`read_json_file`, `read_binary_file`, `write_binary_file`) содержит почти одинаковые блоки `except` с обёрткой в `FileOperationError`. Это не баг, но при добавлении новой функции придётся копировать ту же логику.

**Предложение:** Вынести общую обработку в контекстный менеджер или декоратор. Например:

```python
from contextlib import contextmanager

@contextmanager
def file_operation(action: str, file_path: str):
    try:
        yield
    except FileNotFoundError as e:
        raise FileOperationError(f"Файл не найден: {file_path}") from e
    except PermissionError as e:
        raise FileOperationError(f"Нет прав на {action} файла: {file_path}") from e
    except Exception as e:
        raise FileOperationError(f"Ошибка при {action} {file_path}: {e}") from e
```

Тогда:

```python
def read_binary_file(file_path: str) -> bytes:
    with file_operation("чтении", file_path):
        with open(file_path, "rb") as f:
            return f.read()
```

**nit:** это вопрос вкуса. Текущий вариант тоже рабочий, просто многословный.

---

## Замечание 5: `os.makedirs` создаёт директории даже для пустого пути (см. замечание 1), а также нет проверки на `None`

**Файл:** `file_utils.py`
**Строка:** 76
**Тип проблемы:** Потенциальная ошибка
**Важность:** Квадрант II (низкая важность, высокая конфликтность)

**Описание:**
Дополнение к замечанию 1: если `file_path` — это, например, `"/"` или `"."`, `os.path.dirname` вернёт `/` или пустую строку. В случае `/` `makedirs` не упадёт, но в случае пустой строки — упадёт. Это частный случай замечания 1, но стоит отметить отдельно.

**Предложение:** См. замечание 1.

---




# Code Review: `IDEA_utils.py`

## Общая информация

**Файл:** `IDEA_utils.py`
**Назначение:** Утилиты для шифрования/расшифрования алгоритмом IDEA в режиме CBC: генерация ключа и IV, паддинг ANSI X.923, шифрование и расшифрование.

---

## Замечание 1: `make_padding` падает на пустых данных — противоречие с ANSI X.923

**Файл:** `IDEA_utils.py`
**Строка:** 79
**Тип проблемы:** Ошибка логики
**Важность:** Квадрант IV (высокая важность, высокая конфликтность)

**Описание:**
Функция `make_padding` выбрасывает `ValueError`, если `data` пустая. Но ANSI X.923 **корректно работает с пустыми данными**: он добавляет целый блок паддинга (8 байт: 7 нулей и байт `0x08`). То есть `make_padding(b"", 64)` должен вернуть 8 байт, а не падать. Это ломает шифрование пустых сообщений, что является валидным сценарием.

**Предложение:** Убрать проверку `if len(data) == 0: raise ValueError(...)`. Паддинг сам добавит блок.

**Пример кода:**
```python
# Удалить этот блок:
if len(data) == 0:
    raise ValueError("Нельзя добавить дополнение к пустым данным")
```

---

## Замечание 2: `make_unpadding` падает на пустых данных — противоречие с ANSI X.923

**Файл:** `IDEA_utils.py`
**Строка:** 105
**Тип проблемы:** Ошибка логики
**Важность:** Квадрант IV (высокая важность, высокая конфликтность)

**Описание:**
Аналогично замечанию 1: `make_unpadding` выбрасывает `ValueError`, если `data_with_padding` пустая. Но после расшифрования пустого сообщения (которое было зашифровано с паддингом) мы получим как раз один блок (8 байт) паддинга, и `make_unpadding` должен вернуть `b""`. Проверка `len == 0` здесь не нужна и ломает валидный сценарий.

**Предложение:** Убрать проверку `if len(data_with_padding) == 0: raise ValueError(...)`.

---

## Замечание 3: `make_padding` проверяет `block_size_bits % 8` внутри `case 64`

**Файл:** `IDEA_utils.py`
**Строки:** 83–84
**Тип проблемы:** Мёртвый код
**Важность:** Квадрант III (низкая важность, низкая конфликтность)

**Описание:**
Внутри `case 64:` проверяется `if block_size_bits % 8 != 0`. Но `64 % 8 == 0` всегда. Эта проверка никогда не сработает. Она имеет смысл только до `match`, для произвольного `block_size_bits`, но не внутри `case 64`.

**Предложение:** Удалить проверку или вынести её до `match` (хотя при `match` на конкретное значение она бессмысленна).

---




# Code Review: `RSA_utils.py`

## Общая информация

**Файл:** `RSA_utils.py`
**Назначение:** Утилиты для работы с RSA: генерация ключей, шифрование/расшифрование (OAEP + SHA-256), сериализация/десериализация ключей в PEM, вспомогательные функции чтения/записи бинарных файлов.

---

## Замечание 1: `serialize_public_key` / `serialize_private_key` падают при `file_path` без директории

**Файл:** `RSA_utils.py`
**Строки:** 94, 116
**Тип проблемы:** Ошибка логики
**Важность:** Квадрант IV (высокая важность, высокая конфликтность)

**Описание:**
`os.path.dirname(file_path)` возвращает пустую строку, если в `file_path` нет директории (например, `"public.pem"`). `os.makedirs("", exist_ok=True)` выбрасывает `FileNotFoundError`. То есть сохранение ключа в текущую директорию сломается.

**Предложение:** Проверять, что `dirname` не пустой, перед созданием директорий.

**Пример кода:**
```python
dir_path = os.path.dirname(file_path)
if dir_path:
    os.makedirs(dir_path, exist_ok=True)
```

**Примечание:** Это замечание дублирует аналогичную проблему в `file_utils.py` (замечание 1). См. `file_utils.py` для подробностей.

---

## Замечание 2: `_write_binary_file` не создаёт директории, хотя вызывающий код на это рассчитывает

**Файл:** `RSA_utils.py`
**Строки:** 94–103, 116–125, 177–184
**Тип проблемы:** Дублирование логики / несогласованность
**Важность:** Квадрант II (низкая важность, высокая конфликтность)

**Описание:**
`serialize_public_key` и `serialize_private_key` создают директории через `os.makedirs`, а затем вызывают `_write_binary_file`, которая просто открывает файл. Это дублирование: если кто-то вызовет `_write_binary_file` напрямую, директории не создадутся. Логичнее вынести создание директорий внутрь `_write_binary_file`.

**Предложение:** Перенести `os.makedirs` в `_write_binary_file`:

```python
def _write_binary_file(file_path: str, data: bytes) -> None:
    """Внутренняя функция для записи бинарных данных."""
    dir_path = os.path.dirname(file_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    try:
        with open(file_path, "wb") as f:
            f.write(data)
    except OSError as e:
        raise OSError(f"Ошибка записи в файл {file_path}: {e}") from e
```

Тогда в `serialize_*` останется только сериализация.

**nit:** Это вопрос вкуса. Текущий вариант рабочий, но менее надёжный.

---

## Замечание 3: `load_public_key` / `load_private_key` не проверяют, что загруженный ключ — RSA

**Файл:** `RSA_utils.py`
**Строки:** 140, 162
**Тип проблемы:** Несоответствие контракта и реальности
**Важность:** Квадрант I (высокая важность, низкая конфликтность)

**Описание:**
Аннотация возвращает `rsa.RSAPublicKey` / `rsa.RSAPrivateKey`, но `load_pem_public_key` и `load_pem_private_key` могут вернуть ключ **любого** типа (Ed25519, ECDSA, DSA). Если в файле окажется не RSA-ключ, функция вернёт его без ошибок, а вызывающий код упадёт при попытке использовать RSA-специфичные методы.

**Предложение:** Проверять тип после загрузки.

**Пример кода:**
```python
public_key = load_pem_public_key(pem_data)
if not isinstance(public_key, rsa.RSAPublicKey):
    raise ValueError(f"Ожидался RSA-ключ, получен {type(public_key).__name__}")
return public_key
```

---

## Замечание 4: `encrypt_with_public_key` / `decrypt_with_private_key` не проверяют тип ключа

**Файл:** `RSA_utils.py`
**Строки:** 37, 67
**Тип проблемы:** Несоответствие контракта и реальности
**Важность:** Квадрант II (низкая важность, высокая конфликтность)

**Описание:**
Аннотация говорит `public_key: rsa.RSAPublicKey`, но нет проверки `isinstance`. Если передать ключ другого типа, `public_key.encrypt` может выбросить `AttributeError`, который не пойман в `except TypeError` и уйдёт в `except Exception` с сообщением «Неожиданная ошибка» — что неинформативно.

**Предложение:** Добавить явную проверку типа в начало функции:

```python
if not isinstance(public_key, rsa.RSAPublicKey):
    raise TypeError(f"Ожидался RSAPublicKey, получен {type(public_key).__name__}")
```

---




# Code Review: `interface.py`

## Общая информация

**Файл:** `interface.py`
**Назначение:** GUI-приложение на `tkinter` для гибридного шифрования IDEA + RSA. Управляет генерацией ключей, выбором пользовательских ключей, шифрованием и расшифрованием файлов.

---

## Замечание 1: `_get_path_by_position` возвращает `None`, если позиции нет в конфиге

**Файл:** `interface.py`
**Строки:** 110–130
**Тип проблемы:** Ошибка логики
**Важность:** Квадрант IV (высокая важность, высокая конфликтность)

**Описание:**
Метод `_get_path_by_position` в `match position` для `case 0/1/2` возвращает пользовательский путь, только если он задан. Иначе — падает в `return path_values[position]`. Если в конфиге меньше трёх путей, это выбросит `IndexError`. Аналогично `_get_param_by_position` — если в `crypto_params` меньше пяти параметров, будет `IndexError`. Это не баг конфига, а баг кода: нет проверки границ.

**Предложение:** Проверять длину списка перед доступом по индексу.

**Пример кода:**
```python
def _get_path_by_position(self, position: int) -> str:
    path_values = self._get_path_list()
    if position >= len(path_values):
        raise ValueError(f"В конфиге нет пути с позицией {position}")
    ...
```

---

## Замечание 2: `_get_path_by_position` использует магические числа 0, 1, 2

**Файл:** `interface.py`
**Строки:** 110–130
**Тип проблемы:** Магические числа, плохая читаемость
**Важность:** Квадрант II (низкая важность, высокая конфликтность)

**Описание:**
Позиции 0, 1, 2 в `_get_path_by_position` и 0–4 в `_get_param_by_position` — это магические числа. Непонятно, что соответствует приватному ключу, публичному, зашифрованному IDEA-ключу и т.д. Если порядок в конфиге изменится, придётся искать все использования.

**Предложение:** Ввести константы или `Enum`:

```python
class PathPosition:
    PRIVATE_KEY = 0
    PUBLIC_KEY = 1
    ENCRYPTED_KEY = 2

class ParamPosition:
    RSA_KEY_SIZE = 0
    RSA_EXPONENT = 1
    IDEA_KEY_SIZE = 2
    IV_SIZE = 3
    BLOCK_SIZE_BITS = 4
```

---

## Замечание 3: `encrypt_file` не проверяет совместимость `rsa_key_size` и OAEP

**Файл:** `interface.py`
**Строки:** 175–210 (косвенно — через `RSA_utils.encrypt_with_public_key`)
**Тип проблемы:** Потенциальный баг, неинформативная ошибка
**Важность:** Квадрант I (высокая важность, низкая конфликтность)

**Описание:**
В `encrypt_file` ключ IDEA (16 байт) шифруется RSA с OAEP + SHA-256. OAEP с SHA-256 может зашифровать не более `key_size/8 - 2*32 - 2` байт. Для ключа 2048 бит это ~190 байт — 16 байт влезает. Но если в конфиге `rsa_key_size` окажется меньше 1024 бит (например, 512), OAEP не сможет зашифровать даже 16 байт. Ошибка будет выброшена из `encrypt_with_public_key`, но сообщение будет неинформативным («Ошибка шифрования RSA: ...»).

**Предложение:** Либо проверять совместимость размеров в `encrypt_file` до вызова шифрования, либо добавить более понятное сообщение об ошибке в `RSA_utils.encrypt_with_public_key`.

**Пример кода (проверка в `encrypt_file`):**
```python
# Максимальный размер данных для OAEP + SHA-256: key_size/8 - 2*32 - 2
max_oaep_size = rsa_key_size // 8 - 2 * 32 - 2
if idea_key_size > max_oaep_size:
    raise ValueError(
        f"Ключ IDEA ({idea_key_size} байт) не влезает в OAEP "
        f"с ключом RSA {rsa_key_size} бит (макс. {max_oaep_size} байт)"
    )
```

---

## Замечание 4: `decrypt_file` использует `str.replace` для формирования имени выходного файла

**Файл:** `interface.py`
**Строка:** 245
**Тип проблемы:** Потенциальная ошибка логики
**Важность:** Квадрант III (низкая важность, низкая конфликтность)

**Описание:**
`output_file = input_file.replace(".enc", ".dec")` заменяет **все** вхождения `.enc` в пути, а не только расширение. Если файл называется `my.enc.backup.enc`, результат будет `my.dec.backup.dec`. Это не критично, но может удивить пользователя.

**Предложение:** Использовать `os.path.splitext`:

```python
base, ext = os.path.splitext(input_file)
if ext == ".enc":
    output_file = base + ".dec"
else:
    output_file = input_file + ".dec"
```

---

## Замечание 5: `create_gui` устанавливает `self.root.quit` как команду для кнопки «Выход», но не закрывает окно

**Файл:** `interface.py`
**Строка:** 320
**Тип проблемы:** Потенциальная проблема с UX
**Важность:** Квадрант III (низкая важность, низкая конфликтность)

**Описание:**
`self.root.quit` останавливает `mainloop`, но **не закрывает окно**. Если после `quit` останутся другие `mainloop` (что маловероятно в этом приложении), окно может остаться висеть. Обычно для закрытия используют `self.root.destroy`.

**Предложение:** Заменить `self.root.quit` на `self.root.destroy`.

---

## Замечание 6: `main` проверяет `os.path.exists` для конфига, но не проверяет, что это файл

**Файл:** `interface.py`
**Строка:** 360
**Тип проблемы:** Вводящее в заблуждение поведение
**Важность:** Квадрант III (низкая важность, низкая конфликтность)

**Описание:**
`os.path.exists` вернёт `True` и для директории. Если передать путь к директории, `read_json_file` упадёт с `IsADirectoryError`, который не пойман в `_load_config` (там ловится только `FileOperationError`). Приложение упадёт с неинформативной ошибкой.

**Предложение:** Использовать `os.path.isfile`:

```python
if not os.path.isfile(args.config_file):
    print(f"Ошибка: Файл конфигурации {args.config_file} не найден или не является файлом")
    return
```

---