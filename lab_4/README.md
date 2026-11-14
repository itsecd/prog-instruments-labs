### Сделанные изменения

1. **`crypto.py`** — магические числа заменены именованными константами:
   - `4096` → `CHUNK_SIZE`
   - `10000000` → `DEFAULT_MAX_ITERATIONS`
   - `".sha256"` → `HASH_FILE_EXTENSION`

2. **`gui.py`** — дублирующиеся вызовы `QMessageBox.critical`,
   `QMessageBox.warning`, `QMessageBox.information` вынесены в три хелпера
   `_show_error`, `_show_warning`, `_show_info`.

3. **`gui.py`** — дублирующаяся проверка «выбран ли файл» в `handle_save`
   и `handle_verify` вынесена в метод `_ensure_file_selected`.

4. **`gui.py`** — длинный метод `init_ui` разбит на три метода по секциям
   интерфейса: `_build_file_section`, `_build_collision_section`,
   `_build_log_section`.

5. **`cli.py`** — длинный метод `run_cli` разбит на `_create_parser` и четыре
   обработчика `_handle_collision`, `_handle_save`, `_handle_verify`,
   `_handle_print_hash`.


### Code smells, от которых избавлен исходный код

- **Magic Numbers** — числа без объяснения смысла в `crypto.py`.
- **Duplicated Code** — повторяющиеся вызовы `QMessageBox.*` и повторяющаяся
  проверка выбранного файла в `gui.py`.
- **Long Method** — `init_ui` и `run_cli` разбиты на короткие функции.
- **Нарушение Single Responsibility Principle** — `AppGui` и `run_cli`
  выполняли слишком много разнородных задач в одном методе.