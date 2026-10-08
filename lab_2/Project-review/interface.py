import argparse
import os
import tkinter as tk
from tkinter import filedialog
from typing import Optional, Dict, Any, List

from RSA_utils import (
    generate_keys,
    serialize_private_key,
    serialize_public_key,
    load_private_key,
    load_public_key,
    encrypt_with_public_key,
    decrypt_with_private_key,
)
from IDEA_utils import (
    generate_key_for_IDEA,
    generate_iv,
    encrypt_data,
    decrypt_data
)
from file_utils import (
    read_json_file,
    read_binary_file,
    write_binary_file,
    FileOperationError
)
from check_error_utils import (
    handle_errors,
    show_info,
    show_error
)


class CryptoApplication:
    """
    Управление гибридной криптосистемой IDEA + RSA.

    Обеспечивает генерацию ключей, шифрование и расшифрование файлов
    с использованием гибридной схемы: RSA для шифрования ключа IDEA,
    IDEA для шифрования данных.

    Attributes:
        config_file: Путь к файлу конфигурации.
        _full_config: Полная конфигурация.
        _user_private_key_path: Пользовательский путь к приватному ключу.
        _user_public_key_path: Пользовательский путь к публичному ключу.
        _user_encrypted_key_path: Пользовательский путь к зашифрованному ключу IDEA.
        root: Корневое окно GUI.
        status_label: Метка для отображения статуса.
    """

    def __init__(self, config_file: str = "settings.json") -> None:
        """
        Инициализирует приложение и загружает конфигурацию.

        Args:
            config_file: Путь к файлу конфигурации JSON. По умолчанию "settings.json".
        """
        self.config_file = config_file
        self._full_config = self._load_config()

        self._user_private_key_path: Optional[str] = None
        self._user_public_key_path: Optional[str] = None
        self._user_encrypted_key_path: Optional[str] = None

        self.root: Optional[tk.Tk] = None
        self.status_label: Optional[tk.Label] = None

    def _load_config(self) -> Dict[str, Any]:
        """
        Загружает конфигурацию из JSON файла.

        Returns:
            Словарь с данными конфигурации.
        """
        try:
            return read_json_file(self.config_file)
        except FileOperationError as e:
            show_error("Критическая ошибка", str(e))
            raise SystemExit(1)

    def _get_all_paths(self) -> Dict[str, str]:
        """Возвращает все пути из конфигурации."""
        return self._full_config.get("paths", {})

    def _get_all_params(self) -> Dict[str, Any]:
        """Возвращает все параметры из конфигурации."""
        return self._full_config.get("crypto_params", {})

    def _get_path_list(self) -> List[str]:
        """Возвращает список путей из конфигурации."""
        return list(self._get_all_paths().values())

    def _get_param_list(self) -> List[Any]:
        """Возвращает список параметров из конфигурации."""
        return list(self._get_all_params().values())

    def _update_status(self, message: str) -> None:
        """Обновляет текст в строке статуса GUI."""
        if self.status_label:
            self.status_label.config(text=message)
            self.root.update_idletasks()

    def _get_path_by_position(self, position: int) -> str:
        """
        Возвращает путь по позиции с учётом пользовательских переопределений.
        
        Args:
            position: Позиция пути в списке.

        Returns:
            Путь к файлу ключа.
        """
        path_values = self._get_path_list()

        match position:
            case 0:
                if self._user_private_key_path:
                    return self._user_private_key_path
            case 1:
                if self._user_public_key_path:
                    return self._user_public_key_path
            case 2:
                if self._user_encrypted_key_path:
                    return self._user_encrypted_key_path

        return path_values[position]

    def _get_param_by_position(self, position: int) -> Any:
        """
        Возвращает параметр по позиции из crypto_params.

        Args:
            position: Позиция параметра в списке.

        Returns:
            Значение параметра.
        """
        param_values = self._get_param_list()
        return param_values[position]

    @handle_errors("Ошибка генерации ключей")
    def gen_keys(self) -> None:
        """
        Генерирует новую пару RSA ключей и симметричный ключ IDEA.

        Создаёт:
            Приватный ключ RSA (PEM формат)
            Публичный ключ RSA (PEM формат)
            Ключ IDEA, зашифрованный публичным ключом RSA (бинарный файл)
        """
        self._update_status("Генерация ключей")

        priv_key_path = self._get_path_by_position(0)
        pub_key_path = self._get_path_by_position(1)
        enc_key_path = self._get_path_by_position(2)

        rsa_key_size = self._get_param_by_position(0)
        rsa_exponent = self._get_param_by_position(1)
        idea_key_size = self._get_param_by_position(2)

        idea_key = generate_key_for_IDEA(idea_key_size)
        private_rsa, public_rsa = generate_keys(rsa_key_size, rsa_exponent)

        serialize_private_key(private_rsa, priv_key_path)
        serialize_public_key(public_rsa, pub_key_path)

        encrypted_idea_key = encrypt_with_public_key(idea_key, public_rsa)
        write_binary_file(enc_key_path, encrypted_idea_key)

        self._user_private_key_path = None
        self._user_public_key_path = None
        self._user_encrypted_key_path = None

        self._update_status("Ключи сгенерированы")
        show_info("Успех", f"Ключи успешно сгенерированы!\n\n"
                  f"Приватный ключ: {priv_key_path}\n"
                  f"Публичный ключ: {pub_key_path}\n"
                  f"Зашифрованный IDEA ключ: {enc_key_path}")

    @handle_errors("Ошибка шифрования файла")
    def encrypt_file(self) -> None:
        """
        Шифрует выбранный файл с использованием гибридной схемы.

        Процесс:
            1. Загружается приватный ключ RSA
            2. Расшифровывается ключ IDEA
            3. Генерируется IV
            4. Данные шифруются алгоритмом IDEA в режиме CBC
            5. IV и шифротекст сохраняются в один файл
        """
        input_file = filedialog.askopenfilename(
            title="Выберите файл для шифрования")
        if not input_file:
            return

        self._update_status("Шифрование")

        priv_key_path = self._get_path_by_position(0)
        enc_key_path = self._get_path_by_position(2)

        iv_size = self._get_param_by_position(3)
        block_size_bits = self._get_param_by_position(4)

        plaintext = read_binary_file(input_file)

        rsa_private_key = load_private_key(priv_key_path)
        encrypted_idea_key = read_binary_file(enc_key_path)
        idea_key = decrypt_with_private_key(
            encrypted_idea_key, rsa_private_key)

        iv = generate_iv(iv_size)
        ciphertext = encrypt_data(idea_key, iv, plaintext, block_size_bits)

        output_file = input_file + ".enc"
        write_binary_file(output_file, iv + ciphertext)

        self._update_status("Шифрование завершено")
        show_info("Успех", f"Файл зашифрован:\n{output_file}")

    @handle_errors("Ошибка расшифрования файла")
    def decrypt_file(self) -> None:
        """
        Расшифровывает выбранный файл.

        Процесс:
            1. Загружается приватный ключ RSA
            2. Расшифровывается ключ IDEA
            3. Из файла извлекаются IV и шифротекст
            4. Данные расшифровываются алгоритмом IDEA
            5. Удаляется padding, результат сохраняется в файл
        """
        input_file = filedialog.askopenfilename(
            title="Выберите зашифрованный файл")
        if not input_file:
            return

        self._update_status("Расшифрование")

        priv_key_path = self._get_path_by_position(0)
        enc_key_path = self._get_path_by_position(2)

        iv_size = self._get_param_by_position(3)
        block_size_bits = self._get_param_by_position(4)

        data = read_binary_file(input_file)
        iv = data[:iv_size]
        ciphertext = data[iv_size:]

        rsa_private_key = load_private_key(priv_key_path)
        encrypted_idea_key = read_binary_file(enc_key_path)
        idea_key = decrypt_with_private_key(
            encrypted_idea_key, rsa_private_key)

        plaintext = decrypt_data(idea_key, iv, ciphertext, block_size_bits)

        output_file = input_file.replace(".enc", ".dec")
        if output_file == input_file:
            output_file = input_file + ".dec"

        write_binary_file(output_file, plaintext)

        self._update_status("Расшифрование завершено")
        show_info("Успех", f"Файл расшифрован:\n{output_file}")

    def select_private_key(self) -> None:
        """Открывает диалог выбора пользовательского приватного ключа RSA."""
        filename = filedialog.askopenfilename(
            title="Выберите приватный ключ RSA",
            filetypes=[("PEM files", "*.pem"), ("All files", "*.*")]
        )
        if filename:
            self._user_private_key_path = filename
            self._update_status("Выбран приватный ключ")
            show_info("Успех", f"Выбран приватный ключ:\n{filename}")

    def select_public_key(self) -> None:
        """Открывает диалог выбора пользовательского публичного ключа RSA."""
        filename = filedialog.askopenfilename(
            title="Выберите публичный ключ RSA",
            filetypes=[("PEM files", "*.pem"), ("All files", "*.*")]
        )
        if filename:
            self._user_public_key_path = filename
            self._update_status("Выбран публичный ключ")
            show_info("Успех", f"Выбран публичный ключ:\n{filename}")

    def select_encrypted_key(self) -> None:
        """Открывает диалог выбора зашифрованного ключа IDEA."""
        filename = filedialog.askopenfilename(
            title="Выберите зашифрованный IDEA ключ",
            filetypes=[("Binary files", "*.bin"), ("All files", "*.*")]
        )
        if filename:
            self._user_encrypted_key_path = filename
            self._update_status("Выбран зашифрованный ключ")
            show_info("Успех", f"Выбран зашифрованный ключ:\n{filename}")

    def create_gui(self) -> None:
        """Создаёт графический интерфейс пользователя."""
        self.root = tk.Tk()
        self.root.title("Гибридная криптосистема IDEA + RSA")
        self.root.geometry("480x550")
        self.root.resizable(False, False)

        main_frame = tk.Frame(self.root, padx=10, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        title_label = tk.Label(
            main_frame,
            text="Гибридное шифрование\nIDEA + RSA",
            font=("Arial", 14, "bold"),
            justify="center"
        )
        title_label.pack(pady=(0, 15))

        buttons = [
            ("Сгенерировать ключи", self.gen_keys),
            ("Выбрать приватный ключ", self.select_private_key),
            ("Выбрать публичный ключ", self.select_public_key),
            ("Выбрать зашифрованный ключ IDEA", self.select_encrypted_key),
            ("Зашифровать файл", self.encrypt_file),
            ("Расшифровать файл", self.decrypt_file),
            ("Выход", self.root.quit)
        ]

        for text, command in buttons:
            btn = tk.Button(
                main_frame,
                text=text,
                command=command,
                width=40,
                height=1,
                bg="#f0f0f0",
                relief=tk.RAISED,
                cursor="hand2",
                font=("Arial", 10)
            )
            btn.pack(pady=4)

        security_note = tk.Label(
            main_frame,
            text="Ключи хранятся только в локальных переменных",
            font=("Arial", 8),
            fg="green"
        )
        security_note.pack(pady=(15, 5))

        self.status_label = tk.Label(
            main_frame,
            text="Готов к работе",
            font=("Arial", 8),
            fg="gray",
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X, pady=(5, 0))

        self.root.mainloop()


def main() -> None:
    """Главная функция запуска приложения."""
    parser = argparse.ArgumentParser(
        description="Гибридная криптосистема IDEA + RSA")
    parser.add_argument(
        "config_file",
        nargs="?",
        default="settings.json",
        help="Путь к файлу конфигурации"
    )
    args = parser.parse_args()

    if not os.path.exists(args.config_file):
        print(f"Ошибка: Файл конфигурации {args.config_file} не найден")
        return

    app = CryptoApplication(args.config_file)
    app.create_gui()


if __name__ == "__main__":
    main()
