import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
from tkinter import simpledialog
import threading
from utils.config_loader import ConfigLoader
from utils.file_utils import FileManager
from utils.key_manager import KeyManager
from crypto.hybrid import HybridCrypto
from crypto.constants import AVAILABLE_ALGORITHMS

class CryptoGUI:
    """Главный класс графического приложения."""
    
    def __init__(self):
        """Инициализация GUI."""
        self.root = tk.Tk()
        self.config_loader = ConfigLoader()
        self.config = self.config_loader.load("config.json")
        self.root.title("Гибридная криптосистема")
        self.root.geometry("600x500")
        self.root.resizable(True, True)
        self.file_manager = FileManager(self.config)
        private_path = self.file_manager.get_path('private_rsa_key')
        public_path = self.file_manager.get_path('public_rsa_key')
        self.key_manager = KeyManager(self.config, self.file_manager, private_path, public_path)
        self.hybrid_crypto = HybridCrypto(self.config)
        self.selected_algorithm = tk.StringVar()
        self.decrypt_algorithm = tk.StringVar()
        self.ciphertext_path = tk.StringVar()
        self.key_file_path = tk.StringVar()
        
        self.save_enc_file_path = tk.StringVar()
        self.save_key_file_path = tk.StringVar()
        
        self._setup_gui()
        self._init_keys()
    
    def _setup_gui(self) -> None:
        """Создает элементы интерфейса."""
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        self._create_encrypt_tab()
        self._create_decrypt_tab()
        self._create_keys_tab()
    
    def _create_encrypt_tab(self) -> None:
        """Создает вкладку 'Шифрование'."""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text='Шифрование')
        
        ttk.Label(frame, text='Алгоритм шифрования:').grid(row=0, column=0, sticky='w', padx=10, pady=5)
        algo_combo = ttk.Combobox(frame, textvariable=self.selected_algorithm, values=AVAILABLE_ALGORITHMS, state='readonly')
        algo_combo.grid(row=0, column=1, sticky='ew', padx=10, pady=5)
        self.selected_algorithm.set(AVAILABLE_ALGORITHMS[0])
        
        ttk.Label(frame, text='Текст для шифрования:').grid(row=1, column=0, sticky='nw', padx=10, pady=5)
        self.plaintext_text = scrolledtext.ScrolledText(frame, height=8, width=60)
        self.plaintext_text.grid(row=2, column=0, columnspan=2, padx=10, pady=5, sticky='nsew')
        
        save_frame = ttk.LabelFrame(frame, text='Куда сохранить', padding=10)
        save_frame.grid(row=3, column=0, columnspan=2, sticky='ew', padx=10, pady=5)
        
        ttk.Label(save_frame, text='Файл с шифротекстом:').grid(row=0, column=0, sticky='w', padx=5, pady=2)
        ttk.Entry(save_frame, textvariable=self.save_enc_file_path, width=40).grid(row=0, column=1, padx=5, pady=2)
        ttk.Button(save_frame, text='Обзор', command=self.select_save_enc_file).grid(row=0, column=2, padx=5, pady=2)
        
        ttk.Label(save_frame, text='Файл с ключом:').grid(row=1, column=0, sticky='w', padx=5, pady=2)
        ttk.Entry(save_frame, textvariable=self.save_key_file_path, width=40).grid(row=1, column=1, padx=5, pady=2)
        ttk.Button(save_frame, text='Обзор', command=self.select_save_key_file).grid(row=1, column=2, padx=5, pady=2)
        
        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=4, column=0, columnspan=2, pady=10)
        ttk.Button(btn_frame, text='Загрузить из файла', command=self.load_plaintext_file).pack(side='left', padx=5)
        ttk.Button(btn_frame, text='Зашифровать', command=self.encrypt_data).pack(side='left', padx=5)
        
        ttk.Label(frame, text='Результат:').grid(row=5, column=0, sticky='nw', padx=10, pady=5)
        self.result_text = scrolledtext.ScrolledText(frame, height=5, width=60)
        self.result_text.grid(row=6, column=0, columnspan=2, padx=10, pady=5, sticky='nsew')
        
        frame.grid_columnconfigure(1, weight=1)
        frame.grid_rowconfigure(2, weight=1)
        frame.grid_rowconfigure(6, weight=1)
    
    def _create_decrypt_tab(self) -> None:
        """Создает вкладку 'Расшифрование'."""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text='Расшифрование')
        
        ttk.Label(frame, text='Алгоритм шифрования:').grid(row=0, column=0, sticky='w', padx=10, pady=5)
        algo_combo = ttk.Combobox(frame, textvariable=self.decrypt_algorithm, values=AVAILABLE_ALGORITHMS, state='readonly')
        algo_combo.grid(row=0, column=1, sticky='ew', padx=10, pady=5)
        self.decrypt_algorithm.set(AVAILABLE_ALGORITHMS[0])
        
        info_frame = ttk.LabelFrame(frame, text='Зашифрованные файлы', padding=10)
        info_frame.grid(row=1, column=0, columnspan=2, sticky='ew', padx=10, pady=10)
        
        ttk.Label(info_frame, text='Файл с шифротекстом:').grid(row=0, column=0, sticky='w')
        ttk.Entry(info_frame, textvariable=self.ciphertext_path, width=40).grid(row=0, column=1, padx=5)
        ttk.Button(info_frame, text='Обзор', command=self.select_ciphertext_file).grid(row=0, column=2)
        
        ttk.Label(info_frame, text='Файл с ключом:').grid(row=1, column=0, sticky='w')
        ttk.Entry(info_frame, textvariable=self.key_file_path, width=40).grid(row=1, column=1, padx=5)
        ttk.Button(info_frame, text='Обзор', command=self.select_key_file).grid(row=1, column=2)
        
        ttk.Button(frame, text='Расшифровать', command=self.decrypt_data).grid(row=2, column=0, columnspan=2, pady=10)
        
        ttk.Label(frame, text='Расшифрованный текст:').grid(row=3, column=0, sticky='nw', padx=10, pady=5)
        self.decrypted_text = scrolledtext.ScrolledText(frame, height=12, width=60)
        self.decrypted_text.grid(row=4, column=0, columnspan=2, padx=10, pady=5, sticky='nsew')
        
        frame.grid_columnconfigure(1, weight=1)
        frame.grid_rowconfigure(4, weight=1)
    
    def _create_keys_tab(self) -> None:
        """Создает вкладку 'Управление ключами'."""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text='Управление ключами')
        
        rsa_frame = ttk.LabelFrame(frame, text='RSA ключи', padding=10)
        rsa_frame.pack(fill='x', padx=10, pady=10)
        
        rsa_section = self.config['crypto']['rsa'][0]
        ttk.Label(rsa_frame, text='Размер RSA ключа:').pack(side='left', padx=5)
        ttk.Label(rsa_frame, text=f'{rsa_section} бит').pack(side='left', padx=5)
        ttk.Button(rsa_frame, text='Сгенерировать новые RSA ключи', command=self.generate_rsa_keys).pack(side='left', padx=20)
        
        info_frame = ttk.LabelFrame(frame, text='Информация о ключах', padding=10)
        info_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.key_info_text = scrolledtext.ScrolledText(info_frame, height=10, width=70)
        self.key_info_text.pack(fill='both', expand=True)
        
        ttk.Button(frame, text='Обновить информацию', command=self.refresh_key_info).pack(pady=10)
        self.refresh_key_info()
    
    def _init_keys(self) -> None:
        """Инициализация ключей."""
        self.key_manager.ensure_rsa_keys_exist()
    
    def load_plaintext_file(self) -> None:
        """Загружает текст из файла."""
        file_path = filedialog.askopenfilename(
            title='Выберите файл с текстом',
            filetypes=[('Текстовые файлы', '*.txt'), ('Все файлы', '*.*')]
        )
        
        if not file_path:
            return
            
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        self.plaintext_text.delete('1.0', tk.END)
        self.plaintext_text.insert('1.0', content)
        messagebox.showinfo('Успех', f'Загружено {len(content)} символов')
    
    def select_save_enc_file(self) -> None:
        """Выбор файла для сохранения шифротекста."""
        file_path = filedialog.asksaveasfilename(
            title='Сохранить шифротекст как',
            defaultextension='.bin',
            filetypes=[('Бинарные файлы', '*.bin'), ('Все файлы', '*.*')]
        )
        if not file_path:
            return
        self.save_enc_file_path.set(file_path)
    
    def select_save_key_file(self) -> None:
        """Выбор файла для сохранения зашифрованного ключа."""
        file_path = filedialog.asksaveasfilename(
            title='Сохранить зашифрованный ключ как',
            defaultextension='.bin',
            filetypes=[('Бинарные файлы', '*.bin'), ('Все файлы', '*.*')]
        )
        if not file_path:
            return
        self.save_key_file_path.set(file_path)
    
    def encrypt_data(self) -> None:
        """Шифрует текст."""
        algorithm = self.selected_algorithm.get()
        plaintext = self.plaintext_text.get('1.0', tk.END).strip()
        enc_file = self.save_enc_file_path.get()
        key_file = self.save_key_file_path.get()
        
        if not plaintext:
            messagebox.showwarning('Предупреждение', 'Введите текст для шифрования')
            return
        
        if not algorithm:
            messagebox.showwarning('Предупреждение', 'Выберите алгоритм')
            return
        
        if not enc_file:
            messagebox.showwarning('Предупреждение', 'Выберите файл для сохранения шифротекста')
            return
        
        if not key_file:
            messagebox.showwarning('Предупреждение', 'Выберите файл для сохранения ключа')
            return
        
        cipher = self.hybrid_crypto.get_cipher(algorithm)
        key_size = cipher.get_key_size()
        
        key_hex = simpledialog.askstring(
            "Свой ключ",
            f"Введите ключ {algorithm} в hex ({key_size*2} символов)\n"
            f"или оставьте пустым для генерации случайного ключа:"
        )
        
        if key_hex is None:
            return
        
        custom_key = None
        hex_str = key_hex.strip()
        if hex_str:
            try:
                custom_key = bytes.fromhex(hex_str)
                if len(custom_key) != key_size:
                    messagebox.showerror('Ошибка', f'Ключ должен быть {key_size} байт')
                    return
            except ValueError:
                messagebox.showerror('Ошибка', 'Неверный hex формат')
                return
        
        def encrypt_thread() -> None:
            """Поток шифрования."""
            try:
                public_key = self.key_manager.load_public_key()
                
                if custom_key is None:
                    symmetric_key = self.hybrid_crypto.get_cipher(algorithm).generate_key()
                else:
                    symmetric_key = custom_key
                
                encrypted_data, encrypted_key = self.hybrid_crypto.encrypt_hybrid(
                    plaintext.encode('utf-8'), algorithm, public_key, custom_key
                )
                
                self.file_manager.write_file(enc_file, encrypted_data, binary=True)
                self.file_manager.write_file(key_file, encrypted_key, binary=True)
                
                self.root.after(0, lambda: self._show_encrypt_result(encrypted_data, encrypted_key, enc_file, key_file))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror('Ошибка шифрования', str(e)))
        
        threading.Thread(target=encrypt_thread, daemon=True).start()
        messagebox.showinfo('Обработка', 'Шифрование выполняется')
    
    def select_ciphertext_file(self) -> None:
        """Выбор файла с шифротекстом."""
        file_path = filedialog.askopenfilename(
            title='Выберите файл с шифротекстом',
            filetypes=[('Все файлы', '*.*'), ('Бинарные файлы', '*.bin')]
        )
        if not file_path:
            return
        self.ciphertext_path.set(file_path)
    
    def select_key_file(self) -> None:
        """Выбор файла с ключом."""
        file_path = filedialog.askopenfilename(
            title='Выберите файл с зашифрованным ключом',
            filetypes=[('Все файлы', '*.*'), ('Бинарные файлы', '*.bin')]
        )
        if not file_path:
            return
        self.key_file_path.set(file_path)
    
    def _show_encrypt_result(self, encrypted_data: bytes, encrypted_key: bytes, enc_file: str, key_file: str) -> None:
        """Показывает результат шифрования."""
        self.result_text.delete('1.0', tk.END)
        
        preview = encrypted_data[:100].hex()
        if len(encrypted_data) > 100:
            preview += "..."
        
        result = f"Шифрование выполнено успешно!\n\n"
        result += f"Алгоритм: {self.selected_algorithm.get()}\n"
        result += f"Размер зашифрованных данных: {len(encrypted_data)} байт\n"
        result += f"Размер зашифрованного ключа: {len(encrypted_key)} байт\n\n"
        result += f"Сохранено в:\n  • {enc_file}\n  • {key_file}\n\n"
        result += f"Предпросмотр:\n{preview}"
        
        self.result_text.insert('1.0', result)
        messagebox.showinfo('Успех', 'Шифрование завершено успешно!')
    
    def decrypt_data(self) -> None:
        """Расшифровывает данные."""
        ciphertext_file = self.ciphertext_path.get()
        key_file = self.key_file_path.get()
        algorithm = self.decrypt_algorithm.get()
        
        if not ciphertext_file:
            messagebox.showwarning('Предупреждение', 'Выберите файл с шифротекстом')
            return
        
        if not key_file:
            messagebox.showwarning('Предупреждение', 'Выберите файл с ключом')
            return
        
        if not algorithm:
            messagebox.showwarning('Предупреждение', 'Выберите алгоритм')
            return
        
        def decrypt_thread() -> None:
            """Поток расшифрования."""
            try:
                encrypted_data = self.file_manager.read_file(ciphertext_file, binary=True)
                encrypted_key = self.file_manager.read_file(key_file, binary=True)
                private_key = self.key_manager.load_private_key()
                plaintext_bytes = self.hybrid_crypto.decrypt_hybrid(
                    encrypted_data, encrypted_key, algorithm, private_key
                )
                plaintext = plaintext_bytes.decode('utf-8')
                self.root.after(0, lambda: self._show_decrypt_result(plaintext))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror('Ошибка расшифрования', str(e)))
        
        threading.Thread(target=decrypt_thread, daemon=True).start()
        messagebox.showinfo('Обработка', 'Расшифрование выполняется')
    
    def _show_decrypt_result(self, plaintext: str) -> None:
        """Показывает результат расшифрования."""
        self.decrypted_text.delete('1.0', tk.END)
        self.decrypted_text.insert('1.0', plaintext)
        messagebox.showinfo('Успех', f'Расшифрование завершено! Получено {len(plaintext)} символов')
        
        save_result = messagebox.askyesno('Сохранение', 'Сохранить расшифрованный текст в файл?')
        if save_result:
            file_path = filedialog.asksaveasfilename(
                title='Сохранить расшифрованный текст',
                defaultextension='.txt',
                filetypes=[('Текстовые файлы', '*.txt'), ('Все файлы', '*.*')]
            )
            if not file_path:
                return
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(plaintext)
            messagebox.showinfo('Успех', f'Сохранено в {file_path}')
    
    def generate_rsa_keys(self) -> None:
        """Генерирует новые RSA ключи."""
        confirm = messagebox.askyesno('Подтверждение', 'Сгенерировать новые RSA ключи? Старые ключи будут перезаписаны!')
        if confirm:
            try:
                self.key_manager.generate_and_save_rsa_keys()
                self.refresh_key_info()
                messagebox.showinfo('Успех', 'Новые RSA ключи успешно сгенерированы!')
            except Exception as e:
                messagebox.showerror('Ошибка', f'Не удалось сгенерировать ключи: {e}')
    
    def refresh_key_info(self) -> None:
        """Обновляет информацию о ключах."""
        self.key_info_text.delete('1.0', tk.END)
        
        try:
            public_key = self.key_manager.load_public_key()
            private_path, public_path = self.key_manager.get_key_paths()
            
            info = f"Информация о RSA ключах\n{'='*40}\n\n"
            info += f"Размер ключа: {public_key.key_size} бит\n"
            info += f"Путь к публичному ключу: {public_path}\n"
            info += f"Путь к приватному ключу: {private_path}\n\n"
            info += f"Доступные симметричные алгоритмы:\n"
            info += f"  • SEED: 16-байтный ключ\n"
            info += f"  • ChaCha20: 32-байтный ключ\n"
            
            self.key_info_text.insert('1.0', info)
        except Exception as e:
            self.key_info_text.insert('1.0', f'Не удалось загрузить информацию о ключах: {e}')
    
    def run(self) -> None:
        """Запускает приложение."""
        self.root.mainloop()

def main() -> None:
    """Главная функция."""
    app = CryptoGUI()
    app.run()

if __name__ == "__main__":
    main()