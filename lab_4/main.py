import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.optim import Adam

from dataloader import create_dataloader
from models import PoetryModel, PoetryType


class PoetryTrainer:
    def __init__(self, config):
        """
        Инициализация тренера для модели генерации стихов

        Args:
            config (dict): словарь с конфигурационными параметрами
        """
        self.config = self._validate_config(config)
        self.device = self._setup_device()

        self.model = None
        self.optimizer = None
        self.criterion = None
        self.one_hot_embedding = None
        self.train_loader = None
        self.vocab = None
        self.current_epoch = 0
        self.best_loss = float('inf')

        # Метрики
        self.train_losses = []
        self.validation_losses = []

        self._setup()

    def _validate_config(self, config):
        """
        Проверяет и дополняет конфигурацию значениями по умолчанию

        Args:
            config: исходная конфигурация

        Returns:
            dict: проверенная и дополненная конфигурация

        Raises:
            ValueError: если обязательные параметры отсутствуют
        """
        required_params = ['batch_size', 'learning_rate', 'hidden_size', 'epochs']

        for param in required_params:
            if param not in config:
                raise ValueError(f"Обязательный параметр '{param}' отсутствует в конфигурации")

        default_config = {
            'dropout': 0.5,
            'model_path': 'model.pkl',
            'debug': False,
            'shuffle': True,
            'save_best_only': True,
            'early_stopping_patience': 10,
            'log_interval': 10,
            'validation_split': 0.1,
            'gradient_clip': 1.0
        }

        merged_config = {**default_config, **config}

        if merged_config['batch_size'] <= 0:
            raise ValueError(f"batch_size должен быть положительным, получен: {merged_config['batch_size']}")

        if merged_config['learning_rate'] <= 0:
            raise ValueError(f"learning_rate должен быть положительным, получен: {merged_config['learning_rate']}")

        if merged_config['epochs'] <= 0:
            raise ValueError(f"epochs должен быть положительным, получен: {merged_config['epochs']}")

        print("✅ Конфигурация validated успешно")
        return merged_config

    def _setup_device(self):
        """
        Настраивает устройство для вычислений (GPU/CPU)

        Returns:
            torch.device: выбранное устройство
        """
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"✅ Используется GPU: {torch.cuda.get_device_name()}")
        else:
            device = torch.device('cpu')
            print("✅ Используется CPU")

        return device

    def _setup(self):
        """
        Инициализирует все компоненты для обучения
        """
        print("🔄 Инициализация компонентов обучения...")

        try:
            self._setup_data()

            self._setup_model()

            self._setup_training_components()

            self._load_existing_model()

            print("✅ Все компоненты инициализированы успешно")

        except Exception as e:
            raise RuntimeError(f"Ошибка инициализации компонентов: {e}")

    def _setup_data(self):
        """
        Загружает и подготавливает данные
        """
        print("📊 Загрузка данных...")

        self.train_loader, self.vocab = create_dataloader(
            batch_size=self.config['batch_size'],
            debug=self.config['debug'],
            shuffle=self.config['shuffle']
        )

        self.vocab_size = len(self.vocab.stoi)
        print(f"✅ Данные загружены. Размер словаря: {self.vocab_size}")

        self.one_hot_embedding = nn.Embedding(
            self.vocab_size,
            self.vocab_size,
            _weight=torch.from_numpy(np.eye(self.vocab_size))
        ).to(self.device)

    def _setup_model(self):
        """
        Создает и настраивает модель
        """
        print("🧠 Создание модели...")

        self.model = PoetryModel(
            vocab_size=self.vocab_size,
            hidden_size=self.config['hidden_size'],
            output_size=self.vocab_size,
            dropout=self.config['dropout']
        ).to(self.device)

        # Вывод информации о модели
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print(f"✅ Модель создана. Параметры: {total_params:,} (обучаемые: {trainable_params:,})")

    def _setup_training_components(self):
        """
        Настраивает оптимизатор и функцию потерь
        """
        print("⚙️ Настройка компонентов обучения...")

        self.optimizer = Adam(
            self.model.parameters(),
            lr=self.config['learning_rate']
        )

        self.criterion = nn.CrossEntropyLoss()

        print(f"✅ Оптимизатор: Adam(lr={self.config['learning_rate']})")
        print(f"✅ Функция потерь: CrossEntropyLoss")

    def _load_existing_model(self):
        """
        Загружает существующую модель если она есть
        """
        model_path = self.config['model_path']

        if os.path.exists(model_path):
            print(f"🔄 Загрузка существующей модели из {model_path}...")

            try:
                self.model = torch.load(model_path, map_location=self.device)
                print("✅ Модель загружена успешно")

            except Exception as e:
                print(f"⚠️ Ошибка загрузки модели: {e}. Создаем новую модель.")
                self._setup_model()

    def train(self):
        """
        Запускает полный цикл обучения модели
        """
        model_path = self.config['model_path']

        if self._is_model_trained():
            print("✅ Модель уже обучена. Пропускаем обучение.")
            return

        print("🚀 Начало обучения...")
        print(f"📈 Эпох: {self.config['epochs']}, Batch size: {self.config['batch_size']}")

        patience_counter = 0
        self.best_loss = float('inf')

        for epoch in range(self.config['epochs']):
            self.current_epoch = epoch

            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)

            self._log_progress(epoch, train_loss)

            if self._should_save_model(train_loss):
                self._save_model(train_loss)
                patience_counter = 0
            else:
                patience_counter += 1

            if self._should_stop_early(patience_counter):
                print(f"🛑 Ранняя остановка на эпохе {epoch}")
                break

        print("✅ Обучение завершено!")
        self._print_training_summary()

    def _is_model_trained(self):
        """
        Проверяет, обучена ли уже модель

        Returns:
            bool: True если модель уже обучена
        """
        model_path = self.config['model_path']
        if os.path.exists(model_path) and not self.config['debug']:
            return True
        return False

    def train_epoch(self, epoch):
        """
        Обучение на одной эпохе

        Args:
            epoch: номер текущей эпохи

        Returns:
            float: средние потери на эпохе
        """
        self.model.train()
        total_loss = 0
        total_batches = len(self.train_loader)

        progress_bar = tqdm(
            self.train_loader,
            desc=f'Epoch {epoch + 1}/{self.config["epochs"]}',
            leave=False
        )

        for batch_idx, batch in enumerate(progress_bar):
            try:
                self.optimizer.zero_grad()

                sentences = batch.sent.t().to(self.device)
                x, y = sentences[:, :-1], sentences[:, 1:]

                x_one_hot = self.one_hot_embedding(x).float()

                init_hidden = torch.zeros(1, len(x), self.config['hidden_size']).to(self.device)
                output, _ = self.model(x_one_hot, init_hidden)

                output_flat = output.reshape(-1, output.shape[-1])
                y_flat = y.flatten()
                loss = self.criterion(output_flat, y_flat)

                loss.backward()

                if self.config['gradient_clip'] > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['gradient_clip']
                    )

                self.optimizer.step()

                total_loss += loss.item()

                current_loss = total_loss / (batch_idx + 1)
                progress_bar.set_postfix({
                    'loss': f'{current_loss:.4f}',
                    'batch': f'{batch_idx + 1}/{total_batches}'
                })

            except Exception as e:
                print(f"❌ Ошибка в батче {batch_idx}: {e}")
                continue

        avg_loss = total_loss / total_batches
        return avg_loss

    def _log_progress(self, epoch, train_loss):
        """
        Логирует прогресс обучения

        Args:
            epoch: номер эпохи
            train_loss: потери на обучении
        """
        log_interval = self.config['log_interval']

        if (epoch + 1) % log_interval == 0 or epoch == 0:
            print(f"📊 Эпоха {epoch + 1}/{self.config['epochs']} - Потери: {train_loss:.4f}")

    def _should_save_model(self, current_loss):
        """
        Определяет, нужно ли сохранять модель

        Args:
            current_loss: текущие потери

        Returns:
            bool: True если модель нужно сохранить
        """
        if not self.config['save_best_only']:
            return True

        return current_loss < self.best_loss

    def _save_model(self, current_loss):
        """
        Сохраняет модель

        Args:
            current_loss: текущие потери
        """
        model_path = self.config['model_path']

        try:
            torch.save(self.model, model_path)
            self.best_loss = current_loss
            print(f"💾 Модель сохранена (потери: {current_loss:.4f})")

        except Exception as e:
            print(f"❌ Ошибка сохранения модели: {e}")

    def _should_stop_early(self, patience_counter):
        """
        Проверяет условия для ранней остановки

        Args:
            patience_counter: счетчик терпения

        Returns:
            bool: True если нужно остановиться
        """
        patience = self.config['early_stopping_patience']
        return patience > 0 and patience_counter >= patience

    def _print_training_summary(self):
        """Выводит сводку по обучению"""
        if self.train_losses:
            initial_loss = self.train_losses[0]
            final_loss = self.train_losses[-1]
            improvement = initial_loss - final_loss

            print(f"\n📈 Сводка обучения:")
            print(f"   Начальные потери: {initial_loss:.4f}")
            print(f"   Финальные потери: {final_loss:.4f}")
            print(f"   Улучшение: {improvement:.4f}")
            print(f"   Лучшие потери: {self.best_loss:.4f}")

    def generate_poetry(self, input_text, poetry_type=PoetryType.HIDDEN_HEAD, max_length=15):
        """
        Генерирует стихотворение на основе входного текста

        Args:
            input_text: входной текст (для藏头诗- строка символов)
            poetry_type: тип стихотворения
            max_length: максимальная длина предложения

        Returns:
            str: сгенерированное стихотворение

        Raises:
            ValueError: если входные данные некорректны
        """
        self.model.eval()

        if not input_text and poetry_type == PoetryType.HIDDEN_HEAD:
            raise ValueError("Для藏头诗необходимо указать входной текст")

        print(f"🎨 Генерация стиха...")
        print(f"   Вход: '{input_text}', Тип: {poetry_type.value}")

        try:
            input_tensor = self._prepare_input_tensor(input_text)

            with torch.no_grad():
                result = self.model.generate(
                    x=input_tensor,
                    vocab=self.vocab,
                    poetry_type=poetry_type,
                    sentence_count=len(input_text) if input_text else 4,
                    max_length=max_length
                )

            print(f"✅ Сгенерировано: {result}")
            return result

        except Exception as e:
            print(f"❌ Ошибка генерации: {e}")
            raise

    def _prepare_input_tensor(self, input_text):
        """
        Подготавливает входной тензор из текста

        Args:
            input_text: входной текст

        Returns:
            torch.Tensor: подготовленный тензор
        """
        if not input_text:
            random_idx = torch.randint(0, self.vocab_size, (1, 1))
            input_tensor = random_idx.to(self.device)
        else:

            try:
                char_indices = [self.vocab.stoi[char] for char in input_text]
            except KeyError as e:
                raise ValueError(f"Неизвестный символ в входном тексте: {e}")

            input_tensor = torch.tensor(char_indices).unsqueeze(0).to(self.device)

        input_one_hot = self.one_hot_embedding(input_tensor).float()
        return input_one_hot

    def interactive_generation(self):
        """
        Интерактивный режим генерации стихов
        """
        print("\n🎭 Интерактивная генерация стихов")
        print("   Команды:")
        print("   - Введите текст для藏头诗")
        print("   - Нажмите Enter для случайного стиха")
        print("   - Введите 'quit' для выхода")

        while True:
            try:
                user_input = input("\n📝 Введите текст: ").strip()

                if user_input.lower() == 'quit':
                    print("👋 До свидания!")
                    break

                if user_input == '':
                    # Случайная генерация
                    result = self.generate_poetry(
                        "",
                        poetry_type=PoetryType.BEGIN,
                        max_length=12
                    )
                else:
                    # Генерация藏头诗
                    result = self.generate_poetry(
                        user_input,
                        poetry_type=PoetryType.HIDDEN_HEAD,
                        max_length=15
                    )

            except KeyboardInterrupt:
                print("\n👋 До свидания!")
                break
            except Exception as e:
                print(f"❌ Ошибка: {e}")


def main():
    """
    Основная функция для обучения и тестирования модели
    """
    # Конфигурация обучения
    config = {
        'batch_size': 32,
        'learning_rate': 0.001,
        'hidden_size': 128,
        'epochs': 200,
        'dropout': 0.5,
        'model_path': 'model.pkl',
        'debug': False,  # Установите True для быстрой отладки
        'save_best_only': True,
        'early_stopping_patience': 10,
        'log_interval': 10,
        'gradient_clip': 1.0,
        'shuffle': True
    }

    print("=" * 50)
    print("🎭 Генератор китайской поэзии")
    print("=" * 50)

    try:
        # Инициализация тренера
        trainer = PoetryTrainer(config)

        # Обучение модели
        trainer.train()

        # Тестирование генерации
        test_cases = [
            ("花开有情", PoetryType.HIDDEN_HEAD),
            ("明月清风", PoetryType.HIDDEN_HEAD),
            ("", PoetryType.BEGIN)  # Случайная генерация
        ]

        print("\n🧪 Тестирование генерации:")
        print("-" * 30)

        for input_text, poetry_type in test_cases:
            try:
                result = trainer.generate_poetry(
                    input_text,
                    poetry_type,
                    max_length=15
                )
                print(f"✅ Успех: '{input_text}' → {result}")
            except Exception as e:
                print(f"❌ Ошибка для '{input_text}': {e}")

        # Запуск интерактивного режима
        trainer.interactive_generation()

    except Exception as e:
        print(f"💥 Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
