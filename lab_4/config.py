"""
Конфигурация для обучения модели генерации стихов
"""

class Config:
    # Параметры данных
    BATCH_SIZE = 32
    DATASET_PATH = "../dataset/poetry"

    # Параметры модели
    HIDDEN_SIZE = 128
    DROPOUT = 0.5

    # Параметры обучения
    LEARNING_RATE = 0.001
    EPOCHS = 200

    # Пути
    MODEL_PATH = "model.pkl"

    # Отладка
    DEBUG = False

    # Генерация
    MAX_SENTENCE_LENGTH = 15
    SENTENCE_COUNT = 4