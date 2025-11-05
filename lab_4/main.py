#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Created on 2020/6/18 12:14
@author: phil
"""
from tqdm import tqdm

from dataloader import dataset2dataloader
from torch.optim import Adam
from models import PoetryModel
import torch
import torch.nn as nn
import numpy as np
import os


class PoetryTrainer:
    def __init__(self, config):
        """
        Инициализация тренера для модели генерации стихов

        Args:
            config (dict): словарь с конфигурационными параметрами
        """
        self.config = self._validate_config(config)
        self.device = self._setup_device()

        # Инициализация компонентов
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

        # Значения по умолчанию для опциональных параметров
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

        # Объединяем с пользовательской конфигурацией
        merged_config = {**default_config, **config}

        # Дополнительная валидация значений
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

if __name__ == "__main__":
    batch_size = 32
    learning_rate = 0.001
    hidden_size = 128
    epoch = 200

    train_iter, vocab = dataset2dataloader(batch_size=batch_size)

    vocab_size = len(vocab.stoi)
    # print(vocab_size, hidden_size, batch_size)
    model = PoetryModel(vocab_size=vocab_size, hidden_size=hidden_size, output_size=vocab_size)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    one_hot_embedding = nn.Embedding(vocab_size, vocab_size, _weight=torch.from_numpy(np.eye(vocab_size)))

    model_path = "model.pkl"
    if os.path.exists(model_path):
        model = torch.load(model_path)
    else:
        for ep in tqdm(range(epoch)):
            model.train()
            total_loss = 0
            for i, batch in enumerate(train_iter):
                optimizer.zero_grad()
                sent = batch.sent.t()
                x, y = sent[:, :-1], sent[:, 1:]
                x = one_hot_embedding(x).float()
                init_hidden = torch.zeros(1, len(x), hidden_size)
                output, _ = model(x, init_hidden)
                output = output.reshape(-1, output.shape[-1])
                y = y.flatten()
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if ep % (epoch // 10) == 0:
                print("loss: ", total_loss)
        torch.save(model, model_path)

    model.eval()
    # test = ["我好可爱"]  我病恨无我，。好一解颜色。可怜王经行自远，一解颜色。爱绿溪阴。
    # test = ["花开有情"]  花边行县柳，河桥晚泊船。开远树，山鸟助酣歌。有情何处，箫管凤初来。情何处所，风吹青珊瑚，可怜王孙立
    test = [""]
    for sent in test:
        sent = list(map(lambda x: vocab.stoi[x], list(sent)))
        x = torch.tensor(sent).unsqueeze(0)
        x = one_hot_embedding(x).float()
        with torch.no_grad():
            output = model.generate(x, stoi=vocab.stoi, poetry_type="hidden head")
    ans = torch.cat(output, dim=1).argmax(-1).squeeze(0)
    for word_id in ans:
        print(vocab.itos[word_id.item()], end="")