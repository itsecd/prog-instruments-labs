#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Главный модуль запуска обучения моделей TextRNN и TextCNN
"""

import argparse
import time
from typing import Tuple, Optional
import numpy as np
import torch
from torch import optim
from models import TextRNN, TextCNN
from dataloader_bytorchtext import dataset2dataloader
from dataloader_byhand import make_dataloader


def save_model(model: torch.nn.Module, path: str = 'model.pth') -> None:
    """
    Сохраняет веса модели в файл

    :param model: Объект модели PyTorch
    :param path: Путь для сохранения модели
    """
    torch.save(model.state_dict(), path)


def load_model(model_class, path: str = 'model.pth', *args, **kwargs) -> torch.nn.Module:
    """
    Загружает веса модели из файла и возвращает объект модели.

    :param model_class: Класс модели (TextCNN или TextRNN)
    :param path: Путь к файлу с сохраненными весами
    :param args: Аргументы конструктора модели
    :param kwargs: Именованные аргументы конструктора модели
    :return: Объект модели с загруженными весами
    """
    model = model_class(*args, **kwargs)
    model.load_state_dict(torch.load(path))
    return model


def train_model(
    model: torch.nn.Module,
    train_iter,
    val_iter,
    optimizer: torch.optim.Optimizer,
    loss_fn,
    epochs: int,
    load_data_by_torchtext: bool = True
) -> torch.nn.Module:
    """
    Обучает модель на тренировочной выборке и оценивает на валидационной

    :param model: Объект модели PyTorch
    :param train_iter: Итератор обучающей выборки
    :param val_iter: Итератор валидационной выборки
    :param optimizer: Оптимизатор
    :param loss_fn: Функция потерь
    :param epochs: Количество эпох
    :param load_data_by_torchtext: Использовать ли torchtext для загрузки данных
    :return: Обученная модель
    """
    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0

        # Обучение
        for i, batch in enumerate(train_iter):
            if load_data_by_torchtext:
                x, y = batch.sent.t(), batch.label
            else:
                x, y, lens = batch
            logits = model(x)
            loss = loss_fn(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Оценка
        model.eval()
        train_accs, val_accs = [], []

        with torch.no_grad():
            # Train acc
            for batch in train_iter:
                if load_data_by_torchtext:
                    x, y = batch.sent.t(), batch.label
                else:
                    x, y, lens = batch
                logits = model(x)
                _, y_pre = torch.max(logits, -1)
                acc = torch.mean((torch.tensor(y_pre == y, dtype=torch.float)))
                train_accs.append(acc)

            # Val acc
            for batch in val_iter:
                if load_data_by_torchtext:
                    x, y = batch.sent.t(), batch.label
                else:
                    x, y, lens = batch
                logits = model(x)
                _, y_pre = torch.max(logits, -1)
                acc = torch.mean((torch.tensor(y_pre == y, dtype=torch.float)))
                val_accs.append(acc)

        train_acc = np.array(train_accs).mean()
        val_acc = np.array(val_accs).mean()
        avg_loss = running_loss / len(train_iter)
        elapsed = time.time() - epoch_start

        print(f"Этап {epoch + 1:03d}/{epochs:03d} "
              f"| Потеря: {avg_loss:.4f} "
              f"| Точность на обучении: {train_acc:.2f} "
              f"| Точность на валидации: {val_acc:.2f} "
              f"| Время: {elapsed:.1f}s")

        if train_acc >= 0.99:
            print("Достигнута точность 99%, обучение остановлено.")
            break

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Text Classification with CNN/RNN")
    parser.add_argument("--model", choices=["cnn", "rnn", "lstm"], default="cnn",
                        help="Тип модели для обучения (cnn, rnn или lstm)")
    parser.add_argument("--epochs",
                        type=int, default=500, help="Количество этапов обучения")
    parser.add_argument("--lr",
                        type=float, default=0.001, help="Скорость обучения")
    parser.add_argument("--batch-size",
                        type=int, default=100, help="Размер батча")
    parser.add_argument("--torchtext",
                        action="store_true", help="Использовать torchtext для загрузки данных")
    args = parser.parse_args()

    learning_rate = args.lr
    epoch_num = args.epochs
    num_of_class = 5
    load_data_by_torchtext = args.torchtext

    if load_data_by_torchtext:
        train_iter, val_iter, word_vectors = (
            dataset2dataloader(batch_size=args.batch_size, debug=True))
    else:
        train_iter, val_iter, word_vectors, X_lang = (
            make_dataloader(batch_size=args.batch_size, debug=True))

    if args.model == "rnn":
        model = TextRNN(
            vocab_size=len(word_vectors),
            embedding_dim=50,
            hidden_size=128,
            num_of_class=num_of_class,
            weights=word_vectors,
            rnn_type="RNN"
        )
    elif args.model == "lstm":
        model = TextRNN(
            vocab_size=len(word_vectors),
            embedding_dim=50,
            hidden_size=128,
            num_of_class=num_of_class,
            weights=word_vectors,
            rnn_type="LSTM"
        )
    else:  # CNN
        model = TextCNN(
            vocab_size=len(word_vectors),
            embedding_dim=50,
            num_of_class=num_of_class,
            embedding_vectors=word_vectors
        )

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fun = torch.nn.CrossEntropyLoss()

    model = train_model(model, train_iter, val_iter,
                        optimizer, loss_fun, epoch_num, load_data_by_torchtext)

    save_model(model, path=f"{args.model}_trained.pth")
    print(f"Модель {args.model.upper()} сохранена.")