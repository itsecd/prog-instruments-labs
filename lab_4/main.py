#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Главный модуль запуска обучения моделей TextRNN и TextCNN.
Created on 2020/4/30 8:33
@author: phil
"""

import argparse
import time
from torch import optim
import torch
import numpy as np
from models import TextRNN, TextCNN
from dataloader_bytorchtext import dataset2dataloader
from dataloader_byhand import make_dataloader


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Text Classification with CNN/RNN")
    parser.add_argument("--model", choices=["cnn", "rnn", "lstm"], default="cnn",
                        help="Тип модели для обучения (cnn, rnn или lstm)")
    parser.add_argument("--epochs", type=int, default=500, help="Количество этапов обучения")
    parser.add_argument("--lr", type=float, default=0.001, help="Скорость обучения")
    parser.add_argument("--batch-size", type=int, default=100, help="Размер батча")
    parser.add_argument("--torchtext", action="store_true", help="Использовать torchtext для загрузки данных")
    args = parser.parse_args()

    learning_rate = args.lr
    epoch_num = args.epochs
    num_of_class = 5
    load_data_by_torchtext = args.torchtext

    if load_data_by_torchtext:
        train_iter, val_iter, word_vectors = dataset2dataloader(batch_size=args.batch_size, debug=True)
    else:
        train_iter, val_iter, word_vectors, X_lang = make_dataloader(batch_size=args.batch_size, debug=True)

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

    print(f"\nЗапуск обучения модели: {args.model.upper()} (epochs={epoch_num}, lr={learning_rate})\n")

    for epoch in range(epoch_num):
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
            loss = loss_fun(logits, y)
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

        print(f"Этап {epoch + 1:03d}/{epoch_num:03d} "
              f"| Потеря: {avg_loss:.4f} "
              f"| Точность на обучении: {train_acc:.2f} "
              f"| Точность на валидации: {val_acc:.2f} "
              f"| Время: {elapsed:.1f}s")

        if train_acc >= 0.99:
            print("Достигнута точность 99%, обучение остановлено.")
            break