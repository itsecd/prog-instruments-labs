#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Модуль для подготовки данных и создания загрузчиков данных (DataLoader) с использованием torchtext
"""

import os
import pandas as pd
import spacy
from sklearn.model_selection import train_test_split
from torch.nn import init
from torchtext import data
from typing import Tuple


def prepare_data(
    dataset_path: str,
    sent_col_name: str,
    label_col_name: str,
    debug: bool = False
) -> Tuple[str, str]:
    """
    Читает предложения и метки из TSV-файла, разделяет на обучающую и валидационную выборки
    и сохраняет их в CSV-файлы

    :param dataset_path: Путь к директории с TSV-файлом
    :param sent_col_name: Имя колонки с текстом
    :param label_col_name: Имя колонки с метками
    :param debug: Использовать только случайные 100 примеров для отладки
    :return: пути к CSV-файлам обучающей и валидационной выборки
    """
    file_path = os.path.join(dataset_path, "train.tsv")
    data = pd.read_csv(file_path, sep="\t")
    if debug:
        data = data.sample(n=100)
    X = data[sent_col_name].values
    y = data[label_col_name].values
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    train_df, val_df = pd.DataFrame(), pd.DataFrame()
    train_df["sent"], train_df["label"] = X_train, y_train
    val_df["sent"], val_df["label"] = X_val, y_val

    train_file_path = os.path.join(dataset_path, "train.csv")
    val_file_path = os.path.join(dataset_path, "val.csv")
    train_df.to_csv(train_file_path, index=False)
    val_df.to_csv(val_file_path, index=False)

    return train_file_path, val_file_path


def dataset2dataloader(
    dataset_path: str = "../dataset/kaggle-movie-review",
    sent_col_name: str = "Phrase",
    label_col_name: str = "Sentiment",
    batch_size: int = 32,
    vec_file_path: str = "./.vector_cache/glove.6B.50d.txt",
    debug: bool = False
) -> Tuple[data.Iterator, data.Iterator, data.Vocab.vectors]:
    """
    Создает загрузчики данных для обучения и валидации с использованием torchtext

    :param dataset_path: Путь к директории с данными
    :param sent_col_name: Имя колонки с текстом
    :param label_col_name: Имя колонки с метками
    :param batch_size: Размер батча
    :param vec_file_path: Путь к предобученным векторам GloVe
    :param debug: Использовать только первые 100 примеров для отладки
    :return: tuple[train_iter, val_iter, vocab_vectors]
    """
    train_file_name, val_file_name = prepare_data(dataset_path,
                                                  sent_col_name, label_col_name, debug=debug)
    spacy_en = spacy.load('en_core_web_sm')

    def tokenizer(text: str) -> list[str]:
        """
        Функция токенизации текста с использованием spaCy.

        :param text: Строка текста
        :return: Список токенов
        """
        return [tok.text for tok in spacy_en.tokenizer(text)]

    TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True)
    LABEL = data.Field(sequential=False, use_vocab=False)
    train, val = data.TabularDataset.splits(
        path='', train=train_file_name, validation=val_file_name,
        format='csv', skip_header=True,
        fields=[('sent', TEXT), ('label', LABEL)])

    TEXT.build_vocab(train, vectors='glove.6B.50d')
    TEXT.vocab.vectors.unk_init = init.xavier_uniform

    DEVICE = "cpu"
    train_iter = data.BucketIterator(train, batch_size=batch_size,
                                     sort_key=lambda x: len(x.sent),
                                     device=DEVICE)
    val_iter = data.BucketIterator(val, batch_size=batch_size,
                                   sort_key=lambda x: len(x.sent), shuffle=True,
                                   device=DEVICE)

    return train_iter, val_iter, TEXT.vocab.vectors


if __name__ == "__main__":
    train_iter, val_iter, vectors = dataset2dataloader(batch_size=32, debug=True)

    batch = next(iter(train_iter))
    print(batch.sent.shape)
    print(batch.label.shape)