#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Модуль dataloader_byhand

Реализация функций и классов для чтения данных из TSV-файлов и
создания DataLoader'ов для классификации текстов с использованием PyTorch
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split


def prepare_data(dataset_path: str, sent_col_name: str, label_col_name: str)-> (
        tuple)[np.ndarray, np.ndarray]:
    """
    Читает предложения и метки из TSV-файла
    :param dataset_path: Путь к директории с файлом train.tsv
    :param sent_col_name: Имя колонки с текстом
    :param label_col_name: Имя колонки с метками
    :return: tuple[x, y]
    """
    file_path = os.path.join(dataset_path, "train.tsv")
    data = pd.read_csv(file_path, sep="\t")
    X = data[sent_col_name].values
    y = data[label_col_name].values
    return X, y


class Language:
    """
    Класс для создания словаря и преобразования текстов в числовые представления
    """

    def __init__(self):
        self.word2id = {}
        self.id2word = {}

    def fit(self, sent_list: list[str]) -> None:
        """
        Создает словарь на основе переданных предложений
        :param sent_list: Список строк
        :return: None
        """
        vocab = set()
        for sent in sent_list:
            vocab.update(sent.split(" "))
        word_list = ["<pad>", "<unk>"] + list(vocab)
        self.word2id = {word: i for i, word in enumerate(word_list)}
        self.id2word = {i: word for i, word in enumerate(word_list)}

    def transform(self, sent_list: list[str], reverse: bool = False) -> (
            list)[list[int] | list[str]]:
        """
        Преобразует список предложений в список индексов или наоборот
        :param sent_list: Список предложений или список индексов
        :param reverse: Если True, преобразует индексы в слова
        :return: Список списков индексов или слов
        """
        sent_list_id = []
        word_mapper = self.word2id if not reverse else self.id2word
        unk = self.word2id["<unk>"] if not reverse else None
        for sent in sent_list:
            sent_id = list(map(lambda x: word_mapper.get(x, unk),
                               sent.split(" ") if not reverse else sent))
            sent_list_id.append(sent_id)
        return sent_list_id


class ClsDataset(Dataset):
    """ Датасет для классификации текстов """
    def __init__(self, sents, labels):
        self.sents = sents
        self.labels = labels

    def __getitem__(self, item):
        return self.sents[item], self.labels[item]

    def __len__(self):
        return len(self.sents)


def collate_fn(batch_data: list[tuple[list[int], int]]) -> (
        tuple)[torch.LongTensor, torch.LongTensor, torch.FloatTensor]:
    """
    Функция для организации данных в батчах с дополнением до одинаковой длины
    :param batch_data: Cписок кортежей
    :return: tuple[padded_sents, labels, sents_len]
    """
    batch_data.sort(key=lambda data_pair: len(data_pair[0]), reverse=True)

    sents, labels = zip(*batch_data)
    sents_len = [len(sent) for sent in sents]
    sents = [torch.LongTensor(sent) for sent in sents]
    padded_sents = pad_sequence(sents, batch_first=True, padding_value=0)

    return (torch.LongTensor(padded_sents),
            torch.LongTensor(labels),  torch.FloatTensor(sents_len))


def get_wordvec(word2id: dict[str, int], vec_file_path:
str, vec_dim: int = 50) -> torch.Tensor:
    """
    Загружает предобученные векторные представления слов из текстового файла
    :param word2id: Словарь слово -> индекс
    :param vec_file_path: Путь к файлу с векторами
    :param vec_dim: Размерность векторов
    :return: Матрица эмбеддингов
    """
    print("Начало загрузки векторных представлений слов")
    word_vectors = torch.nn.init.xavier_uniform_(torch.empty(len(word2id), vec_dim))
    word_vectors[0, :] = 0  # <pad>
    found = 0
    with open(vec_file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            splited = line.split(" ")
            if splited[0] in word2id:
                found += 1
                word_vectors[word2id[splited[0]]] = torch.tensor(
                    list(map(lambda x: float(x), splited[1:])))
            if found == len(word2id) - 1:  # 允许<unk>找不到
                break
    print("Всего слов в словаре: %d, из них найдено векторных "
          "представлений: %d" % (len(word2id), found))
    return word_vectors.float()


def make_dataloader(
    dataset_path: str = "../dataset/kaggle-movie-review",
    sent_col_name: str = "Phrase",
    label_col_name: str = "Sentiment",
    batch_size: int = 32,
    vec_file_path: str = "./.vector_cache/glove.6B.50d.txt",
    debug: bool = False
) -> tuple[DataLoader, DataLoader, torch.Tensor, Language]:
    """
    Создает загрузчики данных для обучения и валидации модели классификации текстов
    :param dataset_path: Путь к директории с данными
    :param sent_col_name: Имя колонки с текстом
    :param label_col_name: Имя колонки с метками
    :param batch_size: Размер батча
    :param vec_file_path: Путь к предобученным векторам
    :param debug: Использовать только первые 100 примеров для отладки
    :return: tuple[cls_train_dataloader, cls_val_dataloader, word_vectors, X_language]
    """
    X, y = prepare_data(dataset_path=dataset_path,
                        sent_col_name=sent_col_name, label_col_name=label_col_name)

    if debug:
        X, y = X[:100], y[:100]

    X_language = Language()
    X_language.fit(X)
    X = X_language.transform(X)

    word_vectors = get_wordvec(X_language.word2id,
                               vec_file_path=vec_file_path, vec_dim=50)
    X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                      test_size=0.2, random_state=42)

    cls_train_dataset, cls_val_dataset = ClsDataset(X_train, y_train), ClsDataset(X_val, y_val)
    cls_train_dataloader = DataLoader(cls_train_dataset,
                                      batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    cls_val_dataloader = DataLoader(cls_val_dataset,
                                    batch_size=batch_size, collate_fn=collate_fn)

    return cls_train_dataloader, cls_val_dataloader, word_vectors, X_language


if __name__ == "__main__":
    (cls_train_dataloader, cls_val_dataloader, word_vectors,
     X_language) = make_dataloader(debug=True, batch_size=10)
    for batch in cls_train_dataloader:
        X, y, lens = batch
        print(X.shape, y.shape)
        break