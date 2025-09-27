#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Created on 2020/6/8 11:26
@author: phil
"""

# КОММИТ 3: Добавил импорты для type hints
from typing import List, Tuple, Optional
import os
import numpy as np
import torch
from faker import Faker
import random

from torch.nn import init
from tqdm import tqdm
from babel.dates import format_date
from torchtext import data
import pandas as pd
from sklearn.model_selection import train_test_split

fake = Faker()
Faker.seed(12345)
random.seed(12345)


class DateFormats:
    """Конфигурация форматов дат для генерации данных"""
    FORMATS = [
        'short',  # Краткий формат
        'medium',  # Средний формат
        'long',  # Длинный формат
        'full',  # Полный формат (повторяется для баланса)
        'full',
        'full',
        'd MMM YYY',  # День Сокр.Месяц Год
        'd MMMM YYY',  # День Полн.Месяц Год
        'dd MMM YYY',  # ДД Сокр.Месяц Год
        'd MMM, YYY',  # День Сокр.Месяц, Год
        'd MMMM, YYY',  # День Полн.Месяц, Год
        'dd, MMM YYY',  # ДД, Сокр.Месяц Год
        'd MM YY',  # День ММ ГГ
        'MMMM d YYY',  # Полн.Месяц День Год
        'MMMM d, YYY',  # Полн.Месяц День, Год
        'dd.MM.YY'  # ДД.ММ.ГГ
    ]

    LOCALES = ['en_US']


class DateDatasetGenerator:
    """Класс для генерации датасета дат"""

    @staticmethod
    def generate_single_date() -> Optional[Tuple[str, str]]:
        """
        Возвращает кортеж (человекочитаемая дата, машинночитаемая дата)
        """
        date_object = fake.date_object()

        try:
            human_readable = format_date(
                date_object,
                format=random.choice(DateFormats.FORMATS),
                locale='en_US'
            ).lower().replace(',', '')

            machine_readable = date_object.isoformat()
            return human_readable, machine_readable

        except AttributeError:
            return None

    @staticmethod
    def generate_dataset(num_examples: int) -> List[Tuple[str, str]]:
        dataset = []
        for _ in tqdm(range(num_examples), desc="Generating dates"):
            date_pair = DateDatasetGenerator.generate_single_date()
            if date_pair is not None:
                dataset.append(date_pair)
        return dataset


def prepare_data(dataset_path=r"../dataset/date-normalization", dataset_size=10, debug=False):
    if debug:
        dataset_size = 10
        train_file = os.path.join(dataset_path, "train_samll.csv")
        eval_file = os.path.join(dataset_path, "eval_samll.csv")
    else:
        train_file = os.path.join(dataset_path, "train.csv")
        eval_file = os.path.join(dataset_path, "eval.csv")
    if not os.path.exists(train_file) and not os.path.exists(train_file):
        dataset = load_dataset(dataset_size)
        source, target = zip(*dataset)
        X_train, X_test, y_train, y_test = train_test_split(source, target, random_state=42, test_size=0.2)
        train_df = pd.DataFrame()
        train_df["source"], train_df["target"] = X_train, y_train
        eval_df = pd.DataFrame()
        eval_df["source"], eval_df["target"] = X_test, y_test
        train_df.to_csv(train_file, index=False)
        eval_df.to_csv(eval_file, index=False)
    return train_file, eval_file


def dataset2dataloader(dataset_path, batch_size=10, dataset_size=10, debug=False):
    train_csv, dev_csv = prepare_data(dataset_path, dataset_size=dataset_size, debug=debug)

    def tokenizer(text):
        return list(text)

    # 这里只是定义了数据格式
    SOURCE = data.Field(sequential=True, tokenize=tokenizer, lower=False)
    # 目标输出前后需加入特殊的标志符
    TARGET = data.Field(sequential=True, tokenize=tokenizer, lower=False, init_token="<start>", eos_token="<end>")
    train, val = data.TabularDataset.splits(
        path='', train=train_csv, validation=dev_csv, format='csv', skip_header=True,
        fields=[('source', SOURCE), ('target', TARGET)])


    train_iter = data.BucketIterator(train, batch_size=batch_size, sort_key=lambda x: len(x.sent), shuffle=False)
    val_iter = data.BucketIterator(val, batch_size=batch_size, sort_key=lambda x: len(x.sent), shuffle=False)

    # 在 test_iter , sort一定要设置成 False, 要不然会被 torchtext 搞乱样本顺序
    # test_iter = data.Iterator(dataset=test, batch_size=128, train=False, sort=False, device=DEVICE)

    return train_iter, val_iter, SOURCE.vocab, TARGET.vocab