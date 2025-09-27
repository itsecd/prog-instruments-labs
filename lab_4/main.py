#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Created on 2020/6/8 11:36
@author: phil
"""
from keras.utils import to_categorical

from dataloader import load_dataset, dataset2dataloader
from models import SimpleNMT
from torch import optim
import torch.nn as nn
import torch
import numpy as np
from pprint import pprint
from tqdm import tqdm


class TrainingConfig:
    EPOCHS = 500
    LEARNING_RATE = 0.001
    HIDDEN_SIZE = 64
    BATCH_SIZE = 10
    MAX_INPUT_LENGTH = 25  # Заменил Tx на понятное имя
    MAX_OUTPUT_LENGTH = 10  # Заменил Ty на понятное имя


def train_model():
    config = TrainingConfig()

    train_iter, val_iter, source_vocab, target_vocab = dataset2dataloader(
        dataset_path=r"../dataset/date-normalization",
        batch_size=config.BATCH_SIZE,
        dataset_size=10000,
        debug=True
    )

    source_vocab_size = len(source_vocab.stoi)
    target_vocab_size = len(target_vocab.stoi)

    model = SimpleNMT(
        in_vocab_size=source_vocab_size,
        out_vocab_size=target_vocab_size,
        in_hidden_size=config.HIDDEN_SIZE,
        out_hidden_size=config.HIDDEN_SIZE,
        output_size=target_vocab_size,
        with_attention=True
    )

    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    source_embedding = nn.Embedding(
        source_vocab_size, source_vocab_size,
        _weight=torch.from_numpy(np.eye(source_vocab_size))
    )
    target_embedding = nn.Embedding(
        target_vocab_size, target_vocab_size,
        _weight=torch.from_numpy(np.eye(target_vocab_size))
    )

    model.train()
    for epoch in range(config.EPOCHS):  # Заменил ep на epoch
        epoch_loss = 0
        for batch in train_iter:
            optimizer.zero_grad()

            encoder_input = batch.source.t().long()
            decoder_input = batch.target.t()[:, :-1].long()
            decoder_target = batch.target.t()[:, 1:]

            batch_size = len(encoder_input)
            initial_hidden = torch.zeros(1, batch_size, config.HIDDEN_SIZE)

            encoder_input_embedded = source_embedding(encoder_input).float()
            decoder_input_embedded = target_embedding(decoder_input).float()

            logits = model(encoder_input_embedded, initial_hidden, decoder_input_embedded)
            loss = criterion(logits.view(-1, logits.shape[-1]), decoder_target.flatten())
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

        if epoch % (config.EPOCHS // 10) == 0:
            print(f"Epoch {epoch}, Loss: {epoch_loss}")

    return model, source_vocab, target_vocab, config


def translate_dates(model, sentences, source_vocab, target_vocab, config):
    """
    КОММИТ 1: Вынес логику перевода в отдельную функцию с понятным именем
    и добавил документацию
    """
    encoded_sentences = []
    for sentence in sentences:
        # КОММИТ 1: Улучшил читаемость создания encoded последовательности
        tokens = [source_vocab[char] for char in sentence]
        padding = [source_vocab["<pad>"]] * (config.MAX_INPUT_LENGTH - len(tokens))
        encoded_sentences.append(tokens + padding)

    one_hot_encoded = []
    for encoded_sentence in encoded_sentences:
        one_hot = to_categorical(encoded_sentence, num_classes=len(source_vocab.stoi))
        one_hot_encoded.append(one_hot)

    X_one_hot = torch.from_numpy(np.array(one_hot_encoded))
    encoder_initial_hidden = torch.zeros(1, len(sentences), config.HIDDEN_SIZE)

    predictions = model(
        X_one_hot,
        encoder_initial_hidden,
        decoder_input=None,
        out_word2index=target_vocab.stoi,
        out_index2word=target_vocab.itos,
        max_len=config.MAX_OUTPUT_LENGTH,
        out_size=len(target_vocab.stoi)
    )

    for original, predicted in zip(sentences, predictions):
        print(f"{original} --> {''.join(predicted)}")


if __name__ == "__main__":
    # Тестовые примеры
    test_sentences = [
        "monday may 7 1983",
        "19 march 1998",
        "18 jul 2008",
        "9/10/70",
        "thursday january 1 1981",
        "thursday january 26 2015",
        "saturday april 18 1990",
        "sunday may 12 1988"
    ]

    trained_model, source_vocab, target_vocab, config = train_model()
    translate_dates(trained_model, test_sentences, source_vocab, target_vocab, config)