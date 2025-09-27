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


    def translate(model, sents):
        X = []
        for sent in sents:
            X.append(list(map(lambda x: source_vocab[x], list(sent))) + [source_vocab["<pad>"]] * (Tx - len(sent)))
        Xoh = torch.from_numpy(np.array(list(map(lambda x: to_categorical(x, num_classes=source_vocab_size), X))))
        encoder_init_hidden = torch.zeros(1, len(X), hidden_size)
        preds = model(Xoh, encoder_init_hidden, decoder_input=None, out_word2index=target_vocab.stoi,
                      out_index2word=target_vocab.itos, max_len=Ty, out_size=target_vocab_size)
        for gold, pred in zip(sents, preds):
            print(gold, "-->", "".join(pred))


    translate(model, sents)

    """ 不使用 attention
    dataset_size : 10000
    loss 940.5139790773392
    loss 151.68325132876635
    loss 17.91189043689519
    loss 8.461621267197188
    loss 0.4571912245155545
    loss 4.067497536438168
    loss 0.02432645454427984
    loss 0.022933890589229122
    loss 1.740354736426525
    loss 2.7019595313686295
    monday may 7 1983 --> 1983-05-07
    19 march 1998 --> 1998-03-19
    18 jul 2008 --> 2008-07-18
    9/10/70 --> 1970-09-10
    thursday january 1 1981 --> 1981-01-01
    thursday january 26 2015 --> 2015-01-26
    saturday april 18 1990 --> 1990-04-18
    sunday may 12 1988 --> 1988-05-12
    """

