#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Реализация модели TextRNN
"""

import torch
import torch.nn as nn
from typing import Optional


class TextRNN(nn.Module):
    """
    Класс TextRNN для классификации текстов
    """

    def __init__(
            self,
            vocab_size: int,
            embedding_dim: int,
            hidden_size: int,
            num_of_class: int,
            weights: Optional[torch.Tensor] = None,
            rnn_type: str = "LSTM",
            bidirectional: bool = True,
            num_layers: int = 1,
            dropout: float = 0.5
    ):
        super(TextRNN, self).__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_of_class = num_of_class
        self.embedding_dim = embedding_dim
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1

        # Встраиваемый слой
        if weights is not None:
            self.embed = nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=embedding_dim,
                _weight=weights
            )
        else:
            self.embed = nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=embedding_dim
            )

        dropout_val = dropout if dropout > 0 and num_layers > 1 else 0

        # RNN слои
        if rnn_type == "RNN":
            self.rnn = nn.RNN(
                input_size=embedding_dim,
                hidden_size=hidden_size,
                batch_first=True,
                bidirectional=bidirectional,
                num_layers=num_layers,
                dropout=dropout_val
            )
        elif rnn_type == "LSTM":
            self.rnn = nn.LSTM(
                input_size=embedding_dim,
                hidden_size=hidden_size,
                batch_first=True,
                bidirectional=bidirectional,
                num_layers=num_layers,
                dropout=dropout_val
            )
        else:
            raise ValueError(f"Неподдерживаемый тип RNN: {rnn_type}")

        # Классификатор
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(
            hidden_size * self.num_directions,
            num_of_class
        )

    def forward(self, input_sents: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход модели
        :param input_sents: Тензор индексов слов с размерностью (batch_size, seq_len)
        :return: Логиты для каждого класса
        """
        batch_size, seq_len = input_sents.shape

        # Встраиваемый слой
        embed_out = self.embed(input_sents)  # (batch_size, seq_len, embedding_dim)

        # RNN слой
        if self.rnn_type == "RNN":
            h0 = torch.zeros(
                self.num_layers * self.num_directions,
                batch_size,
                self.hidden_size
            )
            _, hn = self.rnn(embed_out, h0)
        elif self.rnn_type == "LSTM":
            h0 = torch.zeros(
                self.num_layers * self.num_directions,
                batch_size,
                self.hidden_size
            )
            c0 = torch.zeros(
                self.num_layers * self.num_directions,
                batch_size,
                self.hidden_size
            )
            _, (hn, _) = self.rnn(embed_out, (h0, c0))

        # Обработчик двунаправленного вывода
        if self.bidirectional:
            hn = torch.cat([hn[-2], hn[-1]], dim=-1)
        else:
            hn = hn[-1]

        # Классификатор
        hn = self.dropout(hn)
        logits = self.classifier(hn)
        return logits

    def __repr__(self):
        return (f"{self.__class__.__name__}(emb_dim={self.embedding_dim}, "
                f"hidden={self.hidden_size}, num_classes={self.num_of_class})")


if __name__ == "__main__":
    model = TextRNN(vocab_size=200, embedding_dim=50,
                    hidden_size=128, num_of_class=5)
    x = torch.randint(0, 200, (8, 30))
    print(model(x).shape)
