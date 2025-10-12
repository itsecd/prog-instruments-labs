#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Реализация модели TextCNN.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class TextCNN(nn.Module):

    def __init__(
            self,
            vocab_size: int,
            embedding_dim: int,
            num_of_class: int,
            embedding_vectors: Optional[torch.Tensor] = None,
            kernel_num: int = 100,
            kernel_size: List[int] = None,
            dropout: float = 0.5
    ):
        super(TextCNN, self).__init__()

        if kernel_size is None:
            kernel_size = [3, 4, 5]

        self.kernel_num = kernel_num
        self.kernel_size = kernel_size
        self.hidden_size = embedding_dim

        # Embedding layer
        if embedding_vectors is not None:
            self.embed = nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=embedding_dim,
                _weight=embedding_vectors
            )
        else:
            self.embed = nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=embedding_dim
            )

        # Convolutional layers
        self.convs = nn.ModuleList([
            nn.Conv2d(1, kernel_num, (K, embedding_dim))
            for K in kernel_size
        ])

        # Regularization and output
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(len(kernel_size) * kernel_num, num_of_class)

    def forward(self, x: torch.Tensor):

        # Преобразуем индексы слов в вектора и добавляем размер канала
        x_emb = self.embed(x).unsqueeze(1)  # (batch, 1, seq_len, emb_dim)

        # Применяем свертку + ReLU + max pooling для каждого фильтра
        conv_results = [
            F.max_pool1d(F.relu(conv(x_emb)).squeeze(3), kernel_size=F.relu(conv(x_emb)).squeeze(3).size(2)).squeeze(2)
            for conv in self.convs
        ]

        # Объединяем результаты всех фильтров
        features = torch.cat(conv_results, dim=1)
        features = self.dropout(features)

        # Классификация
        return self.classifier(features)

    def __repr__(self):
        return f"{self.__class__.__name__}(emb_dim={self.embedding_dim}, hidden={self.hidden_size}, num_classes={self.num_of_class})"


if __name__ == "__main__":
    model = TextCNN(vocab_size=100, embedding_dim=50, num_of_class=5)
    x = torch.randint(0, 100, (8, 20))
    print(model(x).shape)