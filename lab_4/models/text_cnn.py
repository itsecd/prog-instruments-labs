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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Embedding layer
        embed_out = self.embed(x)  # (batch_size, seq_len, embedding_dim)
        embed_out = embed_out.unsqueeze(1)  # (batch_size, 1, seq_len, embedding_dim)

        # Convolution and pooling
        conv_outputs = []
        for conv in self.convs:
            # Convolution
            conv_out = F.relu(conv(embed_out))  # (batch_size, kernel_num, seq_len-K+1, 1)
            conv_out = conv_out.squeeze(3)  # (batch_size, kernel_num, seq_len-K+1)

            # Max pooling over time
            pool_out = F.max_pool1d(conv_out, conv_out.size(2))  # (batch_size, kernel_num, 1)
            pool_out = pool_out.squeeze(2)  # (batch_size, kernel_num)

            conv_outputs.append(pool_out)

        # Concatenate and classify
        concatenated = torch.cat(conv_outputs, 1)  # (batch_size, len(kerner_size)*kernel_num)
        concatenated = self.dropout(concatenated)
        logits = self.classifier(concatenated)

        return logits


    def __repr__(self):
        return f"{self.__class__.__name__}(emb_dim={self.embedding_dim}, hidden={self.hidden_size}, num_classes={self.num_of_class})"


if __name__ == "__main__":
    import torch
    model = TextCNN(vocab_size=100, embedding_dim=50, num_of_class=5)
    x = torch.randint(0, 100, (8, 20))
    print(model(x).shape)