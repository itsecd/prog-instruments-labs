#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Created on 2020/6/10 11:18
@author: phil
"""
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict


class BaseEncoder(nn.Module):
    """Базовый класс для энкодеров с общей логикой"""

    def __init__(self, vocab_size: int, hidden_size: int, dropout: float = 0.5):
        super().__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(vocab_size, hidden_size, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor, initial_hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sequence_output, last_state = self.gru(x, initial_hidden)
        return sequence_output, last_state


class EncoderRNN(BaseEncoder):
    """Энкодер для последовательностей"""

    def __init__(self, vocab_size: int, hidden_size: int, dropout: float = 0.5):
        super().__init__(vocab_size, hidden_size, dropout)


class BaseDecoder(nn.Module):
    """Базовый класс для декодеров с общей логикой"""

    def __init__(self, vocab_size: int, hidden_size: int, output_size: int, dropout: float = 0.5):
        super().__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(vocab_size, hidden_size, dropout=dropout, batch_first=True)
        self.hidden_to_output = nn.Linear(hidden_size, output_size)


class DecoderRNN(BaseDecoder):
    """Простой декодер без механизма внимания"""

    def forward(self, x: torch.Tensor, initial_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sequence_output, last_state = self.gru(x, initial_state)
        output = self.hidden_to_output(sequence_output)
        return output, last_state


class AttentionDecoderRNN(BaseDecoder):
    """Декодер с механизмом внимания"""

    def __init__(self, vocab_size: int, hidden_size: int, output_size: int, dropout: float = 0.5):
        super().__init__(vocab_size, hidden_size, output_size, dropout)
        self.attention_combine = nn.Linear(hidden_size * 2, hidden_size)

    def compute_attention_weights(self, decoder_hidden: torch.Tensor, encoder_outputs: torch.Tensor) -> torch.Tensor:
        # decoder_hidden shape: (1, batch_size, hidden_size)
        # encoder_outputs shape: (batch_size, sequence_length, hidden_size)

        decoder_hidden_transposed = decoder_hidden.permute(1, 2, 0)  # (batch_size, hidden_size, 1)
        attention_scores = torch.bmm(encoder_outputs, decoder_hidden_transposed).squeeze(
            2)  # (batch_size, sequence_length)
        attention_weights = F.softmax(attention_scores, dim=1).unsqueeze(2)  # (batch_size, sequence_length, 1)

        # Вычисляем контекстный вектор
        context_vector = (attention_weights * encoder_outputs).sum(dim=1)  # (batch_size, hidden_size)
        return context_vector

    def forward(self,
                decoder_input: torch.Tensor,
                initial_state: torch.Tensor,
                encoder_outputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        batch_size, max_length, _ = decoder_input.shape

        current_hidden = initial_state
        decoder_outputs = []

        for time_step in range(max_length):  # Заменил i на time_step
            # Вычисляем веса внимания
            context_vector = self.compute_attention_weights(current_hidden, encoder_outputs)

            # Комбинируем контекст с текущим скрытым состоянием
            combined_input = torch.cat([context_vector.unsqueeze(0), current_hidden], dim=2)
            current_hidden = self.attention_combine(combined_input)

            # Обрабатываем один шаг декодера
            decoder_output, current_hidden = self.gru(
                decoder_input[:, time_step, :].unsqueeze(1),
                current_hidden
            )

            # Преобразуем скрытое состояние в выход
            output = self.hidden_to_output(decoder_output.squeeze(1))
            decoder_outputs.append(output)

        # Объединяем все выходы декодера
        all_outputs = torch.stack(decoder_outputs, dim=1)
        return all_outputs, current_hidden


class SimpleNMT(nn.Module):
    def __init__(self,
                 in_vocab_size: int,
                 out_vocab_size: int,
                 in_hidden_size: int,
                 out_hidden_size: int,
                 output_size: int,
                 with_attention: bool = False):
        super().__init__()
        self.with_attention = with_attention
        self.encoder = EncoderRNN(in_vocab_size, in_hidden_size)

        if self.with_attention:
            self.decoder = AttentionDecoderRNN(out_vocab_size, out_hidden_size, output_size)
        else:
            self.decoder = DecoderRNN(out_vocab_size, out_hidden_size, output_size)

    def forward_training(self,
                         encoder_input: torch.Tensor,
                         encoder_initial_hidden: torch.Tensor,
                         decoder_input: torch.Tensor) -> torch.Tensor:
        encoder_outputs, encoder_final_state = self.encoder(encoder_input, encoder_initial_hidden)

        if self.with_attention:
            logits, _ = self.decoder(decoder_input, encoder_final_state, encoder_outputs)
        else:
            logits, _ = self.decoder(decoder_input, encoder_final_state)

        return logits

    def forward_inference(self,
                          encoder_input: torch.Tensor,
                          encoder_initial_hidden: torch.Tensor,
                          word_to_index: Dict[str, int],
                          index_to_word: Dict[int, str],
                          max_length: int,
                          vocab_size: int) -> List[List[str]]:
        encoder_outputs, encoder_final_state = self.encoder(encoder_input, encoder_initial_hidden)
        batch_size = len(encoder_input)

        decoded_sentences = []

        for batch_index in range(batch_size):
            decoded_sentence = self._decode_single_sentence(
                encoder_outputs[batch_index],
                encoder_final_state[:, batch_index],
                word_to_index,
                index_to_word,
                max_length,
                vocab_size
            )
            decoded_sentences.append(decoded_sentence)

        return decoded_sentences

    def _decode_single_sentence(self,
                                encoder_output: torch.Tensor,
                                encoder_hidden: torch.Tensor,
                                word_to_index: Dict[str, int],
                                index_to_word: Dict[int, str],
                                max_length: int,
                                vocab_size: int) -> List[str]:
        decoded_tokens = []

        # Подготавливаем начальное состояние декодера
        start_token_index = word_to_index["<start>"]
        decoder_input = torch.FloatTensor(np.eye(vocab_size)[[start_token_index]]).unsqueeze(0)
        current_hidden = encoder_hidden.unsqueeze(1)
        encoder_output = encoder_output.unsqueeze(0)  # Добавляем dimension батча

        for time_step in range(max_length):
            if self.with_attention:
                decoder_output, current_hidden = self.decoder(
                    decoder_input, current_hidden, encoder_output
                )
            else:
                decoder_output, current_hidden = self.decoder(decoder_input, current_hidden)

            # Выбираем наиболее вероятный токен
            top_probability, top_index = decoder_output.data.topk(1)
            predicted_token_index = top_index.item()

            if predicted_token_index == word_to_index["<end>"]:
                break

            decoded_tokens.append(index_to_word[predicted_token_index])

            # Подготавливаем вход для следующего шага
            decoder_input = torch.FloatTensor([np.eye(vocab_size)[predicted_token_index]]).unsqueeze(0)

        return decoded_tokens

    def forward(self,
                encoder_input: torch.Tensor,
                encoder_initial_hidden: torch.Tensor,
                decoder_input: Optional[torch.Tensor] = None,
                out_word2index: Optional[Dict[str, int]] = None,
                out_index2word: Optional[Dict[int, str]] = None,
                max_len: Optional[int] = None,
                out_size: Optional[int] = None) -> torch.Tensor:

        if decoder_input is not None:
            # Режим обучения
            return self.forward_training(encoder_input, encoder_initial_hidden, decoder_input)
        else:
            # Режим инференса
            if None in [out_word2index, out_index2word, max_len, out_size]:
                raise ValueError("Все параметры для инференса должны быть указаны")

            return self.forward_inference(
                encoder_input, encoder_initial_hidden, out_word2index,
                out_index2word, max_len, out_size
            )