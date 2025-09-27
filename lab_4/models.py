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


        return alpha

    def forward(self, x, init_state, seq_encoder_output):
        # print(x.shape, init_state.shape, seq_encoder_output.shape)
        batch_size, max_len, _ = x.shape  # 独热码表示
        hi = init_state
        seq_decoder_output = []
        for i in range(max_len):
            # alpha shape (batch_size, hidden_size)
            alpha = self.get_alpha(hi, seq_encoder_output)  # alpha 表示当前time step的隐状态矩阵和encoder的各个time step输出的关联
            hi = torch.cat([alpha.unsqueeze(0), hi], dim=2)
            hi = self.atten_affine(hi)
            output, hi = self.gru(x[:, i, :].unsqueeze(1), hi)
            seq_output = self.hidden2label(output.squeeze(1))
            seq_decoder_output.append(seq_output.squeeze(1))
        seq_decoder_output = torch.stack(seq_decoder_output, dim=1)
        return seq_decoder_output, hi


class SimpleNMT(nn.Module):
    def __init__(self, in_vocab_size, out_vocab_size, in_hidden_size, out_hidden_size, output_size, with_attention=False):
        super(SimpleNMT, self).__init__()
        self.with_attention = with_attention
        self.encoder = EncoderRNN(in_vocab_size, in_hidden_size)

        if self.with_attention:
            self.decoder = DecoderAttenRNN(out_vocab_size, out_hidden_size, output_size)
        else:
            self.decoder = DecoderRNN(out_vocab_size, out_hidden_size, output_size)

    def forward(self, encoder_input, encoder_init_hidden, decoder_input=None, out_word2index=None, out_index2word=None,
                max_len=None, out_size=None):
        encoder_seq_output, encoder_last_state = self.encoder(encoder_input, encoder_init_hidden)
        # 训练时decoder每个time step输入标准答案
        if decoder_input is not None:
            if self.with_attention:
                logits, _ = self.decoder(decoder_input, encoder_last_state, encoder_seq_output)
            else:
                logits, _ = self.decoder(decoder_input, encoder_last_state)
            return logits
        else:
            # 测试时没有标准答案，一直解码直到出现<end>或者达到最大长度
            decoded_sents = []
            for i in range(len(encoder_input)):
                sent = []
                decoder_input = torch.FloatTensor(np.eye(out_size)[[out_word2index["<start>"]]]).unsqueeze(0)
                hi = encoder_last_state[:, i, :].unsqueeze(1)
                for di in range(max_len):
                    if self.with_attention:
                        # alpha = self.decoder.get_alpha(hi, encoder_seq_output[i, :, :].unsqueeze(
                        #     0))  # alpha 表示当前time step的隐状态矩阵和encoder的各个time step输出的关联
                        # hi = torch.cat([alpha.unsqueeze(0), hi], dim=2)
                        # hi = self.decoder.atten_affine(hi)
                        # # print(decoder_input.shape, hi.shape, encoder_seq_output.shape)
                        decoder_output, hdi = self.decoder(decoder_input, hi, encoder_seq_output[i, :, :].unsqueeze(0))
                    else:
                        decoder_output, hdi = self.decoder(decoder_input, hi)
                    topv, topi = decoder_output.data.topk(1)
                    topi = topi.item()
                    if topi == out_word2index["<end>"]:
                        break
                    else:
                        sent.append(out_index2word[topi])
                    decoder_input = torch.FloatTensor([np.eye(out_size)[topi]]).unsqueeze(0)
                    hi = hdi
                decoded_sents.append(sent)
            return decoded_sents

            return self.forward_inference(
                encoder_input, encoder_initial_hidden, out_word2index,
                out_index2word, max_len, out_size
            )