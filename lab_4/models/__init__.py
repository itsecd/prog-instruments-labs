#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Модуль объединяющий модели TextRNN и TextCNN для удобного импорта
"""

from .text_rnn import TextRNN
from .text_cnn import TextCNN

__all__ = ["TextRNN", "TextCNN"]