#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Модуль объединяющий модели TextRNN и TextCNN для удобного импорта
"""

from models.text_rnn import TextRNN
from models.text_cnn import TextCNN

__all__ = ["TextRNN", "TextCNN"]
