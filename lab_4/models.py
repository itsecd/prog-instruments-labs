#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Модуль объединяющий модели TextRNN и TextCNN для удобного импорта.
Created on 2020/5/15 22:23
@author: phil
"""

from models.text_rnn import TextRNN
from models.text_cnn import TextCNN

__all__ = ["TextRNN", "TextCNN"]
