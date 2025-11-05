import torch.nn as nn
import torch
from enum import Enum


class PoetryType(Enum):
    BEGIN = "begin"
    HIDDEN_HEAD = "hidden head"


class PoetryModel(nn.Module):
    BEGIN = "begin"
    HIDDEN_HEAD = "hidden head"

    def __init__(self, vocab_size, hidden_size, output_size, dropout=0.5):
        super(PoetryModel, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size=vocab_size, hidden_size=hidden_size, dropout=dropout, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x, init_hidden):
        # print(x.shape, init_hidden.shape)
        seq_out, hn = self.gru(x, init_hidden)
        output = self.out(seq_out)
        return output, hn

    def generate(self, x, vocab, poetry_type=PoetryType.BEGIN, sentence_count=4, max_length=15):
        """
        Генерация стихов

        Args:
            x: входные данные
            vocab: словарь
            poetry_type: тип стихотворения
            sentence_count: количество предложений
            max_length: максимальная длина предложения

        Returns:
            str: сгенерированное стихотворение
        """
        # 1. Валидация входных данных
        self._validate_input(x, poetry_type, sentence_count)

        # 2. Инициализация скрытого состояния
        hidden_state = self._get_initial_hidden_state(1)

        # 3. Генерация последовательности тензоров
        generated_sequence = self._generate_sequence(
            x, hidden_state, vocab, poetry_type, sentence_count, max_length
        )

        # 4. Конвертация в читаемый текст
        result_text = self._convert_to_text(generated_sequence, vocab)

        return result_text

    def _validate_input(self, x, poetry_type, sentence_count):
        """
        Проверяет корректность входных параметров

        Args:
            x: входной тензор
            poetry_type: тип стихотворения
            sentence_count: количество предложений
        """
        if poetry_type == PoetryType.HIDDEN_HEAD and x.shape[1] != sentence_count:
            raise ValueError(
                f"Для количество входных символов ({x.shape[1]}) "
                f"должно совпадать с количеством предложений ({sentence_count})"
            )

    def _generate_sequence(self, x, hidden_state, vocab, poetry_type, sentence_count, max_length):
        """
        Генерирует последовательность символов для всего стихотворения

        Args:
            x: входные данные
            hidden_state: начальное скрытое состояние
            vocab: словарь
            poetry_type: тип стихотворения
            sentence_count: количество предложений
            max_length: максимальная длина предложения

        Returns:
            list: список тензоров сгенерированных символов
        """
        generated_sequence = []
        current_hidden = hidden_state

        # Генерация каждого предложения
        for sentence_idx in range(sentence_count):
            # Обработка начала предложения (зависит от типа стихотворения)
            gru_input, current_hidden = self._process_sentence_start(
                x, sentence_idx, current_hidden, poetry_type
            )
            generated_sequence.append(gru_input)

            # Генерация остальной части предложения
            sentence_chars = self._generate_sentence(
                gru_input, current_hidden, vocab, max_length
            )
            generated_sequence.extend(sentence_chars)

        return generated_sequence

    def _process_sentence_start(self, x, sentence_idx, hidden_state, poetry_type):
        """
        Обрабатывает начало предложения в зависимости от типа стихотворения

        Args:
            x: входные данные
            sentence_idx: индекс текущего предложения
            hidden_state: текущее скрытое состояние
            poetry_type: тип стихотворения

        Returns:
            tuple: (вход для GRU, новое скрытое состояние)
        """
        if sentence_idx == 0 and poetry_type == PoetryType.BEGIN:
            # Для обычного стихотворения - обрабатываем весь вход сразу
            gru_output, new_hidden = self.gru(x, hidden_state)
            next_input = gru_output[:, -1, :].unsqueeze(1)
            return x, new_hidden

        elif poetry_type == PoetryType.HIDDEN_HEAD:
            # Для藏头诗- обрабатываем по одному символу из входа
            gru_output, new_hidden = self.gru(x[:, sentence_idx, :].unsqueeze(1), hidden_state)
            next_input = gru_output[:, -1, :].unsqueeze(1)
            return x[:, sentence_idx, :].unsqueeze(1), new_hidden

        else:
            # Для других типов стихотворений (можно расширить)
            return x, hidden_state

    def _generate_sentence(self, initial_input, hidden_state, vocab, max_length):
        """
        Генерирует одно предложение символов

        Args:
            initial_input: начальный вход для предложения
            hidden_state: начальное скрытое состояние
            vocab: словарь
            max_length: максимальная длина предложения

        Returns:
            list: список тензоров сгенерированных символов предложения
        """
        current_input = initial_input
        current_hidden = hidden_state
        generated_chars = []

        for position in range(max_length):
            # Получаем следующий символ
            next_char, current_hidden = self._get_next_char(current_input, current_hidden)
            generated_chars.append(next_char)

            # Проверяем, не конец ли предложения
            if self._is_end_of_sentence(next_char, vocab):
                break

            # Обновляем вход для следующей итерации
            current_input = next_char

        return generated_chars

    def _get_next_char(self, current_input, hidden_state):
        """
        Получает следующий символ на основе текущего состояния

        Args:
            current_input: текущий входной тензор
            hidden_state: текущее скрытое состояние

        Returns:
            tuple: (следующий символ как one-hot, новое скрытое состояние)
        """
        with torch.no_grad():
            # Прямой проход через GRU
            gru_output, new_hidden = self.gru(current_input, hidden_state)

            # Получаем распределение вероятностей
            output = self.output_layer(gru_output)

            # Выбираем символ с наибольшей вероятностью
            _, top_index = output.data.topk(1)
            top_index = top_index.item()

            # Создаем one-hot вектор для следующего входа
            next_input = torch.zeros(1, 1, self.vocab_size)
            next_input[0][0][top_index] = 1

            return next_input, new_hidden

    def _is_end_of_sentence(self, char_tensor, vocab):
        """
        Проверяет, является ли символ концом предложения

        Args:
            char_tensor: тензор символа
            vocab: словарь

        Returns:
            bool: True если символ обозначает конец предложения
        """
        char_index = char_tensor.argmax(-1).item()
        return char_index == vocab.stoi["。"]

    def _convert_to_text(self, sequence, vocab):
        """
        Конвертирует последовательность тензоров в читаемый текст

        Args:
            sequence: список тензоров
            vocab: словарь

        Returns:
            str: текст стихотворения
        """
        if not sequence:
            return ""

        # Объединяем все тензоры и получаем индексы символов
        combined_tensor = torch.cat(sequence, dim=1)
        char_indices = combined_tensor.argmax(-1).squeeze(0)

        # Конвертируем индексы в символы
        text_result = "".join(vocab.itos[idx.item()] for idx in char_indices)

        return text_result
