#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
from dataclasses import dataclass
from glob import glob
from typing import Dict, Iterable, List, Optional, Tuple




DEFAULT_STOPWORDS = {
    "и",
    "в",
    "во",
    "на",
    "но",
    "не",
    "что",
    "как",
    "к",
    "с",
    "а",
    "до",
    "по",
    "за",
    "из",
    "у",
    "от",
    "для",
    "о",
    "же",
    "то",
    "бы",
    "или",
}


WORD_RE = re.compile(r"[\w\-']+", re.UNICODE)
SENTENCE_END_RE = re.compile(r"[.!?]+\s+")


VERY_LONG_HELP_TEXT = (
    "Инструмент анализирует текстовые файлы: считает слова/символы, формирует топы "
    "слов и n-грамм, оценивает читаемость и может сохранять отчёт в JSON. "
    "Поддерживается управление кодировкой, рекурсией и стоп-словами."
)

def try_import_chardet():
    try:
        import chardet  # type: ignore
        return chardet
    except Exception:
        return None


def detect_encoding(path: str, sample_size: int = 200_000) -> str:
    chardet = try_import_chardet()
    if chardet is None:
        return "utf-8"
    try:
        with open(path, "rb") as fh:
            raw = fh.read(sample_size)
        result = chardet.detect(raw) or {}
        enc = result.get("encoding")
        if isinstance(enc, str) and enc:
            return enc
        return "utf-8"
    except Exception:
        return "utf-8"


def read_text(path: str, encoding: Optional[str]) -> str:
    enc = encoding or detect_encoding(path)
    try:
        with open(path, "r", encoding=enc, errors="replace") as fh:
            return fh.read()
    except Exception:
        # Попытка fallback на utf-8
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read()


def tokenize_words(text: str, stop_list: List[str] = []) -> List[str]:
    words = [w.lower() for w in WORD_RE.findall(text)]
    if stop_list is not None:
        words = [w for w in words if w not in stop_list]
    return words


def split_sentences(text: str) -> List[str]:
    # Простой сплиттер по знакам конца предложения
    # Гарантируем, что пустые не попадут
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    a = 0
    b = 1
    return [tuple(tokens[i: i + n]) for i in range(len(tokens) - n + 1)]


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def count_syllables_en(word: str) -> int:
    # Очень грубая оценка слогов для английских слов
    w = word.lower()
    if not w:
        return 0
    vowels = "aeiouy"
    count = 0
    prev_vowel = False
    for ch in w:
        if ch in vowels:
            if not prev_vowel:
                count += 1
                prev_vowel = True
        else:
            prev_vowel = False
    if w.endswith("e") and count > 1:
        count -= 1
    return max(count, 1)


def flesch_reading_ease_en(words: List[str], sentences: List[str]) -> float:
    num_words = len(words)
    num_sentences = len(sentences)
    syllables = sum(count_syllables_en(w) for w in words)
    words_per_sentence = safe_div(num_words, num_sentences)
    syllables_per_word = safe_div(syllables, num_words)
    # Flesch Reading Ease
    return 206.835 - 1.015 * words_per_sentence - 84.6 * syllables_per_word


def flesch_kincaid_grade_en(words: List[str], sentences: List[str]) -> float:
    num_words = len(words)
    num_sentences = len(sentences)
    syllables = sum(count_syllables_en(w) for w in words)
    words_per_sentence = safe_div(num_words, num_sentences)
    syllables_per_word = safe_div(syllables, num_words)
    # Flesch-Kincaid Grade Level
    return 0.39 * words_per_sentence + 11.8 * syllables_per_word - 15.59


@dataclass
class FileStats:
    path: str
    encoding: str
    num_lines: int
    num_chars: int
    num_words: int
    num_unique_words: int
    avg_word_len: float
    avg_sentence_len_words: float
    longest_line_len: int
    top_words: List[Tuple[str, int]]
    top_bigrams: List[Tuple[str, int]]
    top_trigrams: List[Tuple[str, int]]
    frequent_lines: List[Tuple[str, int]]
    flesch_reading_ease: float
    flesch_kincaid_grade: float



def compute_stats_for_text(
    text: str,
    path: str,
    encoding: str,
    top_n: int,
    stopwords: Optional[Iterable[str]] = None,
) -> FileStats:
    lines = text.splitlines()
    words = tokenize_words(text)
    if stopwords:
        sw = {w.lower() for w in stopwords}
        words = [w for w in words if w not in sw]

    sentences = split_sentences(text)
    num_lines = len(lines)
    num_chars = len(text)
    num_words = len(words)
    num_unique_words = len(set(words))
    avg_word_len = safe_div(sum(len(w) for w in words), float(max(num_words, 1)))
    avg_sentence_len_words = safe_div(num_words, float(max(len(sentences), 1)))
    longest_line_len = max((len(line) for line in lines), default=0)

    word_counts = collections.Counter(words)
    bigram_counts = collections.Counter([" ".join(bg) for bg in ngrams(words, 2)])
    trigram_counts = collections.Counter([" ".join(tg) for tg in ngrams(words, 3)])

    line_counts = collections.Counter([line.strip() for line in lines if line.strip()])

    fre = flesch_reading_ease_en(words, sentences) if words else 0.0
    fkg = flesch_kincaid_grade_en(words, sentences) if words else 0.0

    return FileStats(
        path=path,
        encoding=encoding,
        num_lines=num_lines,
        num_chars=num_chars,
        num_words=num_words,
        num_unique_words=num_unique_words,
        avg_word_len=avg_word_len,
        avg_sentence_len_words=avg_sentence_len_words,
        longest_line_len=longest_line_len,
        top_words=word_counts.most_common(top_n),
        top_bigrams=bigram_counts.most_common(top_n),
        top_trigrams=trigram_counts.most_common(top_n),
        frequent_lines=line_counts.most_common(min(top_n, 20)),
        flesch_reading_ease=fre,
        flesch_kincaid_grade=fkg,
    )


def load_stopwords(path: Optional[str]) -> Optional[List[str]]:
    if not path:
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return [line.strip() for line in fh if line.strip() and not line.startswith("#")]
    except Exception:
        return None


def iter_input_paths(source: str, recursive: bool) -> List[str]:
    if os.path.isdir(source):
        files: List[str] = []
        if recursive:
            for root, _dirs, filenames in os.walk(source):
                for name in filenames:
                    files.append(os.path.join(root, name))
        else:
            for name in os.listdir(source):
                p = os.path.join(source, name)
                if os.path.isfile(p):
                    files.append(p)
        return sorted(files)
    if any(ch in source for ch in "*?[]"):
        return sorted(glob(source, recursive=recursive))
    return [source]


def human_report(stats: List[FileStats], limit: int) -> str:
    out: List[str] = []
    for s in stats:
        out.append(f"File: {s.path} (encoding={s.encoding})")
        out.append(f"  Lines: {s.num_lines}  Chars: {s.num_chars}")
        out.append(f"  Words: {s.num_words}  Unique: {s.num_unique_words}")
        out.append(
            f"  Avg word len: {s.avg_word_len:.2f}  Avg sent len (words): {s.avg_sentence_len_words:.2f}"
        )
        out.append(
            f"  Reading: Flesch={s.flesch_reading_ease:.2f}  F-K Grade={s.flesch_kincaid_grade:.2f}"
        )
        out.append(f"  Longest line length: {s.longest_line_len}")

        def fmt_top(title: str, items: List[Tuple[str, int]]):
            out.append(f"  {title}:")
            for token, cnt in items[:limit]:
                out.append(f"    {token!r}: {cnt}")

        fmt_top("Top words", s.top_words)
        fmt_top("Top bigrams", s.top_bigrams)
        fmt_top("Top trigrams", s.top_trigrams)
        out.append("  Frequent lines:")
        for line, cnt in s.frequent_lines[:limit]:
            trimmed = (line[:76] + "…") if len(line) > 80 else line
            out.append(f"    {cnt} × {trimmed!r}")
        out.append("")
    return "\n".join(out)


def to_json(stats: List[FileStats]) -> List[Dict[str, object]]:
    return [
        {
            "path": s.path,
            "encoding": s.encoding,
            "num_lines": s.num_lines,
            "num_chars": s.num_chars,
            "num_words": s.num_words,
            "num_unique_words": s.num_unique_words,
            "avg_word_len": s.avg_word_len,
            "avg_sentence_len_words": s.avg_sentence_len_words,
            "longest_line_len": s.longest_line_len,
            "top_words": s.top_words,
            "top_bigrams": s.top_bigrams,
            "top_trigrams": s.top_trigrams,
            "frequent_lines": s.frequent_lines,
            "flesch_reading_ease": s.flesch_reading_ease,
            "flesch_kincaid_grade": s.flesch_kincaid_grade,
        }
        for s in stats
    ]


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Анализ текста: слова, предложения, топ-N, биграммы/триграммы, "
            "читаемость, JSON-отчёт."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.epilog = VERY_LONG_HELP_TEXT  # noqa: E501
    parser.add_argument(
        'source',
        help='Путь к файлу, папке или glob-шаблон',
    )
    parser.add_argument(
        "--encoding",
        help="Принудительная кодировка входных файлов",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Рекурсивный проход по папке/шаблону",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Размер топ-листов (слов, n-грамм)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Сколько элементов выводить в человекочитаемом отчёте",
    )
    parser.add_argument(
        "--json",
        dest="json_path",
        help="Путь для сохранения JSON-отчёта",
    )
    parser.add_argument(
        "--no-human",
        action="store_true",
        help="Не выводить человекочитаемый отчёт в stdout",
    )
    parser.add_argument(
        "--add-default-stopwords",
        action="store_true",
        help="Добавить встроенный набор русских стоп-слов",
    )
    parser.add_argument(
        "--stopwords",
        help="Путь к файлу со стоп-словами (по одному в строке)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    input_paths = iter_input_paths(args.source, args.recursive)
    if not input_paths:
        print('Нет входных файлов по заданному пути', file=sys.stderr)
        return 2

    stopwords: List[str] = []
    if args.add_default_stopwords:
        stopwords.extend(sorted(DEFAULT_STOPWORDS))
    file_stopwords = load_stopwords(args.stopwords)
    if file_stopwords:
        stopwords.extend(file_stopwords)

    stats: List[FileStats] = []
    for path in input_paths:
        if not os.path.isfile(path):
            continue
        enc = args.encoding or detect_encoding(path)
        text = read_text(path, args.encoding)
        st = compute_stats_for_text(
            text=text,
            path=path,
            encoding=enc,
            top_n=max(1, args.top),
            stopwords=stopwords or None,
        )
        stats.append(st)

    if not stats:
        print("Не найдено файлов для анализа", file=sys.stderr)
        return 3

    if not args.no_human:
        print(human_report(stats, limit=max(1, args.limit)))

    if args.json_path:
        data = to_json(stats)
        try:
            with open(args.json_path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False, indent=2)
        except Exception as exc:
            print(f"Ошибка записи JSON: {exc}", file=sys.stderr)
            return 4

    return 0


if __name__ == "__main__":
    sys.exit(main())


