"""Parsing helpers shared by the ChemBench VLMEvalKit implementation."""
from __future__ import annotations

import json
import random
import re
from typing import Dict, List, Optional, Sequence, Tuple

from .constant import (
    ALPHABET,
    DEFAULT_REFUSAL_TOKENS,
    FLOAT_TAG_PATTERN,
    GENERIC_NUMBER_PATTERN,
    MCQ_TAG_PATTERN,
)


def normalize_target_scores(raw_scores) -> Dict[str, float]:
    if raw_scores is None:
        raise ValueError("ChemBench example is missing 'target_scores'.")
    if isinstance(raw_scores, str):
        raw_scores = json.loads(raw_scores)
    if not isinstance(raw_scores, dict):
        raise TypeError("target_scores must be a dictionary or JSON string")
    normalized = {}
    for key, value in raw_scores.items():
        normalized[str(key)] = float(value)
    return normalized


def enumerate_options(
    target_scores: Dict[str, float],
    shuffle: bool = False,
    seed: int = 42,
) -> List[Dict[str, object]]:
    items = list(target_scores.items())
    if shuffle:
        rnd = random.Random(seed)
        rnd.shuffle(items)
    options = []
    for idx, (text, score) in enumerate(items):
        if idx >= len(ALPHABET):
            raise ValueError("ChemBench only supports up to 26 answer options per example.")
        label = ALPHABET[idx]
        options.append({"label": label, "text": text.strip(), "score": float(score)})
    return options


def format_options_block(options: Sequence[Dict[str, object]]) -> str:
    return "\n".join(f"{opt['label']}. {opt['text']}" for opt in options)


def extract_tag_content(text: str, pattern: str) -> Optional[str]:
    match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    return match.group(1).strip()


def split_letters(span: str) -> List[str]:
    span = span.replace("and", ",")
    pieces = re.split(r"[,\s]+", span)
    letters: List[str] = []
    for piece in pieces:
        piece = piece.strip().upper()
        if not piece:
            continue
        if len(piece) > 1 and piece.startswith("OPTION "):
            piece = piece.split()[-1]
        if len(piece) == 1 and piece in ALPHABET:
            if piece not in letters:
                letters.append(piece)
    return letters


def parse_mcq_prediction(text: str, allowed_labels: Sequence[str]) -> List[str]:
    if not isinstance(text, str):
        return []
    span = extract_tag_content(text, MCQ_TAG_PATTERN)
    if span:
        letters = split_letters(span)
    else:
        letters = split_letters(text)
    allowed = [label.upper() for label in allowed_labels]
    return [letter for letter in letters if letter in allowed]


_NUMBER_REGEX = re.compile(GENERIC_NUMBER_PATTERN)


def _text2int(textnum: str) -> Optional[int]:
    units = {
        "zero": 0,
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "eleven": 11,
        "twelve": 12,
        "thirteen": 13,
        "fourteen": 14,
        "fifteen": 15,
        "sixteen": 16,
        "seventeen": 17,
        "eighteen": 18,
        "nineteen": 19,
    }
    tens = {
        "twenty": 20,
        "thirty": 30,
        "forty": 40,
        "fifty": 50,
        "sixty": 60,
        "seventy": 70,
        "eighty": 80,
        "ninety": 90,
    }
    scales = {"hundred": 100, "thousand": 1000, "million": 10**6, "billion": 10**9}

    current = 0
    result = 0
    tokens = re.split(r"[-\s]+", textnum.lower())
    valid = False
    for token in tokens:
        if token in units:
            current += units[token]
            valid = True
        elif token in tens:
            current += tens[token]
            valid = True
        elif token in scales:
            factor = scales[token]
            if current == 0:
                current = 1
            current *= factor
            if factor >= 1000:
                result += current
                current = 0
            valid = True
        elif token in {"and", "a"}:
            continue
        else:
            return None
    if not valid:
        return None
    return result + current


def parse_numeric_prediction(text: str) -> Tuple[Optional[float], Optional[str]]:
    if not isinstance(text, str):
        return None, None
    span = extract_tag_content(text, FLOAT_TAG_PATTERN)
    if span and (value := _safe_float(span)) is not None:
        return value, span.strip()
    match = _NUMBER_REGEX.search(text)
    if match is not None:
        raw = match.group(0)
        value = _safe_float(raw)
        if value is not None:
            return value, raw.strip()
    for candidate in re.findall(r"[a-zA-Z\-\s]+", text):
        candidate = candidate.strip()
        if not candidate:
            continue
        number_word = _text2int(candidate)
        if number_word is not None:
            return float(number_word), candidate
    return None, None


def _safe_float(value: str) -> Optional[float]:
    try:
        cleaned = value.replace(",", "").strip()
        return float(cleaned)
    except (TypeError, ValueError):
        return None


def looks_like_refusal(text: str) -> bool:
    if not isinstance(text, str):
        return False
    lowered = text.lower()
    return any(token.lower() in lowered for token in DEFAULT_REFUSAL_TOKENS)
