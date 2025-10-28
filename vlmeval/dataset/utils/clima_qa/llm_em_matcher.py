# -*- coding: utf-8 -*-
"""
LLM-assisted Exact Match (EM) checker.

Workflow:
1) Do a lightweight canonicalization (unwrap boxes/JSON, strip common prefixes,
   normalize unicode/whitespaces/outer quotes).
2) If still not equal, ask an LLM judge (e.g., gpt-4o) to decide if they match.
   The LLM prompt is in ENGLISH and constrained to return ONLY "MATCH" or "NO MATCH".

Usage in your evaluate():
    from llm_em_matcher import em_with_llm
    from .utils.judge_util import build_judge

    judge_model_name = judge_kwargs.pop('model', 'gpt-4o-1120')
    judge = build_judge(model=judge_model_name, **judge_kwargs)

    # Case 1: prediction may be a list -> em_with_llm handles that
    rec["metrics"]["EM"] = em_with_llm(
        answer=rec["answer"],
        prediction=rec["prediction"],
        task=rec.get("task", ""),
        model=judge,
        enable_llm_on_mismatch=True
    )

Notes:
- task in {"mcq", "cloze", "ffq"}:
  * mcq   : compare only the option letter (a/b/c/d).
  * cloze : allow same term/unit (e.g., "hectopascal" vs "hPa") after unwrapping; otherwise exact.
  * ffq   : strict semantic equivalence (not just similar).
- If model is None or enable_llm_on_mismatch=False, only local EM is used.
"""

import re
import json
import unicodedata
from typing import Any, Dict, Optional, Tuple

# --------- Regex patterns: unwrapping & prefix trimming ---------

_BOX_PATTERNS = [
    re.compile(r"<\|begin_of_box\|>(.*?)<\|end_of_box\|>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE),
    re.compile(r"```(?:\w+)?\s*(.*?)\s*```", re.DOTALL),  # Markdown code block
]

_PREFIX_PATTERNS = [
    re.compile(r"^\s*(final\s+answer\s*[:：-]\s*)", re.IGNORECASE),
    re.compile(r"^\s*(answer\s*[:：-]\s*)", re.IGNORECASE),
    re.compile(r"^\s*(prediction\s*[:：-]\s*)", re.IGNORECASE),
    re.compile(r"^\s*(response\s*[:：-]\s*)", re.IGNORECASE),
]

# MCQ option extraction (a/b/c/d)
_MCQ_LETTER = re.compile(r"^\s*[\(\[\{]?([A-Da-d])[\)\]\}.\s：:,-]*")


def _to_str(x: Any) -> str:
    """Coerce input (possibly list/tuple) to str; pick first element if sequence."""
    if x is None:
        return ""
    if isinstance(x, (list, tuple)) and len(x) > 0:
        x = x[0]
    return str(x)


def _nfkc_and_spaces(s: str) -> str:
    """NFKC normalization + collapse multiple whitespaces."""
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _unwrap_boxes(s: str) -> str:
    """Unwrap JSON shells and common box/code/XML wrappers."""
    t = s
    # Try JSON shells: {"answer":"..."}, {"final_answer":"..."}, {"prediction":"..."}
    try:
        obj = json.loads(t)
        if isinstance(obj, dict):
            for k in ("answer", "final_answer", "prediction", "result", "output"):
                if k in obj and isinstance(obj[k], str):
                    t = obj[k]
                    break
    except Exception:
        pass
    # Generic wrappers
    for pat in _BOX_PATTERNS:
        m = pat.search(t)
        if m:
            t = m.group(1)
    return t.strip()


def _strip_prefixes(s: str) -> str:
    """Remove common leading labels such as 'Answer:'."""
    t = s
    for pat in _PREFIX_PATTERNS:
        t = pat.sub("", t)
    return t.strip()


def _strip_wrapping_quotes(s: str) -> str:
    """Remove one layer of surrounding quotes/brackets."""
    pairs = [("“","”"),("‘","’"),('"','"'),("'", "'"),("（","）"),("(",")"),("[","]"),("{","}")]
    t = s.strip()
    for l, r in pairs:
        if t.startswith(l) and t.endswith(r) and len(t) >= 2:
            return t[1:-1].strip()
    return t


def _normalize_text_for_em(s: str) -> str:
    """
    Lightweight canonicalization for EM:
    - NFKC
    - collapse spaces
    - remove one layer of outer quotes/brackets
    NOTE: keep case and inner punctuation as-is (closer to paper's strictness).
    """
    t = _nfkc_and_spaces(s)
    t = _strip_wrapping_quotes(t)
    return t


def _normalize_mcq_choice(s: str) -> str:
    """Extract leading option letter a/b/c/d; fallback to normalized text if not found."""
    t = _normalize_text_for_em(_strip_prefixes(_unwrap_boxes(s)))
    m = _MCQ_LETTER.match(t)
    if m:
        return m.group(1).lower()
    m2 = re.search(r"\boption\s*([A-Da-d])\b", t, re.IGNORECASE)
    if m2:
        return m2.group(1).lower()
    if len(t) == 1 and t.lower() in {"a", "b", "c", "d"}:
        return t.lower()
    return t


# --------- Local EM first; if mismatch, ask LLM ---------

def _local_em(gold: str, pred: str, task: str) -> Tuple[int, str, str]:
    """
    Local canonical EM:
      - mcq   : compare only the option letter (a/b/c/d)
      - cloze : unwrap/prefix/normalize (keep case), then exact
      - ffq   : same as cloze (EM is rarely used, but supported)
    Returns (em, gold_norm, pred_norm).
    """
    t = (task or "").strip().lower()

    if t == "mcq":
        g = _normalize_mcq_choice(gold)
        p = _normalize_mcq_choice(pred)
        return (1 if g == p else 0), g, p

    # cloze / ffq / others
    g = _normalize_text_for_em(_strip_prefixes(_unwrap_boxes(gold)))
    p = _normalize_text_for_em(_strip_prefixes(_unwrap_boxes(pred)))
    return (1 if g == p else 0), g, p


def _build_llm_prompt(task: str, gold_norm: str, pred_norm: str) -> Dict[str, str]:
    """
    Build the ENGLISH-only instruction.
    The model MUST return ONLY one token: 'MATCH' or 'NO MATCH'.
    """
    t = (task or "").strip().lower()

    if t == "mcq":
        rules = (
            "You are verifying whether two multiple-choice answers are the same.\n"
            "RULES:\n"
            "• Compare ONLY the option letter (a/b/c/d). Ignore all other text.\n"
            "• Case-insensitive for the option letter is acceptable.\n"
            "OUTPUT:\n"
            "Return exactly one of: MATCH or NO MATCH.\n"
        )
    elif t == "cloze":
        rules = (
            "You are verifying whether two cloze answers are the same.\n"
            "RULES:\n"
            "• Treat common wrappers/quotes as irrelevant.\n"
            "• Consider standard synonyms/abbreviations of the same scientific term/unit as the SAME "
            "(e.g., 'hectopascal' == 'hPa').\n"
            "• If they denote the same term/unit/value, return MATCH; otherwise return NO MATCH.\n"
            "OUTPUT:\n"
            "Return exactly one of: MATCH or NO MATCH.\n"
        )
    else:  # ffq or others
        rules = (
            "You are verifying strict semantic equivalence between two answers.\n"
            "RULES:\n"
            "• They must convey the same meaning without missing or extra information.\n"
            "• Mere similarity is NOT enough.\n"
            "OUTPUT:\n"
            "Return exactly one of: MATCH or NO MATCH.\n"
        )

    msg_value = (
        f"{rules}"
        f"\n---\n"
        f"GOLD:\n{gold_norm}\n"
        f"\nPRED:\n{pred_norm}\n"
        f"\nYour response must be exactly one word: MATCH or NO MATCH."
    )

    # Framework expects: {'role':'user', 'type':'text', 'value': msg_value}
    return {'role': 'user', 'type': 'text', 'value': msg_value}


def _call_llm_judge(model: Any, task: str, gold_norm: str, pred_norm: str) -> int:
    """
    Call the framework-provided judge model:
        result = model.generate([prompt_dict])
    Return 1 for MATCH, 0 for NO MATCH or parse failure.
    """
    prompt = _build_llm_prompt(task, gold_norm, pred_norm)
    try:
        resp = model.generate([prompt])  # the framework's judge API (messages list)
        # Normalize response to string
        text = _to_str(resp).strip()
        if isinstance(resp, dict) and "text" in resp:
            text = _to_str(resp["text"]).strip()
        elif isinstance(resp, (list, tuple)) and len(resp) > 0:
            text = _to_str(resp[0]).strip()

        head = text[:16].upper()
        if "MATCH" in head and "NO MATCH" not in head:
            return 1
        if head.startswith("MATCH"):
            return 1
        return 0
    except Exception:
        return 0


def em_with_llm(
    answer: Any,
    prediction: Any,
    task: Optional[str],
    model: Optional[Any] = None,
    enable_llm_on_mismatch: bool = True
) -> int:
    """
    Public entry:
    1) Try local canonical EM.
    2) If mismatch AND LLM is enabled AND model is provided, call LLM judge.
    Returns: 0/1.
    """
    gold = _to_str(answer)
    pred = _to_str(prediction)

    em0, g_norm, p_norm = _local_em(gold, pred, task)
    if em0 == 1:
        return 1

    if not enable_llm_on_mismatch or model is None:
        return 0

    return _call_llm_judge(model, task, g_norm, p_norm)
