# -*- coding: utf-8 -*-
"""
MaScQA dataset for SciEvalKit
"""

from __future__ import annotations
import os.path as osp
import os
import re
import warnings
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from .text_base import TextBaseDataset

from scieval import load, dump                      # project I/O helpers
from ..smp.file import get_intermediate_file_path   # same helper used by sfebench.py
from .utils.judge_util import build_judge_model           # standard judge builder
from ..utils import track_progress_rich             # multiprocessing helper

# ---------- regex helpers ----------
_LETTER_RE = re.compile(r"\b([ABCD])\b", flags=re.IGNORECASE)
_LETTER_ANY_RE = re.compile(r"[ABCD]", flags=re.IGNORECASE)
_NUM_RE    = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _extract_choice_letters(txt: str) -> set[str]:
    """Extract A/B/C/D letters, giving priority to content inside first [...] block."""
    if not txt:
        return set()
    # If answer enclosed in [ ], focus there first
    m_box = re.search(r"\[(.*?)\]", txt)
    segment = m_box.group(1) if m_box else txt
    segment = segment.upper().replace("ANSWER", " ")
    letters = set(_LETTER_RE.findall(segment))
    letters.update(_LETTER_ANY_RE.findall(segment))
    return {ch for ch in letters}

def _gold_mode(gold_str: str) -> str:
    """Return 'OR' if "OR" present, 'AND' otherwise (for multi-letter)."""
    if re.search(r"\bOR\b", gold_str, flags=re.IGNORECASE):
        return 'OR'
    return 'AND'


def _extract_num(txt: str) -> float | None:
    if not txt:
        return None
    # prefer content inside brackets if present
    m_box = re.search(r"\[(.*?)\]", txt)
    segment = m_box.group(1) if m_box else txt
    m = _NUM_RE.search(segment.replace(",", ""))
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None


def _parse_interval(gold: str) -> tuple[float, float] | None:
    """Detect intervals like '0.8 to 0.9'  /  '0.8:0.9' / '0.8 TO 0.9'."""
    if not gold:
        return None
    g = gold.strip()
    has_sep = (":" in g) or bool(re.search(r"\bto\b", g, flags=re.IGNORECASE))
    nums = re.findall(_NUM_RE, g.replace(",", ""))
    if len(nums) >= 2 and has_sep:
        lo, hi = float(nums[0]), float(nums[1])
        return (min(lo, hi), max(lo, hi))
    return None


def _numeric_match(pred: str, gold: str,
                   rel_tol: float | None,
                   abs_tol: float | None) -> bool:
    """Return True if pred satisfies gold."""
    if "marks to all" in gold.lower():
        return True
    x = _extract_num(pred)
    if x is None:
        return False

    inter = _parse_interval(gold)
    if inter:
        lo, hi = inter
        return lo <= x <= hi

    try:
        y = float(gold.replace(",", ""))
    except Exception:
        return gold.strip().lower() in pred.strip().lower()

    if abs_tol is not None and abs(x - y) <= abs_tol:
        return True
    if rel_tol is not None and y != 0 and abs((x - y) / y) <= rel_tol:
        return True
    return x == y


# ---------- main class ----------
class MaScQA(TextBaseDataset):
    """MaScQA TSV-based dataset (index, qid, subject, qtype, question, answer)."""
    MODALITY = 'TEXT'
    TYPE = 'QA'
    DATASET_URL = {
        'MaScQA': 'https://huggingface.co/datasets/InternScience/SciEval/resolve/main/MaScQA.tsv'
    }
    DATASET_MD5 = {
        'MaScQA': '94e67417d44c5c62c6dec6a8b4112ef7'
    }

    def __init__(self) -> None:
        super().__init__(dataset="MaScQA")
        required = {"index", "qid", "qtype", "question", "answer"}
        if not required.issubset(self.data.columns):
            raise ValueError(f"TSV missing columns: {required - set(self.data.columns)}")

    # ------------- judge a single row -------------
    def _judge(self, row: pd.Series, judge_model=None) -> Dict[str, Any]:
        qtype = str(row["qtype"]).upper()
        gold  = str(row["answer"]).strip()
        pred  = str(row["prediction"])

        result = {"hit": False, "pred_norm": "", "judge_used": False, "log": ""}

        if qtype == "NUM":
            hit = _numeric_match(pred, gold, None, None)
            result.update(hit=hit, pred_norm=str(_extract_num(pred) or ""))
            return result

        # MCQ path
        letters_pred = _extract_choice_letters(pred)
        if not letters_pred and judge_model is not None:
            jp = (
                f"Question: {row['question']}\n"  # original question
                f"Ground truth answer: {gold}\n"
                f"Model answer: {pred}\n"
                "Is the model answer fully correct? Reply with a single word: Yes or No."
            )
            judge_out = judge_model.generate(jp)
            result["judge_used"] = True
            result["log"] = judge_out
            result["hit"] = judge_out.strip().lower().startswith("yes")
            result["pred_norm"] = ",".join(sorted(letters_pred)) if letters_pred else ""
            return result

        letters_gold = _extract_choice_letters(gold)
        if letters_gold:
            mode = _gold_mode(gold)
            if mode == 'OR':
                result["hit"] = bool(letters_pred & letters_gold)
            else:
                result["hit"] = (letters_pred == letters_gold)
        else:
            result["hit"] = gold.lower() in pred.lower()
        result["pred_norm"] = ",".join(sorted(letters_pred)) if letters_pred else ""
        return result

    # ---------- prompt builder override ----------
    def build_prompt(self, line):
        """Return a list of message dicts suitable for chat-based models.
        - MCQS (or any type containing 'MCQS'): ask to select *all* correct letters, comma‑separated.
        - NUM (type contains 'NUM'): ask to provide only the numeric answer.
        Default fallback uses base question.
        """
        if isinstance(line, int):
            line = self.data.iloc[line]

        question = str(line["question"]).strip()
        qtype = str(line.get("qtype", "")).upper()

        if "NUM" in qtype:
            prompt_text = (
                f"{question}\n\n"
                "Provide ONLY the numeric answer enclosed in square brackets. "
                "Example: [42.0]"
            )
        else:  # MCQ / MATCH
            prompt_text = (
                f"{question}\n\n"
                "Select ALL correct options. Return ONLY the option letters, separated by commas, "
                "and enclose them in square brackets. Example: [A,B]"
            )

        return [dict(type="text", value=prompt_text)]

    # ------------- public evaluate -------------
    def evaluate(self, eval_file: str, **judge_kwargs):
        """
        Parameters
        ----------
        eval_file : str
            Path to pkl/json in which 'prediction' column already exists.
        judge_kwargs : Any
            Passed to build_judge; example: {'model': 'gpt-4o-1120', 'nproc': 4}
        """
        # Resolve eval_file loading based on extension
        if eval_file.lower().endswith(('.xlsx', '.xls')):
            data = pd.read_excel(eval_file)
        elif eval_file.lower().endswith('.csv'):
            data = pd.read_csv(eval_file)
        else:
            data = load(eval_file)

        assert {'prediction', 'answer', 'qtype'}.issubset(data.columns)

        # ensure str
        data['prediction'] = [str(x) for x in data['prediction']]
        data['answer']     = [str(x) for x in data['answer']]

        storage  = get_intermediate_file_path(eval_file, '_judge')
        tmp_file = get_intermediate_file_path(eval_file, '_tmp', 'pkl')

        # Ensure directories exist
        os.makedirs(osp.dirname(storage), exist_ok=True)
        os.makedirs(osp.dirname(tmp_file), exist_ok=True)

        nproc    = judge_kwargs.pop('nproc', 4)

        if not osp.exists(storage):
            # existing partial cache
            ans_map = {} if not osp.exists(tmp_file) else load(tmp_file)

            # optional judge model
            judge_name = judge_kwargs.pop('model', 'exact_matching')
            judge_model = None
            if judge_name != 'exact_matching':
                judge_model = build_judge_model(model=judge_name, **judge_kwargs)
                if not judge_model.working():
                    warnings.warn("Judge model unavailable; falling back to exact matching.")
                    judge_model = None

            # lines still requiring judgment
            lines   = [data.iloc[i] for i in range(len(data)) if data.iloc[i]['index'] not in ans_map]
            indices = [x['index'] for x in lines]

            def _worker(self_ref, line):
                return self_ref._judge(line, judge_model)

            if lines:
                jobs = [(self, line) for line in lines]
                outs = track_progress_rich(_worker, jobs, nproc=nproc, chunksize=nproc)
                for idx, res in zip(indices, outs):
                    ans_map[idx] = res
                dump(ans_map, tmp_file)

            # attach results
            data['hit'] = [ans_map[x]['hit'] for x in data['index']]
            data['log'] = [ans_map[x]['log'] for x in data['index']]
            dump(data, storage)

        # reload judged data
        data = load(storage)

        # simple accuracy report
        acc = np.mean(data['hit']) * 100.0
        score_df = pd.DataFrame({'MaScQA-Acc(%)': [acc]})

        score_file = get_intermediate_file_path(eval_file, '_score', 'csv')
        dump(score_df, score_file)

        return score_df