import os
import os.path as osp
import re
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import brier_score_loss

from vlmeval import load, dump                      # project I/O helpers
from ..smp.file import get_intermediate_file_path   # consistent path helper
from .text_base import TextBaseDataset


# ------------------ Helper Function ------------------
def extract_answer_and_confidence(generated_str: str) -> Tuple[str, int]:
    """提取 [ANSWER_START]...[ANSWER_END] 中的答案与置信度"""
    if '</think>' in generated_str:
        generated_str = generated_str.split("</think>")[-1]

    pattern = r"\[ANSWER_START\](.*?)\[ANSWER_END\]"
    match = re.search(pattern, generated_str, re.DOTALL)
    if not match:
        raise ValueError("Missing [ANSWER_START] or [ANSWER_END]")

    content = match.group(1).strip()

    # answer & confidence 分离
    if '&' in content:
        parts = content.split('&')
    else:
        parts = content.split(' ')
        parts = [' '.join(parts[:-1]), parts[-1]]

    answer = parts[0].strip()
    conf_match = re.search(r"\d+", parts[-1])
    if not conf_match:
        raise ValueError("Confidence value not found")
    confidence = int(conf_match.group())

    if confidence > 100:
        raise ValueError("Confidence cannot exceed 100")

    return answer, confidence


# ------------------ Main Dataset Class ------------------
class BioProBench_PQA(TextBaseDataset):
    """BioProBench-PQA: biological protocol QA dataset."""

    MODALITY = 'TEXT'
    TYPE = 'QA'
    DATASET_URL = {
        'PQA': 'PQA_test.tsv'
    }
    DATASET_MD5 = {
        'PQA': 'add60d251de2eded937c7739a0fada92'
    }

    # ---------- prompt builder override ----------
    def build_prompt(self, line):
        """
        构建 BioProBench-PQA 的标准化 Prompt。
        输入可以是 DataFrame 的一行或索引。
        输出为供模型调用的消息列表。
        """
        if isinstance(line, int):
            line = self.data.iloc[line]

        # === 提取字段 ===
        question = line['question']
        choices = line['choices']
        if isinstance(choices, (list, tuple)):
            choices_str = "\n".join([f"- {c}" for c in choices])
        else:
            choices_str = str(choices)

        # === 组装官方 prompt ===
        prompt = f"""
You will be given a multiple-choice question related to a biological protocol. The blank in the question (represented as '____') indicates where the correct choice should be filled in.

Question:
{question}

Choices:
{choices_str}

Your task:
- Choose the most likely correct answer from the given choices.
- You must always select *one* answer, even if you are unsure.
- The selected answer must match one of the choices exactly (including case and punctuation).
- Assign a confidence score between 0 and 100 based on your certainty.
- Output your answer *wrapped exactly* between the tags [ANSWER_START] and [ANSWER_END].
- The format of your response must be:
[ANSWER_START]your selected choice & your confidence score[ANSWER_END]
""".strip()

        msgs = [{"type": "text", "value": prompt}]
        return msgs

    # ---------- evaluation logic ----------
    def evaluate(self, eval_file: str, **judge_kwargs) -> pd.DataFrame:
        """
        Evaluate BioProBench-PQA model predictions.

        输入：
            eval_file: JSON / TSV / PKL 文件，包含字段：
                'question', 'answer', 'choices', 'prediction'
        输出：
            DataFrame，包含 Accuracy (%) 与 Brier Score。
        """

        # ===== 载入预测结果 =====
        data = load(eval_file)

        # ---- 统一 DataFrame 格式 ----
        if isinstance(data, list):
            # JSON list → DataFrame
            data = pd.DataFrame(data)
        elif isinstance(data, dict):
            # JSON dict → 尝试从键取出主要部分
            if all(isinstance(v, list) for v in data.values()):
                data = pd.DataFrame(data)
            else:
                raise ValueError(f"Unsupported JSON dict structure in {eval_file}")

        assert isinstance(data, pd.DataFrame), \
            f"Unsupported data type {type(data)}, must be DataFrame or convertible list."

        assert 'answer' in data and 'prediction' in data, \
            "eval_file must contain 'answer' and 'prediction'"

        # ===== 路径管理 =====
        score_file = get_intermediate_file_path(eval_file, '_score', 'csv')
        storage = get_intermediate_file_path(eval_file, '_judge', 'json')

        # ===== 若已有缓存结果，直接返回 =====
        if osp.exists(storage):
            results = load(storage)
        else:
            accs, cfds, failed, total = [], [], 0, 0

            for _, item in tqdm(data.iterrows(), total=len(data), desc="Evaluating BioProBench-PQA"):
                total += 1
                generated_str = str(item['prediction'])
                try:
                    answer, confidence = extract_answer_and_confidence(generated_str)
                    accs.append(1 if answer.strip() == str(item['answer']).strip() else 0)
                    cfds.append(confidence)
                except Exception:
                    failed += 1
                    accs.append(0)
                    cfds.append(0)

            accuracy = np.mean(accs) * 100 if len(accs) > 0 else 0
            brier = brier_score_loss(accs, np.array(cfds) / 100) if len(accs) > 0 else None

            results = {
                'accuracy': accuracy,
                'brier_score': brier,
                'failed': failed,
                'total': total
            }
            dump(results, storage)

        # ===== 输出标准 DataFrame =====
        res = pd.DataFrame({
            'Metric': ['Accuracy (%)', 'Brier Score'],
            'Value': [results['accuracy'], results['brier_score']]
        })
        dump(res, score_file)
        return res

