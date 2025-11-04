import os
import os.path as osp
import re
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm

from vlmeval import load, dump
from ..smp.file import get_intermediate_file_path
from .text_base import TextBaseDataset


# ------------------ Helper Function ------------------
def extract_binary_answer(generated_str: str) -> bool:
    """
    从模型输出中提取 True/False。
    格式示例：[ANSWER_START]True[ANSWER_END]
    """
    if '</think>' in generated_str:
        generated_str = generated_str.split("</think>")[-1]
    if '[/INST]' in generated_str:
        generated_str = generated_str.split("[/INST]")[-1]

    # 提取 [ANSWER_START]...[ANSWER_END]
    pattern = r"\[ANSWER_START\](.*?)\[ANSWER_END\]"
    match = re.search(pattern, generated_str, re.DOTALL)

    if match:
        answer = match.group(1).strip()
    else:
        # fallback: 取最后一行
        last_line = generated_str.strip().split('\n')[-1].strip()
        answer = last_line

    if answer.lower() == "true":
        return True
    elif answer.lower() == "false":
        return False
    else:
        raise ValueError(f"Invalid answer format: {answer}")


# ------------------ Main Dataset Class ------------------
class BioProBench_ERR(TextBaseDataset):
    """BioProBench-ERR: protocol error recognition dataset (True/False)."""

    MODALITY = 'TEXT'
    TYPE = 'QA'
    DATASET_URL = {
        'ERR': 'ERR_test.tsv'
    }
    DATASET_MD5 = {
        'ERR': '016b15ac93033b9bfdaf667a7f4edcac'
    }

    # ---------- Prompt Builder ----------
    def build_prompt(self, line):
        """
        构建 ERR 任务的标准化 Prompt。
        输入可以是 DataFrame 的一行或索引。
        """
        if isinstance(line, int):
            line = self.data.iloc[line]

        step = line["step"]
        context = line["context"]

        prompt = f"""
Determine whether the following target step in a protocol is True or False:
{step}

You may use the following context, which includes the purpose of the step, as well as the preceding and following steps, to inform your decision:
{context}

Please carefully evaluate if the step is logically consistent, necessary, and accurate in the context. If you find anything wrong, answer False.

- Please respond with only True or False, without any additional explanation.
- Output your answer *wrapped exactly* between the tags [ANSWER_START] and [ANSWER_END].
- The format of your response must be:
[ANSWER_START]True or False[ANSWER_END]
""".strip()

        msgs = [{"type": "text", "value": prompt}]
        return msgs

    # ---------- Evaluation Logic ----------
    def evaluate(self, eval_file: str, **judge_kwargs) -> pd.DataFrame:
        """
        Evaluate ERR (True/False) task predictions.

        输入文件需包含：
            'step', 'context', 'is_correct', 'prediction'
        """
        data = load(eval_file)

        # ---- 类型标准化 ----
        if isinstance(data, list):
            data = pd.DataFrame(data)
        elif isinstance(data, dict):
            if all(isinstance(v, list) for v in data.values()):
                data = pd.DataFrame(data)
            else:
                raise ValueError(f"Unsupported JSON dict structure in {eval_file}")

        assert isinstance(data, pd.DataFrame)
        assert 'is_correct' in data and 'prediction' in data, \
            "eval_file must contain 'is_correct' and 'prediction'"

        score_file = get_intermediate_file_path(eval_file, '_score', 'csv')
        storage = get_intermediate_file_path(eval_file, '_judge', 'json')

        # 若已有缓存则加载
        if osp.exists(storage):
            results = load(storage)
        else:
            preds, gts, failed, total = [], [], 0, 0
            for _, item in tqdm(data.iterrows(), total=len(data), desc="Evaluating BioProBench-ERR"):
                total += 1
                generated_str = str(item["prediction"])
                try:
                    pred = extract_binary_answer(generated_str)
                    gt = bool(item["is_correct"])
                    preds.append(pred)
                    gts.append(gt)
                except Exception:
                    failed += 1

            # 计算指标
            TP = sum((p is False and g is False) for p, g in zip(preds, gts))
            FP = sum((p is False and g is True) for p, g in zip(preds, gts))
            FN = sum((p is True and g is False) for p, g in zip(preds, gts))

            accuracy = sum(p == g for p, g in zip(preds, gts)) / len(preds) if preds else 0
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            results = {
                "accuracy": accuracy * 100,
                "precision": precision * 100,
                "recall": recall * 100,
                "f1": f1 * 100,
                "failed": failed,
                "total": total
            }
            dump(results, storage)

        # 输出 DataFrame
        res = pd.DataFrame({
            "Metric": ["Accuracy (%)", "Precision (%)", "Recall (%)", "F1 (%)"],
            "Value": [results["accuracy"], results["precision"], results["recall"], results["f1"]]
        })
        dump(res, score_file)
        return res
