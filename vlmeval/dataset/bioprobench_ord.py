import os
import os.path as osp
import re
import ast
from itertools import combinations
from typing import List, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm

from vlmeval import load, dump
from ..smp.file import get_intermediate_file_path
from .text_base import TextBaseDataset


# ------------------ Helper Function ------------------
def extract_predicted_order(generated_str: str, wrong_steps: List[str], correct_steps: List[str]) -> Tuple[List[str], List[str]]:
    """
    Parses model output to reconstruct predicted order.

    Expected format:
        [ANSWER_START][1, 0, 2, ...][ANSWER_END]
    """
    generated_str = generated_str.split("</think>")[-1]
    pattern = r"\[ANSWER_START\](.*?)\[ANSWER_END\]"
    match = re.findall(pattern, generated_str, re.DOTALL)

    if not match:
        raise ValueError("Missing [ANSWER_START]/[ANSWER_END]")

    index_str = match[-1].strip()
    try:
        generated_indices = ast.literal_eval(index_str)
    except Exception:
        raise ValueError(f"Cannot parse step indices: {index_str}")

    # 验证索引合法性
    if not isinstance(generated_indices, list) or not all(isinstance(i, int) for i in generated_indices):
        raise ValueError("Indices must be a list of integers")
    if set(generated_indices) != set(range(len(correct_steps))):
        raise ValueError("Invalid or incomplete index set")

    predicted_steps = [wrong_steps[i] for i in generated_indices]
    return predicted_steps, correct_steps


def calculate_exact_match(gts: List[List[str]], preds: List[List[str]]) -> float:
    """计算 exact match 精度。"""
    correct = sum(gt == pr for gt, pr in zip(gts, preds))
    return correct / len(gts) if gts else 0


def calculate_kendall_tau(gts: List[List[str]], preds: List[List[str]]) -> float:
    """计算平均 Kendall’s Tau，衡量序列一致性。"""
    total_pairs = 0
    concordant_pairs = 0

    for gt, pr in zip(gts, preds):
        gt_rank = {step: i for i, step in enumerate(gt)}
        pr_rank = {step: i for i, step in enumerate(pr)}

        for a, b in combinations(gt_rank.keys(), 2):
            gt_order = gt_rank[a] - gt_rank[b]
            pr_order = pr_rank[a] - pr_rank[b]
            if gt_order * pr_order > 0:  # 一致排序
                concordant_pairs += 1
            total_pairs += 1

    if total_pairs == 0:
        return 0
    return (2 * concordant_pairs - total_pairs) / total_pairs


# ------------------ Main Dataset Class ------------------
class BioProBench_ORD(TextBaseDataset):
    """BioProBench-ORD: step ordering (reconstruction) task."""

    MODALITY = "TEXT"
    TYPE = "QA"
    DATASET_URL = {
        "ORD": "ORD_test.tsv"
    }
    DATASET_MD5 = {
        "ORD": "577acee8270cfb2c19f6bfd6b5d6d7c7"
    }

    # ---------- Prompt Builder ----------
    def build_prompt(self, line):
        """
        构建 BioProBench-ORD 的标准化 Prompt。
        输入可以是 DataFrame 的一行或索引。
        """
        if isinstance(line, int):
            line = self.data.iloc[line]

        question = line["question"]
        wrong_steps = line["wrong_steps"]

        if isinstance(wrong_steps, list):
            wrong_steps_str = "\n".join([f"{i}. {s}" for i, s in enumerate(wrong_steps)])
        else:
            wrong_steps_str = str(wrong_steps)

        prompt = f"""
{question}
The steps are:
{wrong_steps_str}

- Give me the correct order of the steps as a list of their original indices (start from 0), no other words.
- Output your answer *wrapped exactly* between the tags [ANSWER_START] and [ANSWER_END].
- The format of your response must be:
[ANSWER_START]a list of the original indices[ANSWER_END]
""".strip()

        msgs = [{"type": "text", "value": prompt}]
        return msgs

    # ---------- Evaluation Logic ----------
    def evaluate(self, eval_file: str, **judge_kwargs) -> pd.DataFrame:
        """
        Evaluate step ordering (ORD) predictions.

        输入文件需包含：
            'question', 'wrong_steps', 'correct_steps', 'prediction'
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
        assert all(k in data for k in ["wrong_steps", "correct_steps", "prediction"]), \
            "eval_file must contain 'wrong_steps', 'correct_steps', and 'prediction'"

        score_file = get_intermediate_file_path(eval_file, "_score", "csv")
        storage = get_intermediate_file_path(eval_file, "_judge", "json")

        # 若已有缓存则加载
        if osp.exists(storage):
            results = load(storage)
        else:
            preds, gts, failed, total = [], [], 0, 0

            for _, item in tqdm(data.iterrows(), total=len(data), desc="Evaluating BioProBench-ORD"):
                total += 1
                generated_str = str(item["prediction"])
                try:
                    pr, gt = extract_predicted_order(
                        generated_str, item["wrong_steps"], item["correct_steps"]
                    )
                    preds.append(pr)
                    gts.append(gt)
                except Exception:
                    failed += 1

            exact_match = calculate_exact_match(gts, preds)
            kendall_tau = calculate_kendall_tau(gts, preds)

            results = {
                "exact_match": exact_match * 100,
                "kendall_tau": kendall_tau,
                "failed": failed,
                "total": total,
            }
            dump(results, storage)

        # 输出标准 DataFrame
        res = pd.DataFrame({
            "Metric": ["Exact Match (%)", "Kendall Tau"],
            "Value": [results["exact_match"], results["kendall_tau"]],
        })
        dump(res, score_file)
        return res
