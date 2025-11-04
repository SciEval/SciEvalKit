from __future__ import annotations

import warnings
from typing import Any, Dict, List

import pandas as pd
from datasets import load_dataset

from vlmeval import dump, load

from .text_mcq import TextMCQDataset
from .utils import build_judge, DEBUG_MESSAGE
from .utils.multiple_choice import (
    extract_answer_from_item,
    prefetch_answer,
    report_acc,
)
from ..smp.file import get_intermediate_file_path


class ChemBench4K(TextMCQDataset):
    """ChemBench benchmark (text-only multiple choice).

    The dataset is hosted on Hugging Face at ``AI4Chem/ChemBench4K`` and
    provides natural-language prompts paired with four molecular candidates.
    Each record specifies the correct option label (``'A'-'D'``).  This class
    adapts the benchmark to VLMEvalKit's evaluation pipeline.

    Parameters
    ----------
    split : {"train", "validation", "test"}, default "test"
        Split name to load from the Hugging Face dataset.
    """

    TYPE = "MCQ"
    MODALITY = "TEXT"
    dataset_name = "ChemBench4K"

    def __init__(self, split: str = "test", **kwargs: Any) -> None:
        self.split = split
        super().__init__(dataset=self.dataset_name, **kwargs)

    @classmethod
    def supported_datasets(cls) -> List[str]:
        return [cls.dataset_name]

    def load_data(self, dataset: str) -> pd.DataFrame:
        """Load the ChemBench split from Hugging Face and normalise columns."""
        try:
            hf_ds = load_dataset("AI4Chem/ChemBench4K", split=self.split)
        except Exception as err:  # pragma: no cover - surface helpful error
            raise RuntimeError(
                "Failed to load the ChemBench dataset. Ensure 'datasets' is installed "
                "and network access to Hugging Face is available."
            ) from err

        records: List[Dict[str, Any]] = []
        for idx, item in enumerate(hf_ds):
            records.append(
                {
                    "index": idx,
                    "question": str(item.get("question", "")).strip(),
                    "answer": str(item.get("answer", "")).strip().upper(),
                    "A": _to_option(item, "A"),
                    "B": _to_option(item, "B"),
                    "C": _to_option(item, "C"),
                    "D": _to_option(item, "D"),
                    "split": self.split,
                }
            )

        df = pd.DataFrame(records)
        # Ensure option labels are valid (fallback to empty string if missing).
        for opt in ["A", "B", "C", "D"]:
            if opt not in df:
                df[opt] = ""
        df["answer"] = df["answer"].replace({"": pd.NA})

        return df

    # ---- Prompt construction -------------------------------------------------
    def build_prompt(self, line: pd.Series) -> List[Dict[str, str]]:
        if isinstance(line, int):
            line = self.data.iloc[line]

        question = str(line.get("question", "")).strip()
        prompt = (
            "There is a single choice question about chemistry. Answer the question by replying A, B, C, or D.\n"
            f"Question: {question}\n"
            f"A. {str(line.get('A', '')).strip()}\n"
            f"B. {str(line.get('B', '')).strip()}\n"
            f"C. {str(line.get('C', '')).strip()}\n"
            f"D. {str(line.get('D', '')).strip()}\n"
            "Answer: "
        )
        return [dict(type="text", value=prompt)]

    # ---- Evaluation ----------------------------------------------------------
    def evaluate(self, eval_file: str, **judge_kwargs: Any) -> pd.DataFrame:
        """Evaluate predictions with exact matching first, then LLM fallback."""
        judge_kwargs.pop("nproc", None)  # align signature expectation
        judge_name = judge_kwargs.pop("model", "exact_matching")
        judge_model = None
        if judge_name != "exact_matching":
            judge_model = build_judge(model=judge_name, **judge_kwargs)
            if not judge_model.working():
                warnings.warn(
                    "LLM judge failed to initialise; falling back to exact matching only."
                )
                warnings.warn(DEBUG_MESSAGE)
                judge_model = None

        data = load(eval_file)
        if isinstance(data, dict):
            data = pd.DataFrame(data)
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                f"Unsupported evaluation file content type: {type(data).__name__}"
            )

        if "prediction" not in data.columns or "index" not in data.columns:
            raise KeyError("Evaluation file must contain 'index' and 'prediction' columns.")

        data = data.copy()
        data["prediction"] = data["prediction"].fillna("").astype(str)
        if "split" in data.columns:
            data = data.drop(columns=["split"])

        meta_cols = ["index", "question", "A", "B", "C", "D", "answer", "split"]
        meta = self.data[meta_cols].copy()

        merged = pd.merge(data, meta, on="index", how="inner")

        def _normalise_column(df: pd.DataFrame, name: str) -> pd.DataFrame:
            """Ensure merged dataframe has a single column for each meta field."""
            col_x, col_y = f"{name}_x", f"{name}_y"
            if col_x in df.columns and col_y in df.columns:
                df = df.drop(columns=[col_x]).rename(columns={col_y: name})
            elif col_y in df.columns:
                df = df.rename(columns={col_y: name})
            elif col_x in df.columns:
                df = df.rename(columns={col_x: name})
            elif name not in df.columns:
                df[name] = ""
            return df

        for col in ["question", "A", "B", "C", "D", "answer", "split"]:
            merged = _normalise_column(merged, col)

        merged["answer"] = merged["answer"].fillna("").astype(str).str.strip().str.upper()
        merged["split"] = merged["split"].fillna(self.split)

        results: List[Dict[str, Any]] = []
        for _, row in merged.iterrows():
            row_dict = row.to_dict()
            predicted_letter = prefetch_answer(row_dict) or ""
            predicted_letter = str(predicted_letter).strip().upper()
            existing_log = row_dict.get("log", None)
            if existing_log is not None and not pd.isna(existing_log):
                log_text = str(existing_log)
            else:
                log_text = row_dict.get("prediction", "")
            judge_used = False

            if not predicted_letter and judge_model is not None:
                judge_used = True
                res = extract_answer_from_item(
                    judge_model, row_dict, dataset_name=self.dataset_name
                )
                predicted_letter = str(res.get("opt", "")).strip().upper()
                log_text = res.get("log", log_text)

            hit = int(predicted_letter == row_dict.get("answer", ""))
            results.append(
                {
                    "index": row_dict["index"],
                    "split": row_dict.get("split", self.split),
                    "question": row_dict.get("question", ""),
                    "prediction": row_dict.get("prediction", ""),
                    "answer": row_dict.get("answer", ""),
                    "pred_norm": predicted_letter,
                    "hit": hit,
                    "log": log_text,
                    "judge_used": judge_used,
                }
            )

        result_df = pd.DataFrame(results).sort_values("index").reset_index(drop=True)

        # Persist per-sample diagnostics back to the original prediction file.
        export_df = merged.copy()
        diagnostics = result_df.set_index("index")[["pred_norm", "hit", "log", "judge_used"]]
        for col in diagnostics.columns:
            mapped = export_df["index"].map(diagnostics[col])
            if col in export_df.columns:
                export_df[col] = mapped.combine_first(export_df[col])
            else:
                export_df[col] = mapped
        export_df = export_df.sort_values("index").reset_index(drop=True)
        dump(export_df, eval_file)

        score_df = report_acc(result_df)
        score_file = get_intermediate_file_path(eval_file, "_acc", "csv")
        dump(score_df, score_file)

        return score_df


def _to_option(item: Dict[str, Any], key: str) -> str:
    value = item.get(key, "")
    if value is None:
        return ""
    value = str(value).strip()
    return value
