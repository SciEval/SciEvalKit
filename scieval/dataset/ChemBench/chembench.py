from __future__ import annotations

import json
import math
import os.path
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd
from datasets import get_dataset_config_names, load_dataset

from scieval.smp import load

from ..text_base import TextBaseDataset
from .constant import (
    COT_PROMPT,
    DEFAULT_SPLIT,
    EVAL_RESULT_FILENAME,
    HF_DATASET,
    MCQ_PROMPT_TEMPLATE,
    NUMERIC_PROMPT_TEMPLATE,
)
from .metrics import METRIC_FUNCTIONS, classification_scores, summarise_metric_table
from .parsing import (
    enumerate_options,
    format_options_block,
    looks_like_refusal,
    normalize_target_scores,
    parse_mcq_prediction,
    parse_numeric_prediction,
)


class ChemBench(TextBaseDataset):
    TYPE = "TEXT"
    MODALITY = "TEXT"

    def __init__(
        self,
        dataset: str = "ChemBench",
        source: str = "huggingface",
        data_dir: Optional[str] = None,
        split: str = DEFAULT_SPLIT,
        topics: Optional[Sequence[str]] = None,
        shuffle_options: bool = False,
        use_cot: bool = False,
        random_seed: int = 42,
    ) -> None:
        self.dataset_name = dataset
        normalized_source = source.lower()
        if normalized_source in {"hf", "huggingface"}:
            self.source = "huggingface"
        elif normalized_source in {"dir", "local", "directory"}:
            self.source = "directory"
        else:
            self.source = normalized_source
        self.data_dir = Path(data_dir).expanduser() if data_dir else None
        self.split = split
        self.random_seed = random_seed
        self.topics = list(topics) if topics else None
        self.shuffle_options = shuffle_options
        self.use_cot = use_cot
        self._samples: Dict[str, Dict[str, Any]] = {}
        super().__init__(dataset=dataset)

    @classmethod
    def supported_datasets(cls):
        return ["ChemBench"]

    # ------------------------------------------------------------------
    def load_data(self, dataset: str) -> pd.DataFrame:
        records: List[Dict[str, Any]] = []
        if self.source == "huggingface":
            records = self._load_from_huggingface()
        elif self.source == "directory":
            if not self.data_dir:
                raise ValueError("data_dir must be provided when source='directory'.")
            records = self._load_from_directory(self.data_dir)
        else:
            raise ValueError("source must be either 'huggingface' or 'directory'.")

        if not records:
            raise ValueError("No ChemBench samples were loaded. Check the source settings.")
        return pd.DataFrame(records)

    def post_build(self, dataset: str) -> None:
        self._samples = {}
        for idx in range(len(self.data)):
            row = self.data.iloc[idx].to_dict()
            self._samples[str(row["question_id"])] = row
        self.topics_in_use = sorted(set(self.data["topic"].tolist()))

    # ------------------------------------------------------------------
    def build_prompt(self, line) -> List[Dict[str, str]]:
        if isinstance(line, int):
            line = self.data.iloc[line]
        question = line["question"].strip()
        cot_clause = "\n" + COT_PROMPT if self.use_cot else ""
        if line["question_type"] == "mcq":
            prompt = MCQ_PROMPT_TEMPLATE.format(
                question=question,
                answers=line["options_text"],
                cot=cot_clause,
            )
        else:
            prompt = NUMERIC_PROMPT_TEMPLATE.format(question=question, cot=cot_clause)
        return [dict(type="text", value=prompt)]

    # ------------------------------------------------------------------
    def evaluate(self, eval_file: str, **judge_kwargs) -> Dict[str, Any]:
        predictions = load(eval_file)
        if isinstance(predictions, dict):
            predictions = pd.DataFrame(predictions)
        elif isinstance(predictions, list):
            predictions = pd.DataFrame(predictions)
        if not isinstance(predictions, pd.DataFrame):
            raise TypeError("Predictions must be a path to a JSON/CSV/TSV file or a pandas-compatible object.")
        id_column = self._detect_id_column(predictions.columns)
        if id_column is None:
            raise KeyError("Prediction file must contain a 'question_id' or 'id' column.")
        if "prediction" not in predictions.columns:
            raise KeyError("Prediction file must contain a 'prediction' column with model outputs.")

        per_question: List[Dict[str, Any]] = []
        seen_ids: set[str] = set()
        for _, row in predictions.iterrows():
            question_id = str(row[id_column])
            prediction_text = row["prediction"]
            sample = self._samples.get(question_id)
            if sample is None:
                continue
            seen_ids.add(question_id)
            scored = self._score_prediction(sample, prediction_text)
            per_question.append(scored)

        # Fill unanswered questions so aggregated metrics account for coverage
        for question_id, sample in self._samples.items():
            if question_id in seen_ids:
                continue
            per_question.append(self._score_prediction(sample, "", unanswered=True))

        overall_metrics = summarise_metric_table([record["metrics"] for record in per_question])
        per_topic = {}
        for topic in self.topics_in_use:
            topic_rows = [record["metrics"] for record in per_question if record["topic"] == topic]
            per_topic[topic] = summarise_metric_table(topic_rows)

        accuracy_stats = self._calculate_accuracy_stats(per_question, tolerance=0.01)

        result = {
            "accuracy_stats": accuracy_stats,
            "overall": overall_metrics,
            "per_topic": per_topic,
            "per_question": per_question,
            "answered": len(seen_ids),
            "total_questions": len(self._samples),
            "missing_ids": sorted(set(self._samples) - seen_ids),
        }

        eval_dir = os.path.dirname(eval_file)
        work_dir = Path(judge_kwargs.get("eval_result_dir", eval_dir))
        work_dir.mkdir(parents=True, exist_ok=True)
        output_path = work_dir / EVAL_RESULT_FILENAME
        result["result_path"] = str(output_path)
        serializable = self._json_ready(result)
        output_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
        return result

    def _calculate_accuracy_stats(self, per_question: List[Dict[str, Any]], tolerance: float = 0.01) -> Dict[str, Any]:
        mcq_count = 0
        correct_mcq = 0
        numeric_count = 0
        correct_numeric = 0

        for record in per_question:
            q_type = record.get('question_type')
            parsed = record.get('parsed', {})
            ground_truth = record.get('ground_truth')


            if not all([q_type, parsed, ground_truth]):
                continue

            # --- Evaluate MCQ ---
            if q_type == 'mcq':
                mcq_count += 1
                selected_options = parsed.get('selected_options', [])
                if isinstance(ground_truth, dict) and isinstance(selected_options, list):

                    correct_options = {opt for opt, val in ground_truth.items() if val == 1.0}
                    predicted_options = set(selected_options)

                    if correct_options == predicted_options:
                        correct_mcq += 1

            # --- Evaluate Numeric ---
            elif q_type == 'numeric':
                numeric_count += 1
                predicted_answer = parsed.get('numeric_answer')
                
                if predicted_answer is None:
                    continue

                try:
                    ground_truth_answer = float(ground_truth)
                    allowed_error = tolerance * abs(ground_truth_answer)
                    absolute_difference = abs(predicted_answer - ground_truth_answer)
                    
                    if absolute_difference <= allowed_error:
                        correct_numeric += 1
                except (ValueError, TypeError):
                    continue

        total_correct = correct_mcq + correct_numeric
        total_questions = len(per_question)

        return {
            "mcq_stats": {
                "total": mcq_count,
                "correct": correct_mcq,
                "accuracy": (correct_mcq / mcq_count) if mcq_count > 0 else 0.0
            },
            "numeric_stats": {
                "total": numeric_count,
                "correct": correct_numeric,
                "accuracy": (correct_numeric / numeric_count) if numeric_count > 0 else 0.0,
                "tolerance": tolerance
            },
            "overall_stats": {
                "total_questions_analyzed": total_questions,
                "total_correct": total_correct,
                "accuracy": (total_correct / total_questions) if total_questions > 0 else 0.0
            }
        }

    # ------------------------------------------------------------------
    def _load_from_huggingface(self) -> List[Dict[str, Any]]:
        selected_topics = self.topics or get_dataset_config_names(HF_DATASET)
        rows: List[Dict[str, Any]] = []
        global_index = 0
        for topic in selected_topics:
            dataset = load_dataset(HF_DATASET, topic, split=self.split)
            for task in dataset:
                rows.extend(self._expand_task(topic, task, global_index))
                global_index = len(rows)
        return rows

    def _load_from_directory(self, root: Path) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        global_index = 0
        for json_file in sorted(root.rglob("*.json")):
            topic = json_file.parent.name
            if self.topics and topic not in self.topics:
                continue
            with open(json_file, "r", encoding="utf-8") as handle:
                task = json.load(handle)
            rows.extend(self._expand_task(topic, task, global_index))
            global_index = len(rows)
        return rows

    def _expand_task(self, topic: str, task_record: Dict[str, Any], start_index: int) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        examples = task_record.get("examples", [])
        task_metrics = task_record.get("metrics", [])
        task_uuid = str(task_record.get("uuid", f"{topic}_task"))
        for example_idx, example in enumerate(examples, start=1):
            options_text = ""
            options: Optional[List[Dict[str, object]]] = None
            score_map: Optional[Dict[str, float]] = None
            raw_target_scores = example.get("target_scores")
            if raw_target_scores is not None:
                normalized_scores = normalize_target_scores(raw_target_scores)
                options = enumerate_options(
                    normalized_scores,
                    shuffle=self.shuffle_options,
                    seed=self.random_seed,
                )
                options_text = format_options_block(options)
                score_map = {item["label"]: item["score"] for item in options}
            question_type = "mcq" if score_map else "numeric"
            question_id = f"{topic}.{task_uuid}.{example_idx}"
            target_value = example.get("target")
            numeric_target = None
            if target_value is not None:
                try:
                    numeric_target = float(target_value)
                except (TypeError, ValueError):
                    numeric_target = None
            rows.append(
                dict(
                    index=start_index + len(rows),
                    question_id=question_id,
                    topic=topic,
                    task_uuid=task_uuid,
                    task_name=task_record.get("name", ""),
                    question=str(example.get("input", "")).strip(),
                    question_type=question_type,
                    metrics=list(task_metrics),
                    preferred_score=task_record.get("preferred_score", ""),
                    options=options,
                    options_text=options_text,
                    score_map=score_map,
                    target=target_value,
                    numeric_target=numeric_target,
                    keywords=task_record.get("keywords", []),
                    description=task_record.get("description", ""),
                    example_index=example_idx,
                )
            )
        return rows

    def _score_prediction(self, sample: Dict[str, Any], prediction: Any, unanswered: bool = False) -> Dict[str, Any]:
        prediction_text = "" if prediction is None else str(prediction)
        metrics: Dict[str, Any] = {}
        parsed: Dict[str, Any] = {}
        is_refusal = unanswered or looks_like_refusal(prediction_text)
        ground_truth: Any = None
        if sample["question_type"] == "mcq":
            options = sample.get("options") or []
            allowed = [opt["label"] for opt in options]
            answers = [] if is_refusal else parse_mcq_prediction(prediction_text, allowed)
            parsed["selected_options"] = answers
            parsed["raw"] = prediction_text
            score_map: Dict[str, float] = sample.get("score_map") or {}
            for metric_name in sample["metrics"]:
                metric_fn = METRIC_FUNCTIONS.get(metric_name)
                if metric_fn is None:
                    continue
                metrics[metric_name] = metric_fn(answers, score_map)
            if score_map:
                metrics.update(classification_scores(score_map, answers))
            ground_truth = score_map
        else:
            value, value_span = (None, None) if is_refusal else parse_numeric_prediction(prediction_text)
            parsed["numeric_answer"] = value
            if value_span is not None:
                parsed["answer_span"] = value_span
            parsed["raw"] = prediction_text
            target_text = sample.get("target")
            if (target_text is None or target_text == "") and sample.get("numeric_target") is not None:
                target_text = str(sample.get("numeric_target"))
            for metric_name in sample["metrics"]:
                metric_fn = METRIC_FUNCTIONS.get(metric_name)
                if metric_fn is None:
                    continue
                if metric_name == "exact_str_match":
                    fragment = value_span if value_span is not None else prediction_text
                    metrics[metric_name] = metric_fn(fragment, target_text)
                else:
                    metrics[metric_name] = metric_fn(value, target_text)
            if "all_correct" in sample["metrics"] and target_text is not None:
                metrics["all_correct"] = METRIC_FUNCTIONS["all_correct"](value, target_text)
            ground_truth = target_text
        record = {
            "question_id": sample["question_id"],
            "topic": sample["topic"],
            "task_uuid": sample["task_uuid"],
            "metrics": metrics,
            "parsed": parsed,
            "prediction": prediction_text,
            "question_type": sample["question_type"],
            "is_refusal": is_refusal,
            "ground_truth": ground_truth,
        }
        return record

    @staticmethod
    def _detect_id_column(columns: Iterable[str]) -> Optional[str]:
        for candidate in ("question_id", "id", "index"):
            if candidate in columns:
                return candidate
        return None

    @staticmethod
    def _json_ready(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: ChemBench._json_ready(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [ChemBench._json_ready(v) for v in obj]
        if isinstance(obj, float) and math.isnan(obj):
            return None
        return obj
