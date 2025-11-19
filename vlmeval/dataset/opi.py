import warnings
import pandas as pd
import re
from abc import abstractmethod
from sklearn.metrics import matthews_corrcoef, accuracy_score, precision_score, recall_score
from ..smp import *
from .text_base import TextBaseDataset
from .utils.opi_utils import (
    calculate_metrics,
    calculate_rouge_l,
    extract_prediction_from_output,
    parse_multilabel_string,
    compute_accuracy_fold_type
)

class OPI(TextBaseDataset):
    TYPE = 'TEXT'
    DATASET_URL = {
        'EC_number_CLEAN_EC_number_new': '/OPI_tsv/EC_number_CLEAN_EC_number_new_test.tsv',
        'EC_number_CLEAN_EC_number_price': '/OPI_tsv/EC_number_CLEAN_EC_number_price_test.tsv',
        'Fold_type_fold_type': '/OPI_tsv/Fold_type_fold_type_test.tsv',
        'Function_CASPSimilarSeq_function': '/OPI_tsv/Function_CASPSimilarSeq_function_test.tsv',
        'Function_IDFilterSeq_function': '/OPI_tsv/Function_IDFilterSeq_function_test.tsv',
        'Function_UniProtSeq_function': '/OPI_tsv/Function_UniProtSeq_function_test.tsv',
        'gName2Cancer_gene_name_to_cancer': '/OPI_tsv/gName2Cancer_gene_name_to_cancer_test.tsv',
        'GO_CASPSimilarSeq_go': '/OPI_tsv/GO_CASPSimilarSeq_go_test.tsv',
        'GO_IDFilterSeq_go': '/OPI_tsv/GO_IDFilterSeq_go_test.tsv',
        'GO_UniProtSeq_go': '/OPI_tsv/GO_UniProtSeq_go_test.tsv',
        'gSymbol2Cancer_gene_symbol_to_cancer': '/OPI_tsv/gSymbol2Cancer_gene_symbol_to_cancer_test.tsv',
        'gSymbol2Tissue_gene_symbol_to_tissue': '/OPI_tsv/gSymbol2Tissue_gene_symbol_to_tissue_test.tsv',
        'Keywords_CASPSimilarSeq_keywords': '/OPI_tsv/Keywords_CASPSimilarSeq_keywords_test.tsv',
        'Keywords_IDFilterSeq_keywords': '/OPI_tsv/Keywords_IDFilterSeq_keywords_test.tsv',
        'Keywords_UniProtSeq_keywords': '/OPI_tsv/Keywords_UniProtSeq_keywords_test.tsv',
        'Subcellular_localization_subcell_loc': '/OPI_tsv/Subcellular_localization_subcell_loc_test.tsv',
    }

    DATASET_MD5 = {
        'EC_number_CLEAN_EC_number_new': '',
        'EC_number_CLEAN_EC_number_price': '',
        'Fold_type_fold_type': '',
        'Function_CASPSimilarSeq_function': '',
        'Function_IDFilterSeq_function': '',
        'Function_UniProtSeq_function': '',
        'gName2Cancer_gene_name_to_cancer': '',
        'GO_CASPSimilarSeq_go': '',
        'GO_IDFilterSeq_go': '',
        'GO_UniProtSeq_go': '',
        'gSymbol2Cancer_gene_symbol_to_cancer': '',
        'gSymbol2Tissue_gene_symbol_to_tissue': '',
        'Keywords_CASPSimilarSeq_keywords': '',
        'Keywords_IDFilterSeq_keywords': '',
        'Keywords_UniProtSeq_keywords': '',
        'Subcellular_localization_subcell_loc': '',
    }

    function_tasks = [
        'Function_CASPSimilarSeq_function',
        'Function_IDFilterSeq_function',
        'Function_UniProtSeq_function'
    ]

    subcellular_tasks = ['Subcellular_localization_subcell_loc']

    fold_type_tasks = ['Fold_type_fold_type']

    multilabel_tasks = [
        'EC_number_CLEAN_EC_number_new',
        'EC_number_CLEAN_EC_number_price',
        'GO_CASPSimilarSeq_go',
        'GO_IDFilterSeq_go',
        'GO_UniProtSeq_go',
        'Keywords_CASPSimilarSeq_keywords',
        'Keywords_IDFilterSeq_keywords',
        'Keywords_UniProtSeq_keywords',
        'gSymbol2Tissue_gene_symbol_to_tissue',
        'gSymbol2Cancer_gene_symbol_to_cancer',
        'gName2Cancer_gene_name_to_cancer'
    ]

    def process_function_task(self, task_entries):
        """Process function description task using ROUGE-L"""
        rouge_scores = []

        for index, entry in task_entries.iterrows():
            prediction = extract_prediction_from_output(str(entry.get("prediction", "")))
            label = str(entry.get("answer", ""))

            rouge_l = calculate_rouge_l(prediction, label)
            rouge_scores.append(rouge_l)

        return rouge_scores

    def process_subcellular_task(self, task_entries):
        """Process subcellular localization task using accuracy"""
        accuracies = []

        for index, entry in task_entries.iterrows():
            prediction = extract_prediction_from_output(str(entry.get("prediction", "")))
            label = str(entry.get("answer", ""))

            pred_list = [prediction] if isinstance(prediction, str) else prediction
            label_list = [label] if isinstance(label, str) else label

            accuracy, _, _, _ = calculate_metrics(pred_list, label_list)
            accuracies.append(accuracy)

        return accuracies

    def process_fold_type_task(self, task_entries):
        """Process fold type prediction task using accuracy"""
        predictions = []
        references = []

        for index, entry in task_entries.iterrows():
            prediction = extract_prediction_from_output(str(entry.get("prediction", "")))
            label = str(entry.get("answer", ""))

            predictions.append(prediction.strip())
            references.append(label.strip())

        accuracy = compute_accuracy_fold_type(predictions, references)
        return [accuracy] * len(predictions)

    def process_multilabel_task(self, task_entries):
        """Process multi-label tasks (EC_number, GO, Keywords, etc.)"""
        precisions = []
        recalls = []
        f1_scores = []

        for index, entry in task_entries.iterrows():
            prediction = extract_prediction_from_output(str(entry.get("prediction", "")))
            label = str(entry.get("answer", ""))

            pred_list = parse_multilabel_string(prediction)
            label_list = parse_multilabel_string(label)

            if label_list:
                _, precision, recall, f1 = calculate_metrics(pred_list, label_list)
                precisions.append(precision)
                recalls.append(recall)
                f1_scores.append(f1)

        return precisions, recalls, f1_scores

    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        """
        Evaluate predictions for OPI tasks

        Args:
            eval_file: path to the evaluation file containing predictions
            **judge_kwargs: additional arguments

        Returns:
            dict: evaluation metrics
        """
        data = load(eval_file)
        data = data[~pd.isna(data["prediction"])]

        assert 'answer' in data and 'prediction' in data

        dataset_name = None
        for name in self.DATASET_URL:
            if name in eval_file:
                dataset_name = name
                break

        if dataset_name is None:
            return {"error": "Unknown dataset"}

        if dataset_name in self.function_tasks:
            rouge_scores = self.process_function_task(data)
            mean_rouge = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0
            return {"ROUGE-L": round(mean_rouge, 4)}

        elif dataset_name in self.subcellular_tasks:
            accuracies = self.process_subcellular_task(data)
            mean_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0
            return {"Accuracy": round(mean_accuracy, 4)}

        elif dataset_name in self.fold_type_tasks:
            accuracies = self.process_fold_type_task(data)
            mean_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0
            return {"Accuracy": round(mean_accuracy, 4)}

        elif dataset_name in self.multilabel_tasks:
            precisions, recalls, f1_scores = self.process_multilabel_task(data)

            mean_precision = sum(precisions) / len(precisions) if precisions else 0
            mean_recall = sum(recalls) / len(recalls) if recalls else 0
            mean_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0

            return {
                "Precision": round(mean_precision, 4),
                "Recall": round(mean_recall, 4),
                "F1": round(mean_f1, 4)
            }

        else:
            return {"error": f"Unsupported task: {dataset_name}"}
