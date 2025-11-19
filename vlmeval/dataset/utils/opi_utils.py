import re
import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import MultiLabelBinarizer
from rouge_score import rouge_scorer

def calculate_metrics(output, target):
    """
    Calculate evaluation metrics for classification tasks

    Args:
        output: predicted labels (list or str)
        target: ground truth labels (list or str)

    Returns:
        tuple: (accuracy, precision, recall, f1)
    """
    mlb = MultiLabelBinarizer(classes=sorted(set(output + target)))
    y_true = mlb.fit_transform([target])
    y_pred = mlb.transform([output])

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='micro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='micro', zero_division=0)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0

    return accuracy, precision, recall, f1

def calculate_rouge_l(output, target):
    """
    Calculate ROUGE-L score

    Args:
        output: predicted text (str or list)
        target: ground truth text (str or list)

    Returns:
        float: ROUGE-L F-measure score
    """
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    if isinstance(output, list):
        output_text = " ".join(output)
    else:
        output_text = str(output)

    if isinstance(target, list):
        target_text = " ".join(target)
    else:
        target_text = str(target)

    scores = scorer.score(target_text, output_text)
    return scores['rougeL'].fmeasure

def extract_prediction_from_output(output_text):
    """
    Extract the actual prediction from model output, removing CoT reasoning if present

    Args:
        output_text: raw model output text

    Returns:
        str: extracted prediction
    """
    if not output_text:
        return ""

    think_end = output_text.find('</think>')
    if think_end == -1:
        return output_text.strip()

    prediction = output_text[think_end + len('</think>'):].strip()
    return prediction

def parse_multilabel_string(text):
    """
    Parse a string containing multiple labels separated by delimiters

    Args:
        text: input string with multiple labels

    Returns:
        list: list of parsed labels
    """
    if isinstance(text, str):
        labels = [label.strip() for label in re.split(r'[;,，；]\s*', text) if label.strip()]
    else:
        labels = [str(text)]

    return labels

def compute_accuracy_fold_type(predictions, references):
    """
    Compute accuracy for fold type prediction task

    Args:
        predictions: list of predicted fold types
        references: list of ground truth fold types

    Returns:
        float: accuracy score
    """
    correct = 0
    total = len(predictions)

    for pred, ref in zip(predictions, references):
        if str(pred).strip() == str(ref).strip():
            correct += 1

    return correct / total if total > 0 else 0
